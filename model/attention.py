import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import apply_rope, apply_rope_torch
from .norm import get_norm_class


class MHA(nn.Module):
	"""
	Multi-Head Attention (MHA) with:
	- KV Cache
	- Gated Attention (G1).
	- RoPE
	- Cross-Attention
	"""

	def __init__(self, config, is_decoder: bool = False):
		super().__init__()
		assert config.d_model % config.n_head == 0
		self.config = config
		self.n_head = config.n_head
		self.d_model = config.d_model
		self.d_head = config.d_model // self.n_head
		self.max_seq_len = config.block_size
		self.use_kv_cache = config.use_kv_cache and is_decoder
		self.use_qk_norm = config.use_qk_norm
		self.use_gate = config.use_gate
		self.is_decoder = is_decoder
		self.use_liger_rope = config.use_liger_rope

		# Key, query, value projections
		self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
		self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
		self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
		self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
		self.resid_dropout = nn.Dropout(config.dropout)

		if self.use_gate:
			self.g_gate = nn.Linear(config.d_model, config.d_model, bias=False)

		if self.use_qk_norm:
			RMSNorm = get_norm_class(config)
			self.q_norm = RMSNorm(self.d_head)
			self.k_norm = RMSNorm(self.d_head)

	def forward(
		self,
		hidden_states,
		freqs_cos,
		freqs_sin,
		layer_past=None,
		encoder_hidden_states=None,
		is_causal=False,
	):
		bsz, seq_len, _ = hidden_states.size()

		# Cross-attention
		is_cross_attention = encoder_hidden_states is not None
		if is_cross_attention:
			q = self.q_proj(hidden_states)
			k = self.k_proj(encoder_hidden_states)
			v = self.v_proj(encoder_hidden_states)
		# Self-attention
		else:
			q = self.q_proj(hidden_states)
			k = self.k_proj(hidden_states)
			v = self.v_proj(hidden_states)

		q = q.reshape(bsz, seq_len, self.n_head, self.d_head).transpose(1, 2)
		k = k.reshape(bsz, -1, self.n_head, self.d_head).transpose(1, 2)
		v = v.reshape(bsz, -1, self.n_head, self.d_head).transpose(1, 2)

		seq_len_past = 0
		if self.use_kv_cache and layer_past is not None:
			seq_len_past = layer_past[0].size(2)

		# Apply RoPE for self-attention
		if not is_cross_attention:
			total_seq_len = seq_len + seq_len_past
			if self.use_liger_rope:
				q, k = apply_rope(q, k, freqs_cos, freqs_sin, seq_len=total_seq_len)
			else:
				q, k = apply_rope_torch(q, k, freqs_cos, freqs_sin, seq_len=total_seq_len)

		if self.use_qk_norm:
			q = self.q_norm(q)
			k = self.k_norm(k)

		if self.use_kv_cache and layer_past is not None:
			past_k, past_v = layer_past
			k = torch.cat([past_k, k], dim=2)
			v = torch.cat([past_v, v], dim=2)

		present = (k, v) if self.use_kv_cache else None

		seq_len_kv = k.size(2)

		y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal and seq_len == seq_len_kv)

		if self.use_gate:
			gate_score = torch.sigmoid(self.g_gate(hidden_states))
			gate_score = gate_score.view(bsz, seq_len, self.n_head, self.d_head).transpose(1, 2)
			y = y * gate_score

		y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

		output = self.c_proj(y)
		output = self.resid_dropout(output)

		return output, present


class DisentangledSelfAttention(nn.Module):
	"""
	Disentangled self-attention mechanism, inspired by DeBERTaV3.
	This implementation is for the encoder and uses F.scaled_dot_product_attention.
	"""

	def __init__(self, config):
		super().__init__()
		assert config.d_model % config.n_head == 0
		self.config = config
		self.n_head = config.n_head
		self.d_head = config.d_model // self.n_head
		self.use_qk_norm = getattr(config, "use_qk_norm", False)

		self.q_proj = nn.Linear(config.d_model, config.d_model, bias=True)
		self.k_proj = nn.Linear(config.d_model, config.d_model, bias=True)
		self.v_proj = nn.Linear(config.d_model, config.d_model, bias=True)
		self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
		self.resid_dropout = nn.Dropout(config.dropout)

		if self.use_qk_norm:
			RMSNorm = get_norm_class(config)
			self.q_norm = RMSNorm(self.d_head)
			self.k_norm = RMSNorm(self.d_head)

		# Relative position embeddings
		self.max_relative_positions = getattr(config, "max_relative_positions", 512)
		self.pos_embed = nn.Embedding(self.max_relative_positions * 2, self.d_head)
		self.pos_dropout = nn.Dropout(config.dropout)

	def get_relative_positional_bias(self, q, k, rel_pos_embeds):
		"""Computes the relative positional bias terms."""
		# B, H, L, D
		bsz, n_head, seq_len, d_head = q.size()
		rel_pos_embeds = rel_pos_embeds.view(seq_len, seq_len, d_head)

		# Content-to-Position
		c2p_score = torch.einsum("bhld,ijd->bhij", q, rel_pos_embeds)

		# Position-to-Content
		# We need to flip the relative positions for this term
		p2c_embeds = torch.flip(rel_pos_embeds, dims=[0])
		p2c_score = torch.einsum("bhjd,ijd->bhij", k, p2c_embeds)

		return c2p_score + p2c_score

	def forward(self, hidden_states, attention_mask=None):
		bsz, seq_len, _ = hidden_states.size()

		q = self.q_proj(hidden_states)
		k = self.k_proj(hidden_states)
		v = self.v_proj(hidden_states)

		q = q.reshape(bsz, seq_len, self.n_head, self.d_head).transpose(1, 2)
		k = k.reshape(bsz, seq_len, self.n_head, self.d_head).transpose(1, 2)
		v = v.reshape(bsz, seq_len, self.n_head, self.d_head).transpose(1, 2)

		if self.use_qk_norm:
			q = self.q_norm(q)
			k = self.k_norm(k)

		# Relative Position Handling
		rel_pos_indices = torch.arange(seq_len, device=hidden_states.device)
		rel_pos_mat = rel_pos_indices.unsqueeze(1) - rel_pos_indices.unsqueeze(0)
		rel_pos_mat += self.max_relative_positions
		rel_pos_mat = torch.clamp(rel_pos_mat, 0, 2 * self.max_relative_positions - 1)

		rel_pos_embeds = self.pos_embed(rel_pos_mat)
		rel_pos_embeds = self.pos_dropout(rel_pos_embeds)

		# Get positional bias and add it to the attention mask
		positional_bias = self.get_relative_positional_bias(q, k, rel_pos_embeds)
		if attention_mask is not None:
			# The mask is broadcastable to (B, H, L, L)
			positional_bias += attention_mask

		attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=positional_bias, is_causal=False)

		attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

		output = self.c_proj(attn_output)
		output = self.resid_dropout(output)

		return output, None
