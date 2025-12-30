import torch
import torch.nn as nn

from .block import TransformerBlock
from .norm import get_norm_class
from .utils import precompute_freqs_cis


class BlockStack(nn.Module):
	"""A stack of Transformer blocks."""

	def __init__(self, config):
		super().__init__()
		self.config = config
		self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
		self.norm = get_norm_class(config)(config.d_model)

	def forward(self, x, freqs_cos, freqs_sin, layer_pasts=None):
		new_layer_pasts = [] if self.config.use_kv_cache else None

		for i, layer in enumerate(self.layers):
			layer_past = layer_pasts[i] if layer_pasts is not None else None
			x, present = layer(x, freqs_cos, freqs_sin, layer_past=layer_past)

			if self.config.use_kv_cache:
				new_layer_pasts.append(present)

		return self.norm(x), new_layer_pasts


class Transformer(nn.Module):
	"""A standalone decoder-only Transformer model."""

	def __init__(self, config):
		super().__init__()
		self.config = config

		self.embed = nn.Embedding(config.vocab_size, config.d_model)
		self.drop = nn.Dropout(config.dropout)
		self.transformer = BlockStack(config)
		self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

		if not getattr(config, "tie_word_embeddings", True):
			self.lm_head.weight = self.embed.weight

		# Precompute RoPE frequencies
		head_dim = config.d_model // config.n_head
		freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, config.block_size, config.rope_theta)
		self.register_buffer("freqs_cos", freqs_cos, persistent=False)
		self.register_buffer("freqs_sin", freqs_sin, persistent=False)

		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, input_ids, layer_pasts=None):
		x = self.embed(input_ids)
		x = self.drop(x)

		transformer_output, new_layer_pasts = self.transformer(
			x,
			self.freqs_cos,
			self.freqs_sin,
			layer_pasts=layer_pasts,
		)

		logits = self.lm_head(transformer_output)

		return logits, new_layer_pasts
