import torch
import torch.nn as nn

from .transformer import BlockStack
from .utils import precompute_freqs_cis


class RecurrentDecoder(nn.Module):
	"""A recurrent decoder-only model."""

	def __init__(self, config):
		super().__init__()
		self.config = config

		self.embed = nn.Embedding(config.vocab_size, config.d_model)
		self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

		if not getattr(config, "tie_word_embeddings", True):
			self.lm_head.weight = self.embed.weight

		self.drop = nn.Dropout(config.dropout)

		self.transformer = BlockStack(config)

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

	def _forward_step(self, input_ids, hidden_state=None, layer_pasts=None):
		x = self.embed(input_ids)

		# If a hidden state is provided, combine it with the input embeddings.
		# The state is detached from the previous step (Truncated BPTT).
		if hidden_state is not None:
			x = x + hidden_state.detach()

		# Pass through transformer
		transformer_output, new_layer_pasts = self.transformer(
			x,
			self.freqs_cos,
			self.freqs_sin,
			layer_pasts=layer_pasts,
		)

		new_hidden_state = transformer_output
		logits = self.lm_head(new_hidden_state)

		return logits, new_hidden_state, new_layer_pasts

	def forward(self, input_ids, layer_pasts=None):
		hidden_state = None
		logits = None

		for i in range(self.config.num_recurrences):
			logits, hidden_state, layer_pasts = self._forward_step(
				input_ids, hidden_state=hidden_state, layer_pasts=layer_pasts
			)

		return logits, hidden_state, layer_pasts
