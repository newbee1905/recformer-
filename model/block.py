import torch
import torch.nn as nn

from .attention import MHA
from .feed_forward import FeedForward
from .norm import get_norm_class


class TransformerBlock(nn.Module):
	"""A single Transformer decoder block."""

	def __init__(self, config):
		super().__init__()
		RMSNorm = get_norm_class(config)

		self.self_attn_norm = RMSNorm(config.d_model)
		self.self_attn = MHA(config, is_decoder=True)
		self.self_attn_dropout = nn.Dropout(config.dropout)
		self.self_attn_layerscale = nn.Parameter(config.layer_scale_init * torch.ones(config.d_model))

		self.ffn_norm = RMSNorm(config.d_model)
		if config.use_liger_ff:
			self.ffn = FeedForward(config)
		else:
			from .feed_forward import FeedForwardTorch

			self.ffn = FeedForwardTorch(config)
		self.ffn_dropout = nn.Dropout(config.dropout)
		self.ffn_layerscale = nn.Parameter(config.layer_scale_init * torch.ones(config.d_model))

	def forward(self, x, freqs_cos, freqs_sin, layer_past=None):
		# Self-attention
		self_attn_norm = self.self_attn_norm(x)
		self_attn_out, present = self.self_attn(
			self_attn_norm, freqs_cos, freqs_sin, layer_past=layer_past, is_causal=True
		)
		x = x + self.self_attn_dropout(self_attn_out * self.self_attn_layerscale)

		# FFN
		ffn_norm = self.ffn_norm(x)
		ffn_out = self.ffn(ffn_norm)
		x = x + self.ffn_dropout(ffn_out * self.ffn_layerscale)

		return x, present
