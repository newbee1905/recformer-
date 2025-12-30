import torch.nn as nn
import torch.nn.functional as F
from kernel.swiglu import TritonSwiGLUFunction


class FeedForward(nn.Module):
	"""
	SwiGLU-based Feed-Forward Network (FFN) using Liger Kernel.
	Implements: (input @ W_gate) * SiLU(input @ W_up) @ W_down
	"""

	def __init__(self, config):
		super().__init__()
		d_model = config.d_model
		intermediate_size = int(d_model * config.ffn_multiplier)

		# Linear layers for SwiGLU (3 separate projections)
		self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
		self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
		self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x):
		output = self.down_proj(TritonSwiGLUFunction.apply(self.up_proj(x), self.gate_proj(x)))
		output = self.dropout(output)

		return output


class FeedForwardTorch(nn.Module):
	"""
	SwiGLU-based Feed-Forward Network (FFN) using pure Torch.
	Implements: (input @ W_gate) * SiLU(input @ W_up) @ W_down
	"""

	def __init__(self, config):
		super().__init__()
		d_model = config.d_model
		intermediate_size = int(d_model * config.ffn_multiplier)

		# Linear layers for SwiGLU (3 separate projections)
		self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
		self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
		self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x):
		gate = self.gate_proj(x)
		up = self.up_proj(x)
		output = self.down_proj(F.silu(gate) * up)
		output = self.dropout(output)
		return output
