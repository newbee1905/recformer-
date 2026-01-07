import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# RMSNorm from gemma
# https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
class RMSNormTorch(nn.Module):
	def __init__(
		self,
		dim: int,
		eps: float = 1e-6,
		add_unit_offset: bool = True,
	):
		super().__init__()
		self.eps = eps
		self.add_unit_offset = add_unit_offset
		self.weight = nn.Parameter(torch.zeros(dim))

	def _norm(self, x):
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

	def forward(self, x):
		# Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
		# See https://github.com/huggingface/transformers/pull/29402
		output = self._norm(x.float())
		if self.add_unit_offset:
			output = output * (1 + self.weight.float())
		else:
			output = output * self.weight.float()
		return output.type_as(x)


class SoftHyperballNorm(nn.Module):
	def __init__(self, dim: int, c: float = None):  # Default c to None
		super().__init__()
		self.dim = dim  # Store dim
		self.c = c if c is not None else math.sqrt(dim)  # Calculate if None

	def forward(self, x):
		"""Algebraic (Slope-1) -> Projects to Soft Hyperball (Solid)."""
		vector_norm = x.norm(p=2, dim=-1, keepdim=True)
		scale = torch.rsqrt(1 + (vector_norm / self.c).pow(2))
		return x * scale


def get_qknorm_class(config):
	"""
	Returns the appropriate QKNorm class and its config based on the configuration.
	"""
	if not hasattr(config, "qk_norm") or not hasattr(config.qk_norm, "type"):
		raise ValueError("Configuration for 'use_qk_norm: true' requires a 'qk_norm' block with a 'type'.")

	norm_type = config.qk_norm.type
	if norm_type == "rms":
		return RMSNormTorch, {}
	elif norm_type == "soft_hyperball_norm":
		c_val = config.qk_norm.get("c", None)
		return SoftHyperballNorm, {"c": c_val}
	else:
		raise ValueError(f"Unknown QK norm type: {norm_type}")


_TritonRMSNorm = None


def get_norm_class(config):
	"""
	Returns the appropriate RMSNorm class based on the configuration.
	"""
	global _TritonRMSNorm
	use_triton = getattr(config, "use_liger_norm", False)
	if use_triton:
		if _TritonRMSNorm is None:
			try:
				from kernel.rms_norm import TritonRMSNorm as _TritonRMSNorm_

				_TritonRMSNorm = _TritonRMSNorm_
			except ImportError:
				raise ImportError("Triton is not available. Set `use_liger_norm=False` to use torch RMSNorm.")
		# The TritonRMSNorm is hardcoded for Gemma-style normalization
		# with a unit offset, which is compatible.
		return _TritonRMSNorm
	else:
		# RMSNormTorch requires the `dim` argument.
		# We can return a lambda to make the signature compatible or
		# just return the class and let the caller handle it.
		# Let's check compatibility. Both take hidden_size/dim as the first arg.
		return RMSNormTorch
