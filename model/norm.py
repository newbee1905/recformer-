import torch
import torch.nn as nn
import torch.nn.functional as F


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


from kernel.rms_norm import TritonRMSNorm


def get_norm_class(config):
	"""
	Returns the appropriate RMSNorm class based on the configuration.
	"""
	use_triton = getattr(config, "use_liger_norm", False)
	if use_triton:
		# The TritonRMSNorm is hardcoded for Gemma-style normalization
		# with a unit offset, which is compatible.
		return TritonRMSNorm
	else:
		# RMSNormTorch requires the `dim` argument.
		# We can return a lambda to make the signature compatible or
		# just return the class and let the caller handle it.
		# Let's check compatibility. Both take hidden_size/dim as the first arg.
		return RMSNormTorch
