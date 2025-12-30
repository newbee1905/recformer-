"""
This file centralizes the Triton autotuning configurations for the custom kernels.
The configurations are designed to provide good performance for an A5000 GPU.
"""

import triton


def get_general_autotune_configs():
	"""
	Returns a list of general-purpose Triton configs for autotuning on A5000.
	"""
	return [
		# Configs for H100/A100
		triton.Config({"BLOCK_SIZE": 4096, "num_warps": 8}, num_stages=3),
		triton.Config({"BLOCK_SIZE": 2048, "num_warps": 8}, num_stages=3),
		# Configs for A5000 (Ampere architecture, 64 SMs)
		triton.Config({"BLOCK_SIZE": 2048, "num_warps": 8}, num_stages=3),
		triton.Config({"BLOCK_SIZE": 1024, "num_warps": 4}, num_stages=4),
		# Configs for L40S/3090/V100
		triton.Config({"BLOCK_SIZE": 2048, "num_warps": 4}, num_stages=4),
		triton.Config({"BLOCK_SIZE": 1024, "num_warps": 4}, num_stages=4),
		# Configs for low-end/laptop GPUs (e.g., 1650)
		triton.Config({"BLOCK_SIZE": 1024, "num_warps": 2}, num_stages=2),
		triton.Config({"BLOCK_SIZE": 512, "num_warps": 2}, num_stages=2),
		triton.Config({"BLOCK_SIZE": 256, "num_warps": 2}, num_stages=2),
	]


def get_rope_autotune_configs():
	"""
	Returns a list of Triton configs specifically for RoPE kernel autotuning on A5000.
	The BLOCK_SIZE values are smaller as RoPE operates on smaller head dimensions.
	"""
	return [
		# Configs for H100/A100
		triton.Config({"BLOCK_SIZE": 128, "num_warps": 8}, num_stages=3),
		triton.Config({"BLOCK_SIZE": 64, "num_warps": 8}, num_stages=3),
		# Configs for A5000 (Ampere architecture, 64 SMs)
		triton.Config({"BLOCK_SIZE": 64, "num_warps": 4}, num_stages=3),
		triton.Config({"BLOCK_SIZE": 32, "num_warps": 2}, num_stages=4),
		# Configs for L40S/3090/V100
		triton.Config({"BLOCK_SIZE": 64, "num_warps": 4}, num_stages=4),
		triton.Config({"BLOCK_SIZE": 32, "num_warps": 4}, num_stages=4),
		# Configs for low-end/laptop GPUs (e.g., 1650)
		triton.Config({"BLOCK_SIZE": 32, "num_warps": 2}, num_stages=2),
		triton.Config({"BLOCK_SIZE": 16, "num_warps": 2}, num_stages=2),
	]
