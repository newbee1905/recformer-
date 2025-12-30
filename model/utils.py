import torch
import torch.nn as nn
import torch.nn.functional as F

TritonRoPEFunction = None


class AttributeEncoder(nn.Module):
	"""
	Fourier Feature MLP to encode float attributes (e.g., MW, LogP).
	Projects scalars to high-dim vector space.
	"""

	def __init__(self, num_props, d_model):
		super().__init__()
		# Random Gaussian matrix for Fourier projection
		self.register_buffer("B", torch.randn(num_props, d_model // 2) * 2.0 * np.pi)
		self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

	def forward(self, x):
		# x: [bsz, n_props]
		# Fourier Projection: [bsz, n_props] @ [n_props, d_model / 2] -> [bsz, d_model / 2]
		x_proj = x @ self.B

		# Concatenate Sin and Cos -> [bsz, d_model]
		x_embed = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

		# MLP and reshape to [bsz, 1, d_model] for attention
		return self.mlp(x_embed).unsqueeze(1)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
	"""
	Precomputes the cosine and sine components for Rotary Positional Embedding (RoPE).
	Returns freqs_cos and freqs_sin, both of shape (end, dim // 2).
	"""
	# Calculation of theta_i: freqs = 1 / (theta^(2i/dim))
	freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

	t = torch.arange(end, device=freqs.device)
	freqs = torch.outer(t, freqs).float()

	# cos(m * theta), sin(m * theta)
	return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
	q: torch.Tensor,
	k: torch.Tensor,
	freqs_cos: torch.Tensor,
	freqs_sin: torch.Tensor,
	seq_len: int,
):
	"""
	Applies the Rotary Positional Embedding (RoPE) to the query and key tensors.
	"""
	global TritonRoPEFunction
	if TritonRoPEFunction is None:
		try:
			from kernel.rope import TritonRoPEFunction as _TritonRoPEFunction

			TritonRoPEFunction = _TritonRoPEFunction
		except ImportError:
			raise ImportError("Triton is not available for RoPE. Set `use_liger_rope=False` to use torch RoPE.")

	T = q.size(2)

	freqs_cos = freqs_cos[seq_len - T : seq_len].contiguous()
	freqs_sin = freqs_sin[seq_len - T : seq_len].contiguous()
	freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
	freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)

	q_out, k_out = TritonRoPEFunction.apply(q, k, freqs_cos.to(q.dtype), freqs_sin.to(q.dtype))

	return q_out.type_as(q), k_out.type_as(k)


def apply_rope_torch(
	q: torch.Tensor,
	k: torch.Tensor,
	freqs_cos: torch.Tensor,
	freqs_sin: torch.Tensor,
	seq_len: int,
):
	"""
	Applies the Rotary Positional Embedding (RoPE) to the query and key tensors using pure torch.
	"""
	query_len, head_dim = q.shape[-2], q.shape[-1]
	# Get the correct slice of freqs
	freqs_cos = freqs_cos[seq_len - query_len : seq_len].to(q.device)
	freqs_sin = freqs_sin[seq_len - query_len : seq_len].to(q.device)

	# freqs_cos, freqs_sin are now [query_len, head_dim // 2]
	# Reshape for broadcasting: [1, 1, query_len, head_dim // 2]
	freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)
	freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)

	# Reshape q and k to view the last dim as pairs of features
	# (bs, num_heads, seq_len, head_dim) -> (bs, num_heads, seq_len, head_dim//2, 2)
	q_reshaped = q.float().reshape(*q.shape[:-1], -1, 2)
	k_reshaped = k.float().reshape(*k.shape[:-1], -1, 2)

	# Apply rotation to pairs of features
	# [x, y] * cos + [-y, x] * sin = [x*cos - y*sin, y*cos + x*sin]
	q_x, q_y = q_reshaped.unbind(-1)
	k_x, k_y = k_reshaped.unbind(-1)

	q_rotated_x = q_x * freqs_cos - q_y * freqs_sin
	q_rotated_y = q_y * freqs_cos + q_x * freqs_sin

	k_rotated_x = k_x * freqs_cos - k_y * freqs_sin
	k_rotated_y = k_y * freqs_cos + k_x * freqs_sin

	# Combine back:
	q_out = torch.stack([q_rotated_x, q_rotated_y], dim=-1).flatten(-2)
	k_out = torch.stack([k_rotated_x, k_rotated_y], dim=-1).flatten(-2)

	return q_out.type_as(q), k_out.type_as(k)
