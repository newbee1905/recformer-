"""
This Triton kernel provides an optimized implementation of the Rotary Positional Embedding (RoPE) operation,
drawing inspiration from the liger-kernel implementation (Apache License, Version 2.0).
It is designed for high performance and includes autotuning configurations for a wide range of GPUs,
from high-end datacenter cards like H100 to consumer-grade GPUs like the 1650.
"""

import torch
import triton
import triton.language as tl

from .config import get_rope_autotune_configs


@triton.autotune(
	configs=get_rope_autotune_configs(),
	key=["n_qh", "n_kh", "hd"],
)
@triton.jit
def _triton_rope_kernel(
	q_ptr,
	k_ptr,
	cos_ptr,
	sin_ptr,
	q_row_stride,
	k_row_stride,
	cos_row_stride,
	sin_row_stride,
	seq_len,
	cos_bs: tl.constexpr,
	n_qh: tl.constexpr,
	n_kh: tl.constexpr,
	hd: tl.constexpr,
	BACKWARD_PASS: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
):
	pid = tl.program_id(0)

	# Pointers to the current token's data
	q_ptr += pid * q_row_stride
	k_ptr += pid * k_row_stride

	# Determine batch and sequence indices
	batch_idx = pid // seq_len
	seq_idx = pid % seq_len

	# Pointers to sin/cos for the current token position
	cos_ptr += tl.where(
		cos_bs == 1, seq_idx * cos_row_stride, batch_idx * (seq_len * cos_row_stride) + seq_idx * cos_row_stride
	)
	sin_ptr += tl.where(
		cos_bs == 1, seq_idx * sin_row_stride, batch_idx * (seq_len * sin_row_stride) + seq_idx * sin_row_stride
	)

	cos_offsets = tl.arange(0, hd // 2)
	cos_row = tl.load(cos_ptr + cos_offsets, mask=cos_offsets < hd // 2, other=0.0)
	sin_row = tl.load(sin_ptr + cos_offsets, mask=cos_offsets < hd // 2, other=0.0)

	# Iterate over heads
	max_heads = tl.maximum(n_qh, n_kh)
	for h_idx in range(0, tl.cdiv(max_heads, BLOCK_SIZE)):
		head_offsets = h_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
		q_head_mask = head_offsets < n_qh
		k_head_mask = head_offsets < n_kh

		# --- Process Q ---
		q_offsets_1 = head_offsets[:, None] * hd + tl.arange(0, hd // 2)[None, :]
		q_offsets_2 = q_offsets_1 + hd // 2
		q_mask = q_head_mask[:, None]

		q1 = tl.load(q_ptr + q_offsets_1, mask=q_mask, other=0.0)
		q2 = tl.load(q_ptr + q_offsets_2, mask=q_mask, other=0.0)

		if BACKWARD_PASS:
			new_q1 = q1 * cos_row[None, :] + q2 * sin_row[None, :]
			new_q2 = q2 * cos_row[None, :] - q1 * sin_row[None, :]
		else:
			new_q1 = q1 * cos_row[None, :] - q2 * sin_row[None, :]
			new_q2 = q2 * cos_row[None, :] + q1 * sin_row[None, :]

		tl.store(q_ptr + q_offsets_1, new_q1, mask=q_mask)
		tl.store(q_ptr + q_offsets_2, new_q2, mask=q_mask)

		# --- Process K ---
		k_offsets_1 = head_offsets[:, None] * hd + tl.arange(0, hd // 2)[None, :]
		k_offsets_2 = k_offsets_1 + hd // 2
		k_mask = k_head_mask[:, None]

		k1 = tl.load(k_ptr + k_offsets_1, mask=k_mask, other=0.0)
		k2 = tl.load(k_ptr + k_offsets_2, mask=k_mask, other=0.0)

		if BACKWARD_PASS:
			new_k1 = k1 * cos_row[None, :] + k2 * sin_row[None, :]
			new_k2 = k2 * cos_row[None, :] - k1 * sin_row[None, :]
		else:
			new_k1 = k1 * cos_row[None, :] - k2 * sin_row[None, :]
			new_k2 = k2 * cos_row[None, :] + k1 * sin_row[None, :]

		tl.store(k_ptr + k_offsets_1, new_k1, mask=k_mask)
		tl.store(k_ptr + k_offsets_2, new_k2, mask=k_mask)


class TritonRoPEFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, q, k, cos, sin):
		q_orig_shape, k_orig_shape = q.shape, k.shape
		# Transpose to (bsz, seq_len, n_head, d_head) for memory layout
		q = q.transpose(1, 2)
		k = k.transpose(1, 2)

		bsz, seq_len, n_qh, hd = q.shape
		n_kh = k.shape[2]
		cos_bs = cos.shape[0]

		q = q.contiguous()
		k = k.contiguous()
		cos = cos.contiguous()
		sin = sin.contiguous()

		grid = (bsz * seq_len,)
		_triton_rope_kernel[grid](
			q,
			k,
			cos,
			sin,
			q.stride(1),
			k.stride(1),
			cos.stride(-2),
			sin.stride(-2),
			seq_len,
			cos_bs=cos_bs,
			n_qh=n_qh,
			n_kh=n_kh,
			hd=hd,
			BACKWARD_PASS=False,
		)

		ctx.save_for_backward(cos, sin)
		ctx.q_shape = q_orig_shape
		ctx.k_shape = k_orig_shape
		return q.transpose(1, 2).reshape(q_orig_shape), k.transpose(1, 2).reshape(k_orig_shape)

	@staticmethod
	def backward(ctx, dq, dk):
		cos, sin = ctx.saved_tensors
		q_shape, k_shape = ctx.q_shape, ctx.k_shape

		dq = dq.transpose(1, 2).contiguous()
		dk = dk.transpose(1, 2).contiguous()

		bsz, seq_len, n_qh, hd = dq.shape
		n_kh = dk.shape[2]
		cos_bs = cos.shape[0]

		grid = (bsz * seq_len,)
		_triton_rope_kernel[grid](
			dq,
			dk,
			cos,
			sin,
			dq.stride(1),
			dk.stride(1),
			cos.stride(-2),
			sin.stride(-2),
			seq_len,
			cos_bs=cos_bs,
			n_qh=n_qh,
			n_kh=n_kh,
			hd=hd,
			BACKWARD_PASS=True,
		)

		return dq.transpose(1, 2).reshape(q_shape), dk.transpose(1, 2).reshape(k_shape), None, None, None, None
