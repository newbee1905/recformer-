"""
This Triton kernel is based on the RMSNorm implementation in liger-kernel (Apache License, Version 2.0),
with custom autotune configurations for different classes of GPUs (H100, L40S, V100, 1650, etc.).
It is simplified to only support Gemma-style RMSNorm with a hardcoded offset of 1.0.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.libdevice import rsqrt

from .config import get_general_autotune_configs


@triton.autotune(
	configs=get_general_autotune_configs(),
	key=["n_cols"],
)
@triton.jit
def _rms_norm_fwd_kernel(
	y_ptr,
	x_ptr,
	weight_ptr,
	rstd_ptr,
	y_row_stride,
	x_row_stride,
	n_rows,
	n_cols,
	eps,
	elementwise_affine: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
	BLOCK_ROW: tl.constexpr,
):
	row_idx = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
	col_offsets = tl.arange(0, BLOCK_SIZE)
	row_mask = row_idx < n_rows
	col_mask = col_offsets < n_cols

	x_row = tl.load(
		x_ptr + row_idx[:, None] * x_row_stride + col_offsets[None, :],
		mask=row_mask[:, None] & col_mask[None, :],
		other=0.0,
	)
	x_dtype = x_row.dtype
	x_row = x_row.to(tl.float32)

	if elementwise_affine:
		weight_row = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

	mean_square = tl.sum(x_row * x_row, axis=1) / n_cols
	rstd = rsqrt(mean_square + eps)
	tl.store(rstd_ptr + row_idx, rstd, mask=row_mask)

	normed_x = x_row * rstd[:, None]

	if elementwise_affine:
		y_row = normed_x * (1.0 + weight_row)[None, :]
	else:
		y_row = normed_x

	tl.store(
		y_ptr + row_idx[:, None] * y_row_stride + col_offsets[None, :],
		y_row.to(x_dtype),
		mask=row_mask[:, None] & col_mask[None, :],
	)


@triton.autotune(
	configs=get_general_autotune_configs(),
	key=["n_cols"],
)
@triton.jit
def _rms_norm_bwd_kernel(
	dy_ptr,
	dx_ptr,
	x_ptr,
	weight_ptr,
	rstd_ptr,
	dweight_ptr,
	dy_row_stride,
	dx_row_stride,
	x_row_stride,
	dweight_row_stride,
	n_rows,
	n_cols,
	elementwise_affine: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
	BLOCK_ROW: tl.constexpr,
):
	pid = tl.program_id(0)
	num_sms = tl.num_programs(0)
	col_offsets = tl.arange(0, BLOCK_SIZE)
	col_mask = col_offsets < n_cols

	if elementwise_affine:
		dweight_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
		weight_row = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
		weight_row += 1.0

	for start_row in range(pid * BLOCK_ROW, n_rows, num_sms * BLOCK_ROW):
		row_idx = start_row + tl.arange(0, BLOCK_ROW)
		row_mask = row_idx < n_rows

		dy_row = tl.load(
			dy_ptr + row_idx[:, None] * dy_row_stride + col_offsets[None, :],
			mask=row_mask[:, None] & col_mask[None, :],
			other=0.0,
		).to(tl.float32)
		x_row = tl.load(
			x_ptr + row_idx[:, None] * x_row_stride + col_offsets[None, :],
			mask=row_mask[:, None] & col_mask[None, :],
			other=0.0,
		).to(tl.float32)
		rstd_row = tl.load(rstd_ptr + row_idx, mask=row_mask, other=0.0)

		m = dy_row * weight_row[None, :] if elementwise_affine else dy_row
		dx_row = rstd_row[:, None] * m
		dx_row -= (
			(rstd_row[:, None] * rstd_row[:, None] * rstd_row[:, None] / n_cols)
			* tl.sum(m * x_row, axis=1)[:, None]
			* x_row
		)

		if elementwise_affine:
			normed_x = x_row * rstd_row[:, None]
			dweight_row += tl.sum(dy_row * normed_x, axis=0)

		tl.store(
			dx_ptr + row_idx[:, None] * dx_row_stride + col_offsets[None, :],
			dx_row.to(dy_ptr.dtype.element_ty),
			mask=row_mask[:, None] & col_mask[None, :],
		)
	if elementwise_affine:
		tl.store(dweight_ptr + pid * dweight_row_stride + col_offsets, dweight_row, mask=col_mask)


class _TritonRMSNorm(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, weight, eps, elementwise_affine):
		shape = x.shape
		x = x.reshape(-1, shape[-1])
		n_rows, n_cols = x.shape

		y = torch.empty_like(x)
		rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)

		grid = (triton.cdiv(n_rows, 16),)
		_rms_norm_fwd_kernel[grid](
			y,
			x,
			weight,
			rstd,
			y.stride(0),
			x.stride(0),
			n_rows,
			n_cols,
			eps,
			elementwise_affine=elementwise_affine,
			BLOCK_ROW=16,
		)

		ctx.save_for_backward(x, weight, rstd)
		ctx.elementwise_affine = elementwise_affine
		return y.reshape(shape)

	@staticmethod
	def backward(ctx, dy):
		x, weight, rstd = ctx.saved_tensors
		shape = dy.shape
		dy = dy.reshape(-1, shape[-1])
		n_rows, n_cols = dy.shape

		dx = torch.empty_like(x)

		sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
		dweight = torch.zeros((sm_count, n_cols), dtype=torch.float32, device=x.device)

		grid = (sm_count,)
		_rms_norm_bwd_kernel[grid](
			dy,
			dx,
			x,
			weight,
			rstd,
			dweight,
			dy.stride(0),
			dx.stride(0),
			x.stride(0),
			dweight.stride(0),
			n_rows,
			n_cols,
			elementwise_affine=ctx.elementwise_affine,
			BLOCK_ROW=64,
		)
		dweight = dweight.sum(0).to(weight.dtype) if ctx.elementwise_affine else None
		return dx.reshape(shape), dweight, None, None


class TritonRMSNorm(nn.Module):
	def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
		super().__init__()
		self.hidden_size = hidden_size
		self.eps = eps
		self.elementwise_affine = elementwise_affine

		if self.elementwise_affine:
			self.weight = nn.Parameter(torch.zeros(hidden_size))
		else:
			self.register_parameter("weight", None)

	def forward(self, x):
		return _TritonRMSNorm.apply(x, self.weight, self.eps, self.elementwise_affine)

	def extra_repr(self):
		return (
			f"hidden_size={self.hidden_size}, eps={self.eps}, "
			f"elementwise_affine={self.elementwise_affine} (Gemma-style with offset=1.0)"
		)
