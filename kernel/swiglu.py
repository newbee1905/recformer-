"""
This Triton kernel is an optimized implementation of the SwiGLU activation function,
inspired by the version in liger-kernel (Apache License, Version 2.0).
It includes custom autotune configurations for a range of GPUs (H100, L40S, V100, 1650, etc.),
ensuring efficient performance across different hardware.
"""

import torch
import triton
import triton.language as tl

from .config import get_general_autotune_configs


@triton.jit
def silu(x):
	return x * tl.sigmoid(x)


@triton.autotune(
	configs=get_general_autotune_configs(),
	key=["n_cols"],
)
@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
	program_id = tl.program_id(0)

	a_ptr += program_id * stride
	b_ptr += program_id * stride
	c_ptr += program_id * stride

	col_offsets = tl.arange(0, BLOCK_SIZE)
	mask = col_offsets < n_cols

	a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
	b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
	c_row = (silu(a_row) * b_row).to(a_ptr.dtype.element_ty)
	tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.autotune(
	configs=get_general_autotune_configs(),
	key=["n_cols"],
)
@triton.jit
def _swiglu_backward_kernel(
	dc_ptr, a_ptr, b_ptr, da_ptr, db_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
	program_id = tl.program_id(0)

	dc_ptr += program_id * stride
	a_ptr += program_id * stride
	b_ptr += program_id * stride
	da_ptr += program_id * stride
	db_ptr += program_id * stride

	col_offsets = tl.arange(0, BLOCK_SIZE)
	mask = col_offsets < n_cols

	dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
	a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
	b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0).to(tl.float32)

	sig_a = tl.sigmoid(a_row)
	silu_a = a_row * sig_a

	db_row = dc_row * silu_a
	da_row = dc_row * (silu_a * (1.0 - sig_a) + sig_a) * b_row

	tl.store(da_ptr + col_offsets, da_row.to(a_ptr.dtype.element_ty), mask=mask)
	tl.store(db_ptr + col_offsets, db_row.to(b_ptr.dtype.element_ty), mask=mask)


class TritonSwiGLUFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, a, b):
		ori_shape = a.shape
		n_cols = ori_shape[-1]

		a = a.reshape(-1, n_cols).contiguous()
		b = b.reshape(-1, n_cols).contiguous()
		c = torch.empty_like(a)
		n_rows = a.shape[0]

		grid = (n_rows,)
		_swiglu_forward_kernel[grid](a, b, c, a.stride(0), n_cols=n_cols)

		ctx.save_for_backward(a, b)
		ctx.ori_shape = ori_shape
		return c.reshape(ori_shape)

	@staticmethod
	def backward(ctx, dc):
		a, b = ctx.saved_tensors
		ori_shape = ctx.ori_shape
		n_cols = ori_shape[-1]

		dc = dc.reshape(-1, n_cols).contiguous()
		da = torch.empty_like(a)
		db = torch.empty_like(b)
		n_rows = dc.shape[0]

		grid = (n_rows,)
		_swiglu_backward_kernel[grid](dc, a, b, da, db, dc.stride(0), n_cols=n_cols)
		return da.reshape(ori_shape), db.reshape(ori_shape)
