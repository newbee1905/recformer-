import torch
from torch import Tensor
from typing import List, Dict, Any
import torch.distributed as dist


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
	"""
	Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
	Uses a quintic iteration to maximize the slope at zero.
	"""
	assert G.ndim >= 2
	a, b, c = (3.4445, -4.7750, 2.0315)
	X = G.to(torch.float32)

	if G.size(-2) > G.size(-1):
		X = X.mT

	X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

	for _ in range(steps):
		A = X @ X.mT
		B = b * A + c * A @ A
		X = a * X + B @ X

	if G.size(-2) > G.size(-1):
		X = X.mT
	return X


def adamw_update_kernel(
	p: Tensor,
	grad: Tensor,
	exp_avg: Tensor,
	exp_avg_sq: Tensor,
	lr: float,
	wd: float,
	beta1: float,
	beta2: float,
	eps: float,
	step: int,
):
	"""
	Compiled AdamW kernel for dense updates.
	"""

	if wd != 0:
		p.mul_(1 - lr * wd)

	# Decay the first and second moment running average coefficient
	exp_avg.lerp_(grad, 1 - beta1)
	exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

	# Bias Correction calculations
	bias_correction1 = 1 - beta1**step
	bias_correction2 = 1 - beta2**step

	step_size = lr / bias_correction1
	bias_correction2_sqrt = bias_correction2**0.5

	# Denom = sqrt(v_t) / sqrt(1-beta2^t) + eps
	denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

	# p = p - step_size * (m_t / denom)
	p.addcdiv_(exp_avg, denom, value=-step_size)


class Muon(torch.optim.Optimizer):
	"""
	Single-device Muon with integrated Dense AdamW support.
	"""

	def __init__(
		self,
		param_groups: List[Dict[str, Any]],
		lr=0.02,
		weight_decay=0.01,
		momentum=0.95,
		nesterov=True,
		ns_steps=5,
		adam_betas=(0.8, 0.95),
		adam_eps=1e-8,
	):
		for group in param_groups:
			assert "use_muon" in group, "Each param_group must have a 'use_muon' flag."
			if group["use_muon"]:
				group.setdefault("lr", lr)
				group.setdefault("momentum", momentum)
				group.setdefault("weight_decay", weight_decay)
				group.setdefault("nesterov", nesterov)
				group.setdefault("ns_steps", ns_steps)
			else:
				group.setdefault("lr", lr * 0.01)
				group.setdefault("betas", adam_betas)
				group.setdefault("eps", adam_eps)
				group.setdefault("weight_decay", weight_decay)

		super().__init__(param_groups, {})

	@torch.compile
	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			for p in group["params"]:
				if p.grad is None:
					continue

				state = self.state[p]
				if group["use_muon"]:
					if "momentum_buffer" not in state:
						state["momentum_buffer"] = torch.zeros_like(p.grad)

					buf = state["momentum_buffer"]
					buf.lerp_(p.grad, 1 - group["momentum"])
					g = p.grad.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

					original_shape = p.shape
					if p.ndim > 2:
						g = g.view(g.size(0), -1)

					g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

					if group["weight_decay"] > 0:
						p.mul_(1 - group["lr"] * group["weight_decay"])

					scale = max(1.0, g.size(-2) / g.size(-1)) ** 0.5
					p.add_(g.view(original_shape), alpha=-group["lr"] * scale)
				else:
					if "exp_avg" not in state:
						state["exp_avg"] = torch.zeros_like(p)
						state["exp_avg_sq"] = torch.zeros_like(p)
						state["step"] = 0

					state["step"] += 1
					adamw_update_kernel(
						p,
						p.grad,
						state["exp_avg"],
						state["exp_avg_sq"],
						group["lr"],
						group["weight_decay"],
						group["betas"][0],
						group["betas"][1],
						group["eps"],
						state["step"],
					)
		return loss


class DistMuon(torch.optim.Optimizer):
	"""
	Distributed Muon with integrated Dense AdamW.
	"""

	def __init__(
		self,
		param_groups,
		lr=0.02,
		weight_decay=0.01,
		momentum=0.95,
		nesterov=True,
		ns_steps=5,
		adam_betas=(0.8, 0.95),
		adam_eps=1e-8,
	):
		defaults = dict(
			lr=lr,
			weight_decay=weight_decay,
			momentum=momentum,
			nesterov=nesterov,
			ns_steps=ns_steps,
			adam_betas=adam_betas,
			adam_eps=adam_eps,
		)

		# Re-organize params to separate Muon vs AdamW, but keep track of source group settings
		adamw_groups = []
		muon_params_info = []  # Stores (param, shape, settings_dict)

		for group in param_groups:
			assert "use_muon" in group, "Each param_group must have a 'use_muon' flag."

			if group["use_muon"]:
				# Extract settings for this specific group to preserve them after reshaping
				g_settings = {
					"lr": group.get("lr", lr),
					"weight_decay": group.get("weight_decay", weight_decay),
					"momentum": group.get("momentum", momentum),
					"nesterov": group.get("nesterov", nesterov),
					"ns_steps": group.get("ns_steps", ns_steps),
				}

				for p in group["params"]:
					muon_params_info.append((p, p.shape, g_settings))
			else:
				# Handle AdamW groups
				group.setdefault("betas", adam_betas)
				group.setdefault("eps", adam_eps)
				if "lr" not in group:
					group["lr"] = lr * 0.01

				adamw_groups.append(group)

		# Bucket Muon params by shape
		# Sort by numel for deterministic ordering across ranks
		muon_params_info.sort(key=lambda x: x[0].numel(), reverse=True)

		unique_shapes = sorted(list({x[1] for x in muon_params_info}), key=lambda s: tuple(s), reverse=True)
		muon_shape_groups = []

		for shape in unique_shapes:
			# Get all params with this shape
			shape_params = [x for x in muon_params_info if x[1] == shape]

			# Construct the group
			group_params = [x[0] for x in shape_params]
			group_settings = [x[2] for x in shape_params]  # List of dicts matching params

			new_group = {
				"params": group_params,
				"use_muon": True,
				"zero_buffer": torch.zeros_like(group_params[0]),
				"per_param_settings": group_settings,  # Store individual settings
			}
			muon_shape_groups.append(new_group)

		final_groups = adamw_groups + muon_shape_groups
		super().__init__(final_groups, defaults)

	@torch.compile
	@torch.no_grad()
	def step(self, closure=None):
		if not dist.is_initialized():
			raise RuntimeError("DistMuon requires torch.distributed")
		rank = dist.get_rank()
		world_size = dist.get_world_size()

		muon_futures, adamw_futures = [], []
		muon_ctx, adamw_ctx = [], []

		# Reductions
		for group in self.param_groups:
			if group.get("use_muon"):
				params, zero_buf = group["params"], group["zero_buffer"]
				for base_i in range(0, len(params), world_size):
					chunk = params[base_i : base_i + world_size]
					rs_input = [p.grad if p.grad is not None else zero_buf for p in chunk]
					rs_input.extend([zero_buf] * (world_size - len(rs_input)))
					owner_idx = base_i + rank
					rs_output = (
						params[owner_idx].grad
						if owner_idx < len(params) and params[owner_idx].grad is not None
						else torch.empty_like(zero_buf)
					)
					work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
					muon_futures.append(work)
					muon_ctx.append((group, base_i, rs_output))
			else:
				for p in group["params"]:
					if p.grad is None:
						continue

					grad_flat = p.grad.view(-1)
					padding = (world_size - (grad_flat.numel() % world_size)) % world_size
					if padding > 0:
						grad_flat = torch.cat([grad_flat, torch.zeros(padding, device=p.device, dtype=p.dtype)])

					shard_size = grad_flat.numel() // world_size
					grad_shard = torch.empty(shard_size, device=p.device, dtype=p.dtype)
					work = dist.reduce_scatter_tensor(
						grad_shard, grad_flat, op=dist.ReduceOp.AVG, async_op=True
					).get_future()

					adamw_futures.append(work)
					adamw_ctx.append((group, p, grad_shard, padding, shard_size))

		# Process AdamW
		adamw_gather_futures = []
		for i, work in enumerate(adamw_futures):
			work.wait()
			group, p, g_shard, padding, shard_size = adamw_ctx[i]
			state = self.state[p]
			if len(state) == 0:
				state["step"] = 0
				state["exp_avg"] = torch.zeros_like(g_shard)
				state["exp_avg_sq"] = torch.zeros_like(g_shard)
			state["step"] += 1

			# Param shard needed for WD and update
			p_flat = p.data.view(-1)
			if padding > 0:
				p_flat = torch.cat([p_flat, torch.zeros(padding, device=p.device, dtype=p.dtype)])

			p_shard = p_flat[rank * shard_size : (rank + 1) * shard_size]

			adamw_update_kernel(
				p_shard,
				g_shard,
				state["exp_avg"],
				state["exp_avg_sq"],
				group["lr"],
				group["weight_decay"],
				group["adam_betas"][0],
				group["adam_betas"][1],
				group["adam_eps"],
				state["step"],
			)

			# Gather back
			gathered_p = torch.empty_like(p_flat)
			work = dist.all_gather_into_tensor(gathered_p, p_shard, async_op=True).get_future()
			adamw_gather_futures.append((work, p, gathered_p, padding))

		# Process Muon
		muon_gather_futures = []
		for i, work in enumerate(muon_futures):
			work.wait()
			group, base_i, g_avg = muon_ctx[i]
			params = group["params"]
			owner_idx = base_i + rank
			if owner_idx < len(params):
				p = params[owner_idx]
				state = self.state[p]
				if "momentum_buffer" not in state:
					state["momentum_buffer"] = torch.zeros_like(p)

				buf = state["momentum_buffer"]
				buf.lerp_(g_avg, 1.0 - group["momentum"])
				g = g_avg.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
				g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

				if group["weight_decay"] > 0:
					p.mul_(1 - group["lr"] * group["weight_decay"])

				p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5)
				ag_input = p
			else:
				ag_input = group["zero_buffer"]

			ag_out = params[base_i : base_i + world_size]
			ag_out.extend([torch.empty_like(group["zero_buffer"]) for _ in range(world_size - len(ag_out))])
			work = dist.all_gather(ag_out, ag_input, async_op=True).get_future()
			muon_gather_futures.append(work)

		for work, p, gathered_p, padding in adamw_gather_futures:
			work.wait()
			if padding > 0:
				gathered_p = gathered_p[:-padding]
			p.data.copy_(gathered_p.view_as(p))

		torch.futures.collect_all(muon_gather_futures).wait()
