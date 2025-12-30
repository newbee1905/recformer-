import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup
import itertools

from optimizer import Muon, DistMuon


# Unified logger for TensorBoard and WandB
class UnifiedLogger:
	def __init__(self, cfg: DictConfig, hydra_output_dir: Path, rank: int):
		self.rank = rank
		self.writer = None
		if self.rank != 0:
			return

		self.logger_type = cfg.training.get("logger", "tensorboard")
		self.is_wandb = self.logger_type == "wandb"

		if self.is_wandb:
			try:
				import wandb
				from dotenv import load_dotenv

				wandb_cfg = cfg.get("wandb")
				if not wandb_cfg or not wandb_cfg.get("project"):
					print(
						"Warning: wandb logger enabled but 'wandb.project' not configured. Falling back to TensorBoard."
					)
					self.logger_type = "tensorboard"
					self.is_wandb = False
				else:
					load_dotenv()
					model_name = cfg.model.get("name", "model")
					run_name = wandb_cfg.get("name") or f"{model_name}-{hydra_output_dir.name}"

					wandb.init(
						project=wandb_cfg.project,
						name=run_name,
						config=OmegaConf.to_container(cfg, resolve=True),
						dir=str(hydra_output_dir),
					)
					self.writer = wandb
			except ImportError:
				print("wandb not installed, falling back to tensorboard")
				self.logger_type = "tensorboard"
				self.is_wandb = False

		if not self.is_wandb:
			from torch.utils.tensorboard import SummaryWriter

			self.writer = SummaryWriter(log_dir=str(hydra_output_dir))

	def log(self, data: dict, step: int):
		if self.rank == 0 and self.writer:
			if self.is_wandb:
				self.writer.log({**data, "epoch": step})
			else:
				for key, value in data.items():
					self.writer.add_scalar(key, value, step)

	def close(self):
		if self.rank == 0 and self.writer:
			if self.is_wandb:
				self.writer.finish()
			else:
				self.writer.close()


class Trainer:
	def __init__(
		self,
		cfg: DictConfig,
		model,
		train_loader,
		val_loader,
		device,
		rank,
		world_size,
	):
		self.cfg = cfg
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.device = device
		self.rank = rank
		self.world_size = world_size
		self.is_ddp = self.world_size > 1
		self.is_main_process = self.rank == 0

		self.use_softcap = self.cfg.training.get("softcap.enabled", False)
		self.softcap_value = self.cfg.training.get("softcap.value", 15.0)

		model = model.to(device)
		if self.cfg.training.get("compile", False):
			if self.is_main_process:
				print("Compiling the model...")
			model = torch.compile(model)

		if self.is_ddp:
			if self.device.type == "cuda":
				self.model = DDP(model, device_ids=[self.device.index])
			else:
				self.model = DDP(model)  # For CPU DDP
		else:
			self.model = model

		self.dtype = getattr(torch, cfg.training.get("dtype", "bfloat16"))
		self.use_amp = (self.dtype == torch.float16) and ("cuda" in self.device.type)
		self.scaler = GradScaler(enabled=self.use_amp)
		self.grad_accum_steps = self.cfg.training.get("grad_accum_steps", 1)

		pad_id = self.cfg.model.pad_token_id
		self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

		# Create optimizer parameter groups
		muon_params = []
		adamw_decay_params = []
		adamw_no_decay_params = []

		tie_word_embeddings = self.cfg.model.get("tie_word_embeddings", True)

		for name, p in self.model.named_parameters():
			if not p.requires_grad:
				continue

			# If weights are not tied, the lm_head is optimized separately without weight decay/muon.
			if not tie_word_embeddings and "lm_head" in name:
				adamw_no_decay_params.append(p)
				continue

			if p.ndim >= 2:
				muon_params.append(p)
			else:
				if "norm" in name or "bias" in name:
					adamw_no_decay_params.append(p)
				else:
					adamw_decay_params.append(p)

		optimizer_cfg = self.cfg.training.optimizer
		param_groups = [
			{"params": muon_params, "use_muon": True},
			{
				"params": adamw_decay_params,
				"use_muon": False,
				"weight_decay": optimizer_cfg.weight_decay,
			},
			{"params": adamw_no_decay_params, "use_muon": False, "weight_decay": 0.0},
		]

		if self.is_ddp:
			self.optimizer = DistMuon(param_groups, lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay)
		else:
			self.optimizer = Muon(param_groups, lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay)

		optimizer_cfg = self.cfg.training.optimizer
		total_steps = (len(self.train_loader) * self.cfg.training.epochs) // self.grad_accum_steps
		warmup_steps = optimizer_cfg.get("warmup_steps", 10000)

		self.scheduler = get_linear_schedule_with_warmup(
			self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
		)

		self.grad_clip = self.cfg.training.get("grad_clip", 1.0)
		self.name = self.cfg.model.get("name", "model")
		self.epochs = self.cfg.training.epochs

		self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
		self.best_model_path = self.output_dir / f"best_{self.name}.pth"

		self.best_metric = float("inf")
		self.metric_mode = "min"  # For loss
		self.best_epoch = 0
		self.epochs_no_improve = 0
		self.early_stopping_patience = self.cfg.training.get("early_stopping_patience", 3)
		self.global_step = 0

		self.logger = UnifiedLogger(cfg, self.output_dir, self.rank)

	def _get_model_for_saving(self):
		model_to_save = self.model.module if self.is_ddp else self.model
		if self.cfg.training.get("compile", False):
			if hasattr(model_to_save, "_orig_mod"):
				model_to_save = model_to_save._orig_mod
		return model_to_save

	def save_checkpoint(self, path, epoch, best=True):
		"""Save training checkpoint."""
		if not self.is_main_process:
			return

		model_to_save = self._get_model_for_saving()
		checkpoint = {
			"epoch": epoch,
			"model_state_dict": model_to_save.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
			"scheduler_state_dict": (self.scheduler.state_dict() if self.scheduler else None),
			"best_metric": self.best_metric,
			"epochs_no_improve": self.epochs_no_improve,
			"best_epoch": self.best_epoch,
			"cfg": OmegaConf.to_container(self.cfg, resolve=True),
		}

		torch.save(checkpoint, path)
		print(f"Checkpoint saved to {path}")

	def _train_one_epoch(self, pbar, metrics_dict, epoch):
		self.model.train()
		if self.is_ddp:
			self.train_loader.sampler.set_epoch(epoch)
		running_loss = 0.0
		running_gen_loss = 0.0
		running_correct_tokens = 0
		running_total_tokens = 0
		pad_id = self.cfg.model.pad_token_id

		for batch_idx, batch in enumerate(self.train_loader):
			batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

			input_ids = batch["input_ids"]
			labels = batch["labels"]

			with autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
				gen_logits, _, _ = self.model(input_ids)

				if self.use_softcap:
					gen_logits = self.softcap_value * torch.tanh(gen_logits / self.softcap_value)

				loss = self.criterion(gen_logits.view(-1, gen_logits.size(-1)), labels.view(-1))

				# Scale loss for gradient accumulation
				loss_for_backward = loss / self.grad_accum_steps

			self.scaler.scale(loss_for_backward).backward()

			with torch.no_grad():
				preds = torch.argmax(gen_logits, dim=-1)
				mask = labels != pad_id
				running_correct_tokens += torch.sum(preds[mask] == labels[mask]).item()
				running_total_tokens += torch.sum(mask).item()

			if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
				if self.grad_clip is not None:
					self.scaler.unscale_(self.optimizer)
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

				self.scaler.step(self.optimizer)
				self.scaler.update()
				self.optimizer.zero_grad(set_to_none=True)
				self.scheduler.step()
				self.global_step += 1
				if self.is_main_process:
					pbar.update(1)

			running_loss += loss_for_backward.item() * self.grad_accum_steps
			running_gen_loss += loss.item()

			if self.is_main_process:
				metrics_dict["batch_loss"] = f"{loss_for_backward.item() * self.grad_accum_steps:.4f}"
				metrics_dict["lr"] = f"{self.optimizer.param_groups[0]['lr']:.6f}"
				pbar.set_postfix(metrics_dict)
				if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
					self.logger.log({"Loss/batch": loss_for_backward.item() * self.grad_accum_steps}, self.global_step)
					self.logger.log({"lr_per_batch": self.optimizer.param_groups[0]["lr"]}, self.global_step)

		epoch_loss = torch.tensor(running_loss / len(self.train_loader), device=self.device)
		epoch_gen_loss = torch.tensor(running_gen_loss / len(self.train_loader), device=self.device)
		epoch_token_accuracy = torch.tensor(
			(running_correct_tokens / running_total_tokens) if running_total_tokens > 0 else 0.0, device=self.device
		)

		if self.is_ddp:
			dist.all_reduce(epoch_loss, op=dist.ReduceOp.AVG)
			dist.all_reduce(epoch_gen_loss, op=dist.ReduceOp.AVG)
			dist.all_reduce(epoch_token_accuracy, op=dist.ReduceOp.AVG)

		return (
			epoch_loss.item(),
			epoch_gen_loss.item(),
			epoch_token_accuracy.item(),
		)

	def _evaluate(self, dataloader):
		self.model.eval()
		running_loss = 0.0
		running_gen_loss = 0.0

		total_tokens = 0
		correct_tokens = 0
		exact_matches = 0
		total_sequences = 0
		pad_id = self.cfg.model.pad_token_id

		with torch.no_grad():
			for batch in dataloader:
				batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

				input_ids = batch["input_ids"]
				labels = batch["labels"]

				with autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
					gen_logits, _, _ = self.model(input_ids)

					if self.use_softcap:
						gen_logits = self.softcap_value * torch.tanh(gen_logits / self.softcap_value)
					loss = self.criterion(gen_logits.view(-1, gen_logits.size(-1)), labels.view(-1))

				running_loss += loss.item()
				running_gen_loss += loss.item()

				# Accuracy calculation
				preds = torch.argmax(gen_logits, dim=-1)
				mask = labels != pad_id
				correct_tokens += torch.sum(preds[mask] == labels[mask]).item()
				total_tokens += torch.sum(mask).item()

				# Exact match calculation
				correct_in_sequence = (preds == labels) | ~mask
				exact_matches += torch.all(correct_in_sequence, dim=1).sum().item()
				total_sequences += labels.size(0)

		# Aggregate losses
		epoch_loss = torch.tensor(running_loss / len(dataloader), device=self.device)
		epoch_gen_loss = torch.tensor(running_gen_loss / len(dataloader), device=self.device)

		if self.is_ddp:
			dist.all_reduce(epoch_loss, op=dist.ReduceOp.AVG)
			dist.all_reduce(epoch_gen_loss, op=dist.ReduceOp.AVG)

		# Aggregate accuracy stats
		total_tokens = torch.tensor(total_tokens, device=self.device)
		correct_tokens = torch.tensor(correct_tokens, device=self.device)
		exact_matches = torch.tensor(exact_matches, device=self.device)
		total_sequences = torch.tensor(total_sequences, device=self.device)
		if self.is_ddp:
			dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
			dist.all_reduce(correct_tokens, op=dist.ReduceOp.SUM)
			dist.all_reduce(exact_matches, op=dist.ReduceOp.SUM)
			dist.all_reduce(total_sequences, op=dist.ReduceOp.SUM)

		token_accuracy = (correct_tokens / total_tokens) if total_tokens > 0 else torch.tensor(0.0)
		exact_accuracy = (exact_matches / total_sequences) if total_sequences > 0 else torch.tensor(0.0)

		return (
			epoch_loss.item(),
			epoch_gen_loss.item(),
			token_accuracy.item(),
			exact_accuracy.item(),
		)

	def train(self):
		total_steps = (len(self.train_loader) * self.epochs) // self.grad_accum_steps
		metrics_dict = {"phase": "train", "best_metric": f"{self.best_metric:.4f}"}

		with tqdm(total=total_steps, desc="Training Progress", disable=not self.is_main_process) as pbar:
			for epoch in range(self.epochs):
				self.current_epoch = epoch + 1
				self.optimizer.zero_grad(set_to_none=True)

				if self.is_main_process:
					metrics_dict["epoch"] = f"{self.current_epoch}/{self.epochs}"

				train_loss, train_gen_loss, train_token_acc = self._train_one_epoch(pbar, metrics_dict, epoch)
				(
					val_loss,
					val_gen_loss,
					val_token_acc,
					val_exact_acc,
				) = self._evaluate(self.val_loader)

				if self.is_main_process:
					metrics_dict["train_loss"] = f"{train_loss:.4f}"
					metrics_dict["train_token_acc"] = f"{train_token_acc:.4f}"
					metrics_dict["val_loss"] = f"{val_loss:.4f}"
					metrics_dict["val_token_acc"] = f"{val_token_acc:.4f}"
					metrics_dict["val_exact_acc"] = f"{val_exact_acc:.4f}"

					log_data = {
						"Loss/train": train_loss,
						"Loss/val": val_loss,
						"Loss/train_generator": train_gen_loss,
						"Loss/val_generator": val_gen_loss,
						"Accuracy/token_train": train_token_acc,
						"Accuracy/token_val": val_token_acc,
						"Accuracy/exact_val": val_exact_acc,
						"lr": self.optimizer.param_groups[0]["lr"],
					}

					self.logger.log(log_data, self.current_epoch)

					current_metric = val_loss
					if current_metric < self.best_metric:
						self.best_metric = current_metric
						self.best_epoch = self.current_epoch
						self.save_checkpoint(self.best_model_path, self.current_epoch, best=True)
						self.epochs_no_improve = 0
						metrics_dict["best_metric"] = f"{self.best_metric:.4f}"
					else:
						self.epochs_no_improve += 1

					pbar.set_postfix(metrics_dict)

				if self.is_ddp:
					dist.barrier()

				if self.epochs_no_improve >= self.early_stopping_patience:
					if self.is_main_process:
						print(f"\nEarly stopping triggered after {self.epochs_no_improve} epochs with no improvement.")
					break

		if self.is_ddp:
			dist.barrier()
		self.logger.close()
		if self.is_main_process:
			print(f"\nTraining complete. Best model from epoch {self.best_epoch} saved to {self.best_model_path}")
