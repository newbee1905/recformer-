import time
import hydra
import torch
from torch.amp import autocast, GradScaler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import numpy as np

# Adjust these imports to match your project structure
from model.bart import Bart
from chemformer_rs.tokenizer import SMILESTokenizer
from dataset.zinc import ZincDataset


def count_parameters(model):
	"""Counts the number of trainable parameters in a model."""
	num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	if num_params >= 1_000_000:
		return f"{num_params / 1_000_000:.1f}M"
	return f"{num_params / 1_000:.1f}K"


def get_gpu_memory_usage(device):
	"""Returns allocated and reserved memory in GB."""
	if device.type == "cuda":
		allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
		reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
		return allocated, reserved
	return 0.0, 0.0


@hydra.main(config_path="../config", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig):
	"""
	Main benchmarking function with AMP (Automatic Mixed Precision) support.
	"""
	print("--- Training Benchmark Script (AMP Enabled) ---")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		# Enable CuDNN benchmark for consistent input sizes (helps V100)
		torch.backends.cudnn.benchmark = True
		print(f"GPU: {torch.cuda.get_device_name(0)}")

	# Determine Data Type & AMP Settings
	dtype = getattr(torch, cfg.training.get("dtype", "float16"))
	use_amp = (dtype == torch.float16) and ("cuda" in device.type)

	print(f"AMP Mode: {'Enabled' if use_amp else 'Disabled'}")
	print(f"Target Dtype: {dtype}")

	# 3. Setup Dataset
	print("\nSetting up tokenizer and dataset...")
	tokenizer = SMILESTokenizer.from_vocab(hydra.utils.to_absolute_path(cfg.model.vocab_path))
	lmdb_path = hydra.utils.to_absolute_path(cfg.dataset.lmdb_path)
	train_indices = ZincDataset.read_split_indices(lmdb_path, "train")

	train_ds = ZincDataset(
		lmdb_path=lmdb_path,
		subset_indices=train_indices,
		tokenizer=tokenizer,
		max_length=cfg.model.max_length,
		is_training=True,
		mask_prob=cfg.dataset.mask_prob,
		span_len=cfg.dataset.span_len,
		augment_prob=cfg.dataset.augment_prob,
		span_mask_proportion=cfg.dataset.span_mask_proportion,
		span_random_proportion=cfg.dataset.span_random_proportion,
	)

	train_dl = DataLoader(
		train_ds,
		batch_size=cfg.training.batch_size,
		shuffle=True,
		num_workers=cfg.training.num_workers,
		pin_memory=True,
	)

	# 4. Setup Model
	OmegaConf.set_struct(cfg, False)
	cfg.model.vocab_size = tokenizer.vocab_size
	cfg.model.pad_token_id = tokenizer.token_to_index("<PAD>")
	OmegaConf.set_struct(cfg, True)

	model = Bart(cfg.model).to(device)
	print(f"Total trainable parameters: {count_parameters(model)}")

	if cfg.training.get("compile", False):
		print("Compiling the model...")
		model = torch.compile(model)

	criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.model.pad_token_id)
	if cfg.model.get("electra_task", False):
		electra_criterion = torch.nn.BCEWithLogitsLoss()

	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.optimizer.lr)

	# Required for float16 to prevent underflow. Not typically needed for bfloat16.
	scaler = GradScaler(enabled=(use_amp))

	# Benchmark Variables
	num_steps = cfg.get("bench_steps", 100)
	warmup_steps = cfg.get("warmup_steps", 10)
	data_times, model_times, total_times = [], [], []

	print(f"\nStarting benchmark: {warmup_steps} warmup + {num_steps} steps.")
	data_iterator = iter(train_dl)
	model.train()

	for i in range(warmup_steps + num_steps):
		# Reset memory stats right before actual benchmark
		if i == warmup_steps and device.type == "cuda":
			torch.cuda.reset_peak_memory_stats(device)

		t_total_start = time.time()

		# Data Loading
		t_data_start = time.time()
		try:
			batch = next(data_iterator)
		except StopIteration:
			data_iterator = iter(train_dl)
			batch = next(data_iterator)
		t_data_end = time.time()

		# Model Execution
		t_model_start = time.time()
		batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

		with autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
			gen_logits, _, disc_logits, _, _ = model(batch["src"], batch["tgt"])

			loss = criterion(gen_logits.view(-1, gen_logits.size(-1)), batch["tgt"].view(-1))

			if cfg.model.get("electra_task", False) and disc_logits is not None:
				electra_loss = electra_criterion(disc_logits, batch["electra_labels"])
				loss += electra_loss * cfg.training.electra_loss_weight

		optimizer.zero_grad()

		if use_amp:
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			# Standard backward for FP32 or BF16
			loss.backward()
			optimizer.step()

		if device.type == "cuda":
			torch.cuda.synchronize()
		t_model_end = time.time()

		t_total_end = time.time()

		# Store timings (skip warmup)
		if i >= warmup_steps:
			data_times.append(t_data_end - t_data_start)
			model_times.append(t_model_end - t_model_start)
			total_times.append(t_total_end - t_total_start)

		if (i + 1) % 10 == 0:
			print(f"  Step {i + 1} completed...")

	# Reporting
	print(f"\nThroughput: {1 / np.mean(total_times):.2f} batch/s")
	print("Benchmark complete.")

	print("\n--- Benchmark Results ---")

	def print_stats(name, timings, total_duration):
		timings_np = np.array(timings)
		avg = np.mean(timings_np)
		std = np.std(timings_np)
		percent_total = (np.sum(timings_np) / total_duration) * 100 if total_duration > 0 else 0

		print(f"\n{name}:")
		print(f"  - Average Time: {avg:.4f}s per batch")
		print(f"  - Std Dev:	  {std:.4f}s")
		print(f"  - Min Time:	 {np.min(timings_np):.4f}s")
		print(f"  - Max Time:	 {np.max(timings_np):.4f}s")
		print(f"  - Total Time:   {np.sum(timings_np):.2f}s (across {len(timings)} steps)")
		print(f"  - Percentage:   {percent_total:.2f}% of total benchmark time")

	total_benchmark_duration = np.sum(total_times)
	print_stats("Data Loading", data_times, total_benchmark_duration)
	print_stats("Model Processing (fwd+bwd)", model_times, total_benchmark_duration)

	# Memory Reporting
	if device.type == "cuda":
		max_allocated, max_reserved = get_gpu_memory_usage(device)
		print("\nMemory Usage (Peak during Benchmark):")
		print(f"  - Max Allocated: {max_allocated:.2f} GB (Tensor data)")
		print(f"  - Max Reserved:  {max_reserved:.2f} GB (PyTorch Cache)")
		print(f"  - Device Name:   {torch.cuda.get_device_name(device)}")

	print("\n--- Overall ---")
	print(f"Total benchmark time for {num_steps} steps: {total_benchmark_duration:.2f}s")
	print(f"Average total time per step: {np.mean(total_times):.4f}s")
	print(f"Batches per second (throughput): {1 / np.mean(total_times):.2f} batch/s")
	print("\nBenchmark complete.")


if __name__ == "__main__":
	main()
