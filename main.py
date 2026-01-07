import os
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.recurrent_decoder import RecurrentDecoder
from transformers import AutoTokenizer
from trainer import Trainer
from dataset.fineweb import FineWebEduDataset 


def setup():
	if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
		dist.init_process_group(backend="nccl")
		rank = int(os.environ["RANK"])
		local_rank = int(os.environ["LOCAL_RANK"])
		world_size = int(os.environ["WORLD_SIZE"])
		torch.cuda.set_device(local_rank)
		device = torch.device("cuda", local_rank)
		is_ddp = True
	else:
		rank = 0
		world_size = 1
		local_rank = 0
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		is_ddp = False
	return rank, world_size, device, local_rank, is_ddp


def cleanup_ddp():
	if "WORLD_SIZE" in os.environ:
		dist.destroy_process_group()


def count_parameters(model):
	num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	if num_params >= 1_000_000:
		return f"{num_params / 1_000_000:.1f}M"
	return f"{num_params / 1_000:.1f}K"


@hydra.main(config_path="config", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig):
	rank, world_size, device, local_rank, is_ddp = setup()

	if rank == 0:
		print(f"Hydra configuration:\n{OmegaConf.to_yaml(cfg)}")
		print(f"Running on device: {device}")

	tokenizer_path = cfg.model.get("tokenizer_path", "gpt2")

	# Load tokenizer just to get vocab_size and pad_token_id
	temp_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
	temp_tokenizer.pad_token = temp_tokenizer.eos_token

	initial_vocab_size = temp_tokenizer.vocab_size
	pad_token_id = temp_tokenizer.pad_token_id

	# Adjust vocab_size to be divisible by 128
	adjusted_vocab_size = (initial_vocab_size + 127) // 128 * 128
	if rank == 0:
		print(f"Initial tokenizer vocab size: {initial_vocab_size}")
		print(f"Adjusted vocab size (divisible by 128): {adjusted_vocab_size}")

	if rank == 0:
		print(f"Tokenizer loaded with {initial_vocab_size} tokens. Adjusted to {adjusted_vocab_size} for model embedding.")

	train_ds = FineWebEduDataset( 
		tokenizer_path=tokenizer_path,
		max_length=cfg.model.max_length,
		split="train",
		data_path=cfg.dataset.path
	)
	train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
	train_dl = DataLoader(
		train_ds,
		batch_size=cfg.training.batch_size,
		sampler=train_sampler,
		shuffle=(train_sampler is None),
		num_workers=cfg.training.num_workers,
		pin_memory=True,
	)

	val_ds = FineWebEduDataset(
		tokenizer_path=tokenizer_path,
		max_length=cfg.model.max_length,
		split="validation", 
		data_path=cfg.dataset.path
	)
	val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
	val_dl = DataLoader(
		val_ds,
		batch_size=cfg.training.batch_size,
		sampler=val_sampler,
		shuffle=False,
		num_workers=cfg.training.num_workers,
		pin_memory=True,
	)

	OmegaConf.set_struct(cfg, False)
	cfg.model.vocab_size = adjusted_vocab_size # Use adjusted vocab size
	cfg.model.pad_token_id = pad_token_id # Use obtained pad_token_id
	OmegaConf.set_struct(cfg, True)

	model = RecurrentDecoder(cfg.model).to(device)
	if rank == 0:
		print(f"Total number of trainable parameters in the model: {count_parameters(model)}")

	trainer = Trainer(
		cfg=cfg,
		model=model,
		train_loader=train_dl,
		val_loader=val_dl,
		device=device,
		rank=rank,
		world_size=world_size,
	)

	trainer.train()
	cleanup_ddp()


if __name__ == "__main__":
	main()
