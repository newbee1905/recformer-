import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from pathlib import Path
import glob
from tqdm import tqdm

from model.recurrent_decoder import RecurrentDecoder
from dataset.wikitext import WikiTextDataset


def find_checkpoints(outputs_dir="outputs"):
	"""Finds all model checkpoints in the outputs directory."""
	return list(Path(outputs_dir).rglob("*.pth"))


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_softcap, softcap_value, pad_token_id):
	"""Evaluates the model on the given dataloader."""
	model.eval()
	running_loss = 0.0
	total_tokens = 0
	correct_tokens = 0
	exact_matches = 0
	total_sequences = 0

	for batch in tqdm(dataloader, desc="Evaluating", leave=False):
		batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
		input_ids = batch["input_ids"]
		labels = batch["labels"]

		with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
			gen_logits, _, _ = model(input_ids)

			if use_softcap:
				gen_logits = softcap_value * torch.tanh(gen_logits / softcap_value)

			loss = criterion(gen_logits.view(-1, gen_logits.size(-1)), labels.view(-1))

		running_loss += loss.item()

		# Accuracy calculation
		preds = torch.argmax(gen_logits, dim=-1)
		mask = labels != pad_token_id
		correct_tokens += torch.sum(preds[mask] == labels[mask]).item()
		total_tokens += torch.sum(mask).item()

		# Exact match calculation
		correct_in_sequence = (preds == labels) | ~mask
		exact_matches += torch.all(correct_in_sequence, dim=1).sum().item()
		total_sequences += labels.size(0)

	epoch_loss = running_loss / len(dataloader)
	valid_ppl = torch.exp(torch.tensor(epoch_loss)).item()
	token_accuracy = (correct_tokens / total_tokens) if total_tokens > 0 else 0.0
	exact_accuracy = (exact_matches / total_sequences) if total_sequences > 0 else 0.0

	return epoch_loss, token_accuracy, exact_accuracy, valid_ppl


@hydra.main(config_path="config", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig):
	"""Main inference script."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Running on device: {device}")

	checkpoints = find_checkpoints()
	if not checkpoints:
		print("No checkpoints found in the 'outputs' directory.")
		return

	tokenizer_path = cfg.model.get("tokenizer_path", "gpt2")
	tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
	tokenizer.pad_token = tokenizer.eos_token
	pad_token_id = tokenizer.pad_token_id

	print(f"Tokenizer loaded with {tokenizer.vocab_size} tokens.")

	# Load test dataset
	test_ds = WikiTextDataset(
		tokenizer=tokenizer, max_length=cfg.model.max_length, split="test", data_path=cfg.dataset.path
	)
	test_dl = DataLoader(
		test_ds,
		batch_size=cfg.training.batch_size,
		shuffle=False,
		num_workers=cfg.training.num_workers,
		pin_memory=True,
	)

	criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

	print(f"\nFound {len(checkpoints)} model checkpoints. Starting evaluation...")

	for checkpoint_path in checkpoints:
		hydra_cfg_path = checkpoint_path.parent / ".hydra" / "config.yaml"
		if not hydra_cfg_path.exists():
			print(f"Skipping {checkpoint_path}: config.yaml not found.")
			continue

		print(f"\n--- Evaluating {checkpoint_path.parent.name}/{checkpoint_path.name} ---")

		# Load model-specific config
		model_cfg = OmegaConf.load(hydra_cfg_path)

		OmegaConf.set_struct(model_cfg, False)
		model_cfg.model.vocab_size = tokenizer.vocab_size
		model_cfg.model.pad_token_id = tokenizer.pad_token_id
		OmegaConf.set_struct(model_cfg, True)

		model = RecurrentDecoder(model_cfg.model).to(device)

		checkpoint = torch.load(checkpoint_path, map_location=device)

		# Adjust for compiled models
		state_dict = checkpoint["model_state_dict"]
		if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
			state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

		model.load_state_dict(state_dict)

		print(f"Model: {model_cfg.model.name}, Version: {checkpoint_path.parent.name}")

		use_softcap = model_cfg.training.get("softcap.enabled", False)
		softcap_value = model_cfg.training.get("softcap.value", 15.0)

		loss, token_acc, exact_acc, ppl = evaluate(
			model, test_dl, criterion, device, use_softcap, softcap_value, pad_token_id
		)

		print(f"  Test Loss: {loss:.4f}")
		print(f"  Test PPL: {ppl:.4f}")
		print(f"  Token Accuracy: {token_acc:.4f}")
		print(f"  Exact Match Accuracy: {exact_acc:.4f}")


if __name__ == "__main__":
	main()
