import os
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np


class FineWebEduDataset(Dataset):
	def __init__(
		self, tokenizer_path: str, max_length: int, split: str = "train", data_path=None, num_proc=os.cpu_count()
	):
		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
		self.max_length = max_length

		if data_path and os.path.exists(data_path):
			print(f"Loading pre-processed dataset from disk: {data_path}")
			self.dataset = load_from_disk(data_path)
		else:
			print("Loading raw dataset from HuggingFace...")
			raw_dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split=split)

			print("Tokenizing and packing dataset (this may take a while)...")

			self.dataset = raw_dataset.map(
				self._group_texts,
				batched=True,
				num_proc=num_proc,
				remove_columns=raw_dataset.column_names,
				desc=f"Packing tokens into chunks of {max_length}",
			)

			if data_path:
				print(f"Saving processed dataset to {data_path}")
				self.dataset.save_to_disk(data_path)

		self.dataset.set_format(type="torch", columns=["input_ids", "labels"])

	def _group_texts(self, examples):
		tokenized_inputs = self.tokenizer(examples["text"], truncation=False, add_special_tokens=False)
		concatenated_examples = {k: sum(tokenized_inputs[k], []) for k in tokenized_inputs.keys()}

		total_length = len(concatenated_examples["input_ids"])

		if total_length >= self.max_length:
			total_length = (total_length // self.max_length) * self.max_length

		result = {
			"input_ids": [
				concatenated_examples["input_ids"][i : i + self.max_length]
				for i in range(0, total_length, self.max_length)
			],
			# Auto-generate labels here (same as input_ids for Causal LM)
			"labels": [
				concatenated_examples["input_ids"][i : i + self.max_length]
				for i in range(0, total_length, self.max_length)
			],
		}
		return result

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		return self.dataset[idx]
