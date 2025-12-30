import os
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import numpy as np


class WikiTextDataset(Dataset):
	def __init__(self, tokenizer: GPT2Tokenizer, max_length: int, split: str = "train", data_path=None):
		self.tokenizer = tokenizer
		self.max_length = max_length

		if data_path and os.path.exists(data_path):
			print(f"Loading dataset from local disk: {data_path}")
			dataset = load_from_disk(data_path)
		else:
			print("Local path not found, attempting download...")
			dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

		self.texts = dataset[split]['text']
		tokenized_texts = [self.tokenizer.encode(text) for text in self.texts if len(text) > 0]

		all_token_ids = [token_id for text in tokenized_texts for token_id in text]

		self.examples = []
		for i in range(0, len(all_token_ids) - self.max_length, self.max_length):
			self.examples.append(all_token_ids[i : i + self.max_length])

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		input_ids = torch.tensor(self.examples[idx], dtype=torch.long)
		labels = input_ids.clone()
		return {"input_ids": input_ids, "labels": labels}
