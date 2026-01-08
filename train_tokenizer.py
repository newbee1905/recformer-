from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
import os
import argparse


def train_tokenizer(vocab_size):
	"""
	Trains a Byte-Pair Encoding tokenizer on the FineWeb-Edu dataset.
	"""
	print("Loading FineWeb-Edu dataset (full train split)...")
	dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

	def text_iterator():
		for item in dataset:
			yield item["text"]

	print(f"Training tokenizer with vocab size: {vocab_size}")
	# Initialize a tokenizer
	tokenizer = ByteLevelBPETokenizer()

	# Customize training
	tokenizer.train_from_iterator(
		text_iterator(),
		vocab_size=vocab_size,
		min_frequency=2,
		special_tokens=[
			"<s>",
			"<pad>",
			"</s>",
			"<unk>",
			"<mask>",
		],
	)

	fast_tokenizer = PreTrainedTokenizerFast(
		tokenizer_object=tokenizer,
		model_max_length=1024,
		bos_token="<s>",
		eos_token="</s>",
		unk_token="<unk>",
		pad_token="<pad>",
		mask_token="<mask>",
	)

	# Save files
	tokenizer_dir = f"fineweb-edu-tokenizer-vocab-{vocab_size}"
	if not os.path.exists(tokenizer_dir):
		os.makedirs(tokenizer_dir)
	print(f"Saving tokenizer to {tokenizer_dir}")
	fast_tokenizer.save_pretrained(tokenizer_dir)

	print("Tokenizer training complete.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train a BPE tokenizer on FineWeb-Edu")
	parser.add_argument("--vocab_size", type=int, default=16384, help="Vocabulary size for the tokenizer")
	args = parser.parse_args()
	train_tokenizer(args.vocab_size)
