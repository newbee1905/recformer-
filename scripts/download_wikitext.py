import argparse
import os
from datasets import load_dataset


def download_wikitext(output_path: str):
	"""
	Downloads the wikitext-103-raw-v1 dataset and saves it to the specified path.
	"""
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	print(f"Downloading wikitext-103-raw-v1 to {output_path}...")
	# Load all splits
	train_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
	val_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
	test_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

	# Create a DatasetDict to save all splits together
	from datasets import DatasetDict

	full_dataset = DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})

	full_dataset.save_to_disk(output_path)
	print(f"Wikitext-103-raw-v1 dataset downloaded and saved to {output_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Download and save WikiText-103-raw-v1 dataset offline.")
	parser.add_argument(
		"--output_path", type=str, default="./data/wikitext-103-raw-v1", help="Path to save the downloaded dataset."
	)
	args = parser.parse_args()
	download_wikitext(args.output_path)
