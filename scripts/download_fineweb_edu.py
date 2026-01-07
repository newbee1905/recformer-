from datasets import load_dataset
import os


def download_fineweb_edu():
	"""
	Downloads and saves the entire FineWeb-Edu dataset to a local offline path.
	"""
	save_path = "offline_cache/fineweb-edu"
	if os.path.exists(save_path):
		print(f"Dataset already found at {save_path}. Skipping download.")
		return

	print(f"Starting download of FineWeb-Edu dataset (full train split)...")
	split_string = "train"
	dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split=split_string)

	print(f"Saving dataset to {save_path}...")
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	dataset.save_to_disk(save_path)

	print(f"\nFineWeb-Edu dataset (full train split) saved to {save_path} successfully.")


if __name__ == "__main__":
	download_fineweb_edu()
