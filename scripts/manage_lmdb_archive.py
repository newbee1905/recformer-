import argparse
import lmdb
import zstandard as zstd
import pickle
import os
from tqdm import tqdm
from pyroaring import BitMap


def compress_lmdb(args: argparse.Namespace):
	"""Compresses an LMDB database to a custom .zst archive, separating metadata."""
	if not os.path.isdir(args.input):
		print(f"Error: Input LMDB path '{args.input}' not found or is not a directory.")
		return

	print(f"Scanning LMDB database at '{args.input}'...")
	try:
		env = lmdb.open(args.input, readonly=True, lock=False)
	except lmdb.Error as e:
		print(f"Error opening LMDB database: {e}")
		return

	metadata = {}
	with env.begin() as txn:
		with txn.cursor() as cursor:
			for key, value in cursor:
				if not key.isdigit():
					metadata[key] = value

	if b"__len__" not in metadata:
		print("Error: '__len__' key not found in the database. Cannot determine number of data records.")
		env.close()
		return
	num_data_records = int(metadata[b"__len__"].decode("ascii"))

	print(f"Found {num_data_records:,} data records and {len(metadata)} metadata keys.")

	cctx = zstd.ZstdCompressor(level=args.level)
	print(f"Compressing to '{args.output}' using zstd (level={args.level})...")
	try:
		with open(args.output, "wb") as f_out, cctx.stream_writer(f_out) as compressor:
			# Write the entire metadata dictionary
			pickle.dump(metadata, compressor, protocol=pickle.HIGHEST_PROTOCOL)

			# Write the data records sequentially
			with env.begin() as txn:
				with tqdm(range(num_data_records), desc="Compressing data", unit="recs") as pbar:
					for i in pbar:
						key = f"{i}".encode("ascii")
						value = txn.get(key)
						if value is None:
							print(f"Warning: Key {i} not found, skipping record.")
							continue
						pickle.dump((key, value), compressor, protocol=pickle.HIGHEST_PROTOCOL)

	except Exception as e:
		print(f"An error occurred during compression: {e}")
	finally:
		env.close()

	print("Compression complete.")


def decompress_lmdb(args: argparse.Namespace):
	"""Decompresses a custom .zst archive to an LMDB database."""
	if not os.path.isfile(args.input):
		print(f"Error: Input file '{args.input}' not found.")
		return

	if os.path.exists(args.output):
		print(f"Error: Output path '{args.output}' already exists. Please remove it first.")
		return

	print(f"Decompressing '{args.input}' to LMDB at '{args.output}'...")
	try:
		env = lmdb.open(args.output, map_size=args.map_size, writemap=True)
	except lmdb.Error as e:
		print(f"Error creating new LMDB database: {e}")
		return

	try:
		with open(args.input, "rb") as f_in, zstd.ZstdDecompressor().stream_reader(f_in) as reader:
			unpickler = pickle.Unpickler(reader)

			metadata = unpickler.load()
			print(f"Found {len(metadata)} metadata keys to restore.")

			# Get number of data records from metadata
			if b"__len__" not in metadata:
				print("Error: Archive is missing the '__len__' metadata key. Cannot determine record count.")
				return
			num_data_records = int(metadata[b"__len__"].decode("ascii"))
			print(f"Expecting {num_data_records:,} data records.")

			with env.begin(write=True) as txn:
				for key, value in metadata.items():
					txn.put(key, value)

			print("Successfully restored metadata.")

			# Write data records
			with tqdm(total=num_data_records, desc="Decompressing data", unit="recs") as pbar:
				txn = env.begin(write=True)
				for i in range(num_data_records):
					try:
						key, value = unpickler.load()
						txn.put(key, value)
					except EOFError:
						print(f"\nError: Archive ended prematurely. Expected {num_data_records} records, found {i}.")
						break
					pbar.update(1)

					if (i + 1) % args.chunk_size == 0:
						txn.commit()
						pbar.set_description(f"Decompressing (commit {(i + 1) // args.chunk_size})")
						txn = env.begin(write=True)

				txn.commit()
				env.sync()

	except (pickle.UnpicklingError, zstd.ZstdError) as e:
		print(f"An error occurred during decompression. The file may be corrupt. Error: {e}")
	except Exception as e:
		print(f"An unexpected error occurred: {e}")
	finally:
		env.close()

	print("Decompression complete.")


def verify_lmdb(args: argparse.Namespace):
	"""Verifies that two LMDB databases are identical without loading all data into memory."""
	for path in [args.db1, args.db2]:
		if not os.path.isdir(path):
			print(f"Error: LMDB path '{path}' not found or is not a directory.")
			return

	print(f"Verifying integrity between '{args.db1}' and '{args.db2}'...")

	try:
		env1 = lmdb.open(args.db1, readonly=True, lock=False)
		env2 = lmdb.open(args.db2, readonly=True, lock=False)

		with env1.begin() as txn1, env2.begin() as txn2:
			# --- Verify Metadata ---
			print("Verifying metadata...")

			meta_keys1 = {key for key, _ in txn1.cursor() if not key.isdigit()}
			meta_keys2 = {key for key, _ in txn2.cursor() if not key.isdigit()}

			if meta_keys1 != meta_keys2:
				print("Verification FAILED: Metadata keys do not match.")
				print(f"  Keys only in DB1: {meta_keys1 - meta_keys2}")
				print(f"  Keys only in DB2: {meta_keys2 - meta_keys1}")
				return

			for key in sorted(list(meta_keys1)):
				val1 = txn1.get(key)
				val2 = txn2.get(key)

				if key.startswith(b"split_"):
					try:
						raw_bytes1 = zstd.decompress(val1)
						bitmap1 = BitMap.deserialize(raw_bytes1)

						raw_bytes2 = zstd.decompress(val2)
						bitmap2 = BitMap.deserialize(raw_bytes2)

						if bitmap1 != bitmap2:
							print(f"\nVerification FAILED: BitMap mismatch for key '{key.decode()}'.")
							print(f"  DB1 has {len(bitmap1)} indices, DB2 has {len(bitmap2)} indices.")
							return
					except Exception as e:
						print(f"\nVerification FAILED: Could not compare BitMaps for key '{key.decode()}'. Error: {e}")
						return
				elif val1 != val2:
					print(f"\nVerification FAILED: Metadata value mismatch for key '{key.decode('utf-8', 'ignore')}'.")
					return
			print(f"SUCCESS: {len(meta_keys1)} metadata entries are identical.")

			# Get total length from metadata to guide data comparison
			len1 = int(txn1.get(b"__len__").decode("ascii"))
			len2 = int(txn2.get(b"__len__").decode("ascii"))

			if len1 != len2:
				print(f"\nVerification FAILED: Data record counts do not match ('__len__' metadata).")
				print(f"  DB1 has {len1:,} records.")
				print(f"  DB2 has {len2:,} records.")
				return

			# --- Verify Data Records  ---
			print("\nVerifying data records...")
			print(f"Data record counts match: {len1:,}. Comparing records one-by-one...")

			with tqdm(range(len1), desc="Comparing data", unit="recs") as pbar:
				for i in pbar:
					key_bytes = f"{i}".encode("ascii")
					val1 = txn1.get(key_bytes)
					val2 = txn2.get(key_bytes)

					if val1 != val2:
						if val1 is None:
							print(f"\nVerification FAILED: Key {i} not found in DB1, but exists in DB2.")
						elif val2 is None:
							print(f"\nVerification FAILED: Key {i} not found in DB2, but exists in DB1.")
						else:
							print(f"\nVerification FAILED: Data mismatch at key {i}.")
						return

			print("SUCCESS: All data records are identical.")
			print("\nOverall verification SUCCESS: The two LMDB databases are identical.")

	except Exception as e:
		print(f"An error occurred during verification: {e}")
	finally:
		env1.close()
		env2.close()


def main():
	parser = argparse.ArgumentParser(
		description="Compress, decompress, and verify LMDB databases using Zstandard.",
		formatter_class=argparse.RawTextHelpFormatter,
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	# --- Compression ---
	parser_compress = subparsers.add_parser(
		"compress", help="Compress an LMDB database folder into a single .zst file."
	)
	parser_compress.add_argument("--input", required=True, help="Path to the input LMDB database directory.")
	parser_compress.add_argument(
		"--output", required=True, help="Path for the output compressed file (e.g., data.lmdb.zst)."
	)
	parser_compress.add_argument(
		"--level",
		type=int,
		default=3,
		help="Zstandard compression level. 1 is fastest, 22 is highest compression. (default: 3)",
	)
	parser_compress.set_defaults(func=compress_lmdb)

	# --- Decompression ---
	parser_decompress = subparsers.add_parser("decompress", help="Decompress a .zst file into an LMDB database folder.")
	parser_decompress.add_argument("--input", required=True, help="Path to the input compressed .zst file.")
	parser_decompress.add_argument("--output", required=True, help="Path for the output LMDB database directory.")
	parser_decompress.add_argument(
		"--map_size", type=int, default=10**12, help="LMDB map size for the new database (default 1TB)."
	)
	parser_decompress.add_argument(
		"--chunk_size", type=int, default=100000, help="Records per LMDB commit during decompression."
	)
	parser_decompress.set_defaults(func=decompress_lmdb)

	# --- Verification ---
	parser_verify = subparsers.add_parser("verify", help="Verify that two LMDB databases are identical.")
	parser_verify.add_argument("db1", help="Path to the first LMDB database directory.")
	parser_verify.add_argument("db2", help="Path to the second LMDB database directory to compare against.")
	parser_verify.set_defaults(func=verify_lmdb)

	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
