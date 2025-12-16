"""
Script to create the MIDI dataset from a collection of MIDI files.
"""

from __future__ import annotations
import os
import logging
import hashlib
import shutil
import json
import typing as t
from copy import deepcopy
from pathlib import Path
import argparse
from tqdm import tqdm
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

CHUNK_SIZE = 8 * 1024


def find_midi_files(
    base_path: str,
    exts: tuple[str, ...] = ('.mid', '.midi'),
    verbose: bool = True,
) -> t.Generator[str, None, None]:
    """Returns a list of all MIDI files in the directory and subdirectories."""
    midi_files_found = tqdm(desc=" MIDI files found...", unit="file", leave=True, disable=not verbose)
    for ext in exts:
        for path in Path(base_path).rglob('*' + ext):
            yield str(path)
            midi_files_found.update(1)
    midi_files_found.close()


def compute_hash(file_path: str) -> str | None:
    """Return MD5 hex digest of file content, or None on error."""
    try:
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            md5.update(f.read(CHUNK_SIZE))
            return md5.hexdigest()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None


def deduplicate_files(
    files: t.Iterable[str],
) -> t.Generator[str, None, None]:
    """
    Consume an iterable/generator of file paths, compute MD5 and yield only unique files.
    """
    used_hashes = set()
    for file_path in files:
        file_hash = compute_hash(file_path)
        if file_hash is not None and file_hash not in used_hashes:
            used_hashes.add(file_hash)
            yield file_path
    return


def grab_metadata(metadata: dict[str, t.Any], files: t.Iterable[str]) -> t.Generator[dict[str, t.Any], None, None]:
    """Extract metadata from a MIDI file."""
    for file_path in files:
        key = os.path.basename(file_path).split("_")[0].lstrip("0")
        if key not in metadata:
            logging.warning(f"Metadata for file {file_path} ({key}) not found.")
            continue

        try:
            midi_bytes = Path(file_path).read_bytes()
        except Exception as e:
            logging.error(f"Error reading MIDI file {file_path}: {e}")
            continue

        try:
            song_metadata = deepcopy(metadata[key]["metadata"])
            song_metadata["audio_scores"] = metadata[key]["audio_scores"]
            song_metadata["file"] = midi_bytes
            yield song_metadata
        except Exception as e:
            logging.error(f"Error processing metadata for file {file_path}: {e}")
            continue


def collect_files(metadata_keys: frozenset[str], file_metadata_iter: t.Iterable[dict[str, t.Any]], count: int = -1) -> Dataset:
    # Transpose the whole thing
    dataset: dict[str, list[t.Any]] = {key: [] for key in metadata_keys}
    for song_metadata in tqdm(file_metadata_iter, desc="Collecting files...", unit="file"):
        for key in metadata_keys:
            dataset[key].append(song_metadata.get(key, None))
        count -= 1
        if count == 0:
            break

    # Convert to hf dataset
    ds = Dataset.from_dict(dataset)
    return ds


def login_to_hf():
    """Login to Hugging Face."""
    from huggingface_hub import login
    import dotenv
    dotenv.load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("Hugging Face token not found in environment variables.")
    login(token=hf_token)


def upload_to_hf(dataset: Dataset, ds_name: str):
    """Upload dataset to Hugging Face."""
    dataset.push_to_hub(ds_name)
    logging.info(f"Dataset uploaded to Hugging Face under the name: {ds_name}")


def main(base_directory: str, ds_name: str):
    """Create a dataset from a directory."""
    if not os.path.exists(base_directory):
        raise FileNotFoundError(f"Directory {base_directory} does not exist.")
    if not os.path.isdir(base_directory):
        raise NotADirectoryError(f"{base_directory} is not a directory.")

    # Login to HF
    login_to_hf()

    with open(os.path.join(base_directory, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    keys = set()
    for key in metadata.keys():
        keys.update(metadata[key]["metadata"].keys())
    keys.add("audio_scores")
    keys.add("file")
    metadata_keys = frozenset(keys)

    midi_files = find_midi_files(base_directory)
    unique_midi_files = deduplicate_files(midi_files)
    file_metadata_iter = grab_metadata(metadata, unique_midi_files)
    dataset = collect_files(metadata_keys, file_metadata_iter)

    dataset.save_to_disk(f"./{ds_name}_dataset")

    logging.info(f"Dataset created with {len(dataset)} entries.")
    dataset.push_to_hub(ds_name)
    logging.info(f"Dataset '{ds_name}' successfully created and uploaded.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python preprocess_aria.py <base_directory>")
        sys.exit(1)

    base_directory = sys.argv[1]
    ds_name = os.path.basename(os.path.normpath(base_directory))
    print(f"Creating dataset from directory: {base_directory} with name: {ds_name}")
    dataset = main(
        base_directory=base_directory,
        ds_name=ds_name,
    )
