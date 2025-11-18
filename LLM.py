"""LLM helper script

Given a directory of cropped person images, call a local LLM vision endpoint
for each image and decide whether it matches the textual prompt.

The script collects all person IDs that the LLM judges as a match and prints
JSON like: {"target_ids": [1, 3, 5]}.
"""

import argparse
import base64
import json
import os
import re
import sys
from typing import Iterable, List, Set, Tuple

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_image(path: str) -> str:
    """Read an image file and return its Base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_filename(filename: str) -> Tuple[int, int]:
    """Extract (person_id, frame_id) from a filename.

    Expected pattern: i{person_id}_f{frame_id}.*
    Returns (-1, -1) if the filename does not match.
    """

    match = re.match(r"i(\d+)_f(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return -1, -1


def list_crop_files(crops_dir: str) -> List[str]:
    """Return a list of absolute file paths in crops_dir sorted by (frame_id, person_id)."""

    files = [
        os.path.abspath(os.path.join(crops_dir, fname))
        for fname in os.listdir(crops_dir)
        if os.path.isfile(os.path.join(crops_dir, fname))
    ]

    # Sort by frame_id first, then person_id for deterministic order
    files.sort(key=lambda path: (
        parse_filename(os.path.basename(path))[1],
        parse_filename(os.path.basename(path))[0],
    ))
    return files


def call_llm(image_path: str, prompt: str) -> str:
    """Send a single image to the LLM endpoint and return the raw text response.

    The LLM is instructed to answer 'yes' or 'no'. Any network errors result in
    an empty string being returned.
    """

    prompt_text = (
        f"{prompt}\n"
        "Respond strictly with a single token: 'yes' if this crop visually "
        "matches the described target (clothes, appearance, etc.), otherwise "
        "'no'."
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "images": [encode_image(image_path)],
    }

    try:
        response = requests.post(API_URL, json=payload, stream=True)
        response.raise_for_status()
    except Exception as exc:  # broad, but we want to keep the script robust
        print(f"[ERROR] LLM request failed for {os.path.basename(image_path)}: {exc}")
        return ""

    result_chunks: List[str] = []
    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            # Skip malformed chunks
            continue
        if "response" in data:
            result_chunks.append(data["response"])

    result_text = "".join(result_chunks).strip().lower()
    print(f"[DEBUG] LLM response for {os.path.basename(image_path)}: {result_text}")
    return result_text


def collect_positive_ids(files: Iterable[str], prompt: str) -> Set[int]:
    """Iterate over crop files, ask the LLM, and collect matching person IDs."""

    positive_ids: Set[int] = set()
    files_list = list(files)

    for idx, file_path in enumerate(files_list, start=1):
        filename = os.path.basename(file_path)
        person_id, frame_id = parse_filename(filename)
        print(f"[INFO] ({idx}/{len(files_list)}) Processing {filename}")

        result_text = call_llm(file_path, prompt)

        # Do not stop at the first positive; collect all matching IDs
        if "yes" in result_text:
            positive_ids.add(person_id)
            print(f"[INFO] Marked as positive: person_id={person_id}, frame_id={frame_id}")

    return positive_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call LLM on cropped images and collect matching IDs.")
    parser.add_argument("--crops", type=str, required=True, help="Directory containing cropped person images.")
    parser.add_argument("--prompt", type=str, required=True, help="Text description of the target person.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.crops):
        print(f"[ERROR] Crops directory does not exist: {args.crops}")
        sys.exit(1)

    files = list_crop_files(args.crops)
    if not files:
        print(f"[WARN] No crop files found in directory: {args.crops}")

    positive_ids = collect_positive_ids(files, args.prompt)

    # Output all positive IDs as JSON
    output = {"target_ids": sorted(positive_ids)}
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
