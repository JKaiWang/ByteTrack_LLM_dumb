import os
import requests
import base64
import json
import re
import argparse
import sys

API_URL = "http://localhost:11434/api/generate"


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_filename(fname):
    match = re.match(r"i(\d+)_f(\d+)", fname)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (-1, -1)


def sort_key(x):
    # fname: the outmost filename
    fname = os.path.basename(x)
    # match i<number>_f<number>: i12_f004
    person_id, frame_id = parse_filename(fname)
    return (frame_id, person_id)


def select_targets(crops_dir, prompt, threshold=0.30, device="cpu", quiet=False):
    """Select target IDs from cropped images using LLM.

    Args:
        crops_dir: Directory containing cropped images named as "i{track_id}_f{frame_id}.jpg"
        prompt: Text prompt for LLM target selection
        threshold: Similarity threshold (not used in LLM-based selection)
        device: Device to use (cpu/cuda)
        quiet: Whether to suppress output

    Returns:
        List of selected track IDs
    """
    if not os.path.exists(crops_dir):
        return []

    files = sorted(
        [
            os.path.abspath(os.path.join(crops_dir, f))
            for f in os.listdir(crops_dir)
            if os.path.isfile(os.path.join(crops_dir, f))
            and (f.endswith(".jpg") or f.endswith(".png"))
        ],
        key=sort_key,  # sort frame_id first, then person_id
    )

    if not files:
        return []

    selected_ids = []

    for idx, f in enumerate(files, 1):
        fname = os.path.basename(f)
        person_id, frame_id = parse_filename(fname)

        if not quiet:
            print(f"[INFO] ({idx}/{len(files)}) Processing {fname}")

        payload = {
            "model": "qwen2.5vl",
            "prompt": f"{prompt}\n \
            Confirm whether it meets the requirements. \
            If yes, answer 'yes'. If not, answer 'no'.",
            "images": [encode_image(f)],
        }

        try:
            resp = requests.post(API_URL, json=payload, stream=True)
            resp.raise_for_status()
        except Exception as e:
            if not quiet:
                print(f"[ERROR] LLM request failed: {e}")
            continue

        result_text = ""
        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        result_text += data["response"]
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

        if not quiet:
            print(f"[DEBUG] LLM response: {result_text.strip()}")

        if "yes" in result_text.lower():
            selected_ids.append(person_id)
            if not quiet:
                print(f"[RESULT] Target ID: {person_id} selected")

    return selected_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    def get_sorted_files(crops_dir):
        return sorted(
            [
                os.path.abspath(os.path.join(crops_dir, f))
                for f in os.listdir(crops_dir)
                if os.path.isfile(os.path.join(crops_dir, f))
                and (f.endswith(".jpg") or f.endswith(".png"))
            ],
            key=sort_key,
        )

    files = get_sorted_files(args.crops)
    target_id = -1

    for idx, f in enumerate(files, 1):
        fname = os.path.basename(f)
        person_id, frame_id = parse_filename(fname)
        print(f"[INFO] ({idx}/{len(files)}) Processing {fname}")

        payload = {
            "model": "qwen2.5vl",
            "prompt": (
                f"{args.prompt}\n"
                "Confirm whether it meets the requirements. "
                "If yes, answer 'yes'. If not, answer 'no'."
            ),
            "images": [encode_image(f)],
        }

        try:
            resp = requests.post(API_URL, json=payload, stream=True)
            resp.raise_for_status()
        except Exception as e:
            print(f"[ERROR] LLM request failed: {e}")
            continue

        result_text = ""
        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        result_text += data["response"]
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

        print(f"[DEBUG] LLM response: {result_text.strip()}")

        if "yes" in result_text.lower():
            target_id = person_id
            print(f"[RESULT] Target ID: {target_id}")
            break

    print(json.dumps({"target_id": target_id}, ensure_ascii=False))
    sys.exit(0)
