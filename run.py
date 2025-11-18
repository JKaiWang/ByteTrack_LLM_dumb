"""Entry point to run the tracker with per-frame LLM filtering.

This script simply wraps `tools/demo_track.py` and forwards a natural-language
prompt that describes the target person to track.
"""

import argparse
import os
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ByteTrack demo with per-frame LLM filtering.")
    parser.add_argument("--prompt", type=str, required=True, help="Text description of the target person to track.")
    return parser.parse_args()


def build_demo_track_cmd(prompt: str) -> list[str]:
    """Construct the command to launch the demo tracker.

    Using absolute paths and `sys.executable` makes the script more robust
    when called from different working directories and Python environments.
    """

    return [
        sys.executable,
        os.path.join(ROOT_DIR, "tools", "demo_track.py"),
        "video",
        "-f", os.path.join(ROOT_DIR, "exps", "example", "mot", "yolox_x_mix_det.py"),
        "-c", os.path.join(ROOT_DIR, "pretrained", "bytetrack_x_mot17.pth.tar"),
        "--fp16",
        "--fuse",
        "--save_result",
        "--path", os.path.join(ROOT_DIR, "videos", "test.mp4"),
        "--track_thresh", "0.3",
        "--min_box_area", "5",
        "--prompt", prompt,
        "--per_frame_llm",
    ]


def main() -> None:
    args = parse_args()

    cmd = build_demo_track_cmd(args.prompt)
    print(f"[INFO] Launching tracker with per_frame_llm, prompt = {args.prompt}")

    # Inherit stdout/stderr so the user sees the tracker logs directly.
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
