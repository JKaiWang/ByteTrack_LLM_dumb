import subprocess
import argparse
import os, sys

# --- Add project paths so YOLOX can be imported correctly ---
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "yolox")))

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "yolox"))

# --- Argument parser for LLM prompt ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt",
    type=str,
    required=True,
    help="Prompt for the LLM to determine which target to track"
)
args = parser.parse_args()

# --- Input video path ---
video_path = "videos/test.mp4"

# --- Command to run the tracking demo ---
cmd = [
    "python", "tools/demo_track.py", "video",
    "-f", "exps/example/mot/yolox_x_mix_det.py",
    "-c", "pretrained/bytetrack_x_mot17.pth.tar",
    "--fp16", "--fuse", "--save_result",
    "--path", video_path,
    "--track_thresh", "0.3",
    "--min_box_area", "5",
    "--prompt", args.prompt   # Pass the prompt to demo_track.py
]

# --- Execute the command ---
subprocess.run(cmd)
