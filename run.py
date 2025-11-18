import subprocess
import argparse
import json
import os, sys, re

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True)
args = parser.parse_args()

# 直接讓 demo_track.py 在每一禎偵測到新 ID 時，呼叫 LLM.py 判斷是否符合 prompt
# 不再事先只用第一禎決定 target_ids 一路追到結束。

cmd = [
    "python", "tools/demo_track.py", "video",
    "-f", "exps/example/mot/yolox_x_mix_det.py",
    "-c", "pretrained/bytetrack_x_mot17.pth.tar",
    "--fp16", "--fuse", "--save_result",
    "--path", "videos/test.mp4",
    "--track_thresh", "0.3",
    "--min_box_area", "5",
    "--prompt", args.prompt,
    "--per_frame_llm",
]

print(f"[INFO] Launching tracker with per_frame_llm, prompt = {args.prompt}")
subprocess.run(cmd)
