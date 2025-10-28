import subprocess
import argparse
import json
import os, sys, re

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True)
args = parser.parse_args()

# --- Step 1: Run LLM.py first ---
llm_cmd = [
    "python", "LLM.py",
    "--crops", "crops",
    "--prompt", args.prompt
]

print("llm_cmd set.......")

try:
    print("Into try............")
    print(f"args.prompt: {args.prompt}")
    raw = subprocess.check_output(llm_cmd, stderr=subprocess.STDOUT)
    text = raw.decode("utf-8", errors="ignore")
    print("=== LLM Raw Output ===")
    # print(f"Json:::  {text.strip()}")
    print(text)
    print(type(text))
    match = re.search(r"\{.*\}",text)
    if match: text = match.group(0)
    # print((text.replace("\n"," ").replace("\r"," ").split()))
    
    data = json.loads(text)
    target_ids = data.get("target_ids", -1)
except Exception as e:
    print(f"[ERROR] Failed to get target_ids: {e}")
    target_ids = -1

# --- Step 2: Run demo_track.py and pass target_ids ---
cmd = [
    "python", "tools/demo_track.py", "video",
    "-f", "exps/example/mot/yolox_x_mix_det.py",
    "-c", "pretrained/bytetrack_x_mot17.pth.tar",
    "--fp16", "--fuse", "--save_result",
    "--path", "videos/test.mp4",
    "--track_thresh", "0.3",
    "--min_box_area", "5",
    # "--target_ids", str(target_ids),
    "--prompt", args.prompt
]
allow_ids_str = ",".join(map(str, target_ids))
cmd += ["--allow_ids", allow_ids_str]

print(f"[INFO] Launching tracker with target_ids = {target_ids}")
subprocess.run(cmd)
