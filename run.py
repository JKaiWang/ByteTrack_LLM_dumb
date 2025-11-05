import subprocess
import argparse
import json
import os, sys, re
from subprocess import TimeoutExpired

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True)
# Optional controls
parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="inference device for tracker/LLM")
parser.add_argument("--enable-fp16", action="store_true", help="enable fp16 when device=gpu")
parser.add_argument("--similarity-threshold", type=float, default=0.30, help="cosine similarity threshold for Qwen2-CLIP")
parser.add_argument("--qwen2-model", type=str, default="openai/clip-vit-large-patch14", help="HF model id for CLIP-like model (e.g., openai/clip-vit-large-patch14)")
parser.add_argument("--llm-timeout", type=int, default=600, help="timeout (seconds) for LLM.py")
args = parser.parse_args()

# --- Step 1: Run LLM.py (Qwen2-CLIP cosine similarity) ---
llm_cmd = [
    sys.executable, "LLM.py",
    "--crops", "crops",
    "--prompt", args.prompt,
    "--threshold", str(args.similarity_threshold),
    "--qwen2-model", args.qwen2_model,
]
if args.device == "gpu":
    llm_cmd += ["--device", "cuda"]

print("[INFO] Launching LLM.py to select targets via cosine similarity...")
print(f"[INFO] prompt: {args.prompt}")

try:
    proc = subprocess.run(
        llm_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=max(1, args.llm_timeout),
    )
    text = proc.stdout or ""
except TimeoutExpired as e:
    text = e.stdout.decode("utf-8", errors="ignore") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")
    print("[WARN] LLM.py timed out; proceeding without restricting target ids.")
    proc = type("obj", (), {"returncode": -1})()  # sentinel

print("=== LLM Raw Output ===")
print(text)

# Parse JSON robustly (take the last JSON object with key target_ids)
target_ids = []
try:
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for cand in reversed(matches):
        try:
            data = json.loads(cand)
            if isinstance(data.get("target_ids"), list):
                target_ids = data["target_ids"]
                break
        except Exception:
            continue
except Exception as e:
    print(f"[WARN] JSON parse failed: {e}")

if getattr(proc, "returncode", 0) != 0:
    print(f"[WARN] LLM.py returned non-zero exit code: {getattr(proc, 'returncode', 'N/A')}")

# --- Step 2: Run demo_track.py and pass target_ids ---
cmd = [
    sys.executable, "tools/demo_track.py", "video",
    "-f", "exps/example/mot/yolox_x_mix_det.py",
    "-c", "pretrained/bytetrack_x_mot17.pth.tar",
    "--fuse", "--save_result",
    "--path", "videos/test.mp4",
    "--track_thresh", "0.3",
    "--min_box_area", "5",
    "--prompt", args.prompt,
]

if args.device == "gpu":
    cmd += ["--device", "gpu"]
else:
    cmd += ["--device", "cpu"]

if args.device == "gpu" and args.enable_fp16:
    cmd.insert(6, "--fp16")  # after ckpt for readability

if target_ids:
    allow_ids_str = ",".join(map(str, target_ids))
    cmd += ["--allow_ids", allow_ids_str]
    # Prevent demo_track.py from invoking its own LLM pass on frame 0
    # (it checks only target_id == -1 to decide). Since allow_ids takes precedence in filtering,
    # setting a dummy non-negative target_id safely skips the second query.
    cmd += ["--target_id", "0"]
else:
    print("[INFO] No ids passed threshold; using default (no restriction)")

print(f"[INFO] Launching tracker with target_ids = {target_ids} | device={args.device} | fp16={'on' if (args.device=='gpu' and args.enable_fp16) else 'off'}")
subprocess.run(cmd)
