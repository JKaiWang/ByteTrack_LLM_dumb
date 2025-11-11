import subprocess
import argparse
import os, sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from LLM import select_targets

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True)
# Optional controls
parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="inference device for tracker/LLM")
parser.add_argument("--enable-fp16", action="store_true", help="enable fp16 when device=gpu")
parser.add_argument("--similarity-threshold", type=float, default=0.30, help="cosine similarity threshold for Qwen2-CLIP")
parser.add_argument("--qwen2-model", type=str, default="openai/clip-vit-large-patch14", help="HF model id for CLIP-like model (e.g., openai/clip-vit-large-patch14)")
parser.add_argument("--llm-timeout", type=int, default=600, help="timeout (seconds) for LLM.py")
args = parser.parse_args()

# --- Step 1: Run LLM target selection ---
print("[INFO] Running LLM target selection via cosine similarity...")
print(f"[INFO] prompt: {args.prompt}")

# Find optional bbox metadata if available
bbox_meta_path = None
for meta_name in ("llm_input.json", "bboxes.json"):
    meta_path = os.path.join(ROOT_DIR, meta_name)
    if os.path.exists(meta_path):
        bbox_meta_path = meta_path
        break

# Call select_targets directly
device_str = "cuda" if args.device == "gpu" else "cpu"
target_ids = select_targets(
    crops_dir="crops",
    prompt=args.prompt,
    threshold=args.similarity_threshold,
    device=device_str,
    qwen2_model=args.qwen2_model,
    bbox_meta=bbox_meta_path
)

print(f"[INFO] Selected target IDs: {target_ids}")

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
