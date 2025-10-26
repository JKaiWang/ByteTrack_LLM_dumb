import subprocess
import argparse
import os, sys

# 取得當前 run.py 所在的根目錄
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 把根目錄和 yolox 資料夾加進 Python 模組搜尋路徑
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "yolox"))

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True, help="Prompt 給 LLM，例如：找出穿粉紅色衣服的人")
args = parser.parse_args()

video_path = "videos/test.mp4"

cmd = [
    "python", "tools/demo_track.py", "video",
    "-f", "exps/example/mot/yolox_x_mix_det.py",
    "-c", "pretrained/bytetrack_x_mot17.pth.tar",
    "--fp16", "--fuse", "--save_result",
    "--path", video_path,
    "--track_thresh", "0.3",
    "--min_box_area", "5",
    "--prompt", args.prompt   # ✅ prompt 傳進去 demo_track.py
]

subprocess.run(cmd)
