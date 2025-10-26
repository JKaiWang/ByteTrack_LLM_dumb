import os

videos_dir = "videos/"
videos = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]

for v in videos:
    print(f"Processing {v} ...")
    os.system(
        f"python tools/demo_track.py video "
        f"-f exps/example/mot/yolox_x_mix_det.py "
        f"-c pretrained/bytetrack_x_mot17.pth "
        f"--fp16 --fuse --save_result "
        f"--path {os.path.join(videos_dir, v)}"
    )
