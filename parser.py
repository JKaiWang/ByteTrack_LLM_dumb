import argparse

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo")
    parser.add_argument("demo", default="image", help="demo type: image | video | webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # Path to images or video
    parser.add_argument("--path", default="./videos/palace.mp4", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam camera id")

    parser.add_argument("--save_result", action="store_true", help="save visualization results")

    # Exp / ckpt / device
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint for eval")
    parser.add_argument("--device", default="gpu", type=str, help="cpu | gpu")
    parser.add_argument("--conf", default=None, type=float, help="test confidence threshold")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test image size (square)")
    parser.add_argument("--fps", default=30, type=int, help="fallback FPS if video metadata missing")

    # Inference toggles
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="mixed precision")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="fuse conv+bn")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="use TensorRT")

    # Tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="frames to keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="filter overly tall boxes")
    parser.add_argument("--min_box_area", type=float, default=10, help="filter tiny boxes")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test MOT20 dataset")

    # end-to-end rmot settings
    parser.add_argument('--rmot_path', default='./datasets/refer-kitti-v2', type=str)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--exp_name', default='submit', type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    
    # LLM filtering settings
    parser.add_argument('--filter_every_n_frames', type=int, default=1,
                        help='Run LLM filtering every N frames (1=every frame, 0=first frame only)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt for LLM target selection')
    parser.add_argument('--similarity_threshold', type=float, default=0.30,
                        help='Similarity threshold for LLM target selection')
    
    return parser
