import os
import os.path as osp
import time
import cv2
import torch
import parser
import json
import sys
from LLM import select_targets
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as F
import gc
from loguru import logger

# --- Ensure project root on sys.path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import Dataset, DataLoader

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


class ListImgDataset(Dataset):
    def __init__(self, img_list) -> None:
        super().__init__()
        self.img_list = img_list
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(f_path)
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        return cur_img

    def init_img(self, img):
        # Don't copy the entire image - we'll use the original directly
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        resized = cv2.resize(img, (target_w, target_h))
        processed = F.normalize(F.to_tensor(resized), self.mean, self.std)
        processed = processed.unsqueeze(0)
        del resized  # Free resized image immediately
        return processed, img

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = self.load_img_from_file(img_path)
        processed_img, ori_img = self.init_img(img)
        return processed_img, ori_img, img_path


class Detector(object):
    def __init__(
        self,
        args,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False,
        seq_num=None,
    ):

        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        self.seq_num = seq_num
        self.args = args
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        img_list = os.listdir(
            os.path.join(
                self.args.rmot_path, "KITTI/training/image_02", self.seq_num[0]
            )
        )
        img_list = [
            os.path.join(
                self.args.rmot_path, "KITTI/training/image_02", self.seq_num[0], _
            )
            for _ in img_list
            if ("jpg" in _) or ("png" in _)
        ]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)

        self.json_path = os.path.join(
            self.args.rmot_path, "expression", seq_num[0], seq_num[1]
        )
        with open(self.json_path, "r") as f:
            json_info = json.load(f)
        self.json_info = json_info
        self.sentence = [json_info["sentence"]]

        checkpoint_id = int(args.resume.split("/")[-1].split(".")[0].split("t")[-1])
        self.save_path = os.path.join(
            self.args.output_dir,
            "results_epoch{}/{}/{}".format(
                checkpoint_id, seq_num[0], seq_num[1].split(".")[0]
            ),
        )
        os.makedirs(self.save_path, exist_ok=True)

        self.predict_path = os.path.join(self.args.output_dir, self.args.exp_name)
        os.makedirs(self.predict_path, exist_ok=True)
        if os.path.exists(os.path.join(self.predict_path, f"{self.seq_num}.txt")):
            os.remove(os.path.join(self.predict_path, f"{self.seq_num}.txt"))

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        processed_img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img_tensor = (
            torch.from_numpy(processed_img).unsqueeze(0).float().to(self.device)
        )
        del processed_img  # Free numpy array immediately
        if self.fp16:
            img_tensor = img_tensor.half()  # FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img_tensor)
            del img_tensor  # Free tensor immediately after use
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )

        return outputs, img_info

    def save_crops_from_detections(self, img_info, online_targets, frame_id, crops_dir):
        """Save cropped images of detected objects for LLM analysis."""
        os.makedirs(crops_dir, exist_ok=True)
        for t in online_targets:
            x1, y1, w, h = t.tlwh
            x2, y2 = int(x1 + w), int(y1 + h)
            x1, y1 = int(x1), int(y1)
            crop = img_info["raw_img"][y1:y2, x1:x2]
            if crop.size > 0:  # Make sure crop is valid
                save_path = os.path.join(crops_dir, f"i{t.track_id}_f{frame_id}.jpg")
                cv2.imwrite(save_path, crop)

    def _open_result_files(self, vis_folder, current_time, args):
        results_file_path = os.path.join(self.save_path, "predict.txt")
        results_file = open(results_file_path, "w", encoding="utf-8")

        vis_results_file = None
        vis_results_path = None
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            vis_results_path = osp.join(vis_folder, f"{timestamp}.txt")
            vis_results_file = open(vis_results_path, "w", encoding="utf-8")

        return results_file, vis_results_file, results_file_path, vis_results_path

    def _should_filter(self, frame_id, args):
        filter_every_n = getattr(args, "filter_every_n_frames", 1)
        if filter_every_n == 0:
            return frame_id == 0
        if filter_every_n > 0:
            return frame_id % filter_every_n == 0
        return False

    def _run_llm_selection(
        self, frame_crops_dir, frame_id, img_info, online_targets, args
    ):
        os.makedirs(frame_crops_dir, exist_ok=True)
        self.save_crops_from_detections(
            img_info, online_targets, frame_id, frame_crops_dir
        )
        device_str = "cuda" if str(self.device) == "cuda" else "cpu"
        selected_target_ids = select_targets(
            crops_dir=frame_crops_dir,
            prompt=(self.sentence[0] if hasattr(self, "sentence") else args.prompt),
            threshold=getattr(args, "similarity_threshold", 0.30),
            device=device_str,
            quiet=True,
        )
        logger.info(
            f"Frame {frame_id}: LLM filtered, selected IDs: {selected_target_ids}"
        )
        import shutil

        try:
            shutil.rmtree(frame_crops_dir)
        except Exception:
            pass
        return selected_target_ids

    def _process_targets_and_write(
        self,
        results_file,
        vis_results_file,
        frame_id,
        img_info,
        online_targets,
        selected_target_ids,
        args,
    ):
        online_tlwhs, online_ids, online_scores = [], [], []
        use_llm_selection = (hasattr(self, "sentence") and self.sentence) or (
            hasattr(args, "prompt") and args.prompt
        )
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id

            if (
                use_llm_selection
                and selected_target_ids
                and tid not in selected_target_ids
            ):
                continue

            vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)

                # Write result immediately to file in byetrack format
                formatted_line = (
                    f"{frame_id+1},{int(tid)},{float(tlwh[0])},{float(tlwh[1])},"
                    f"{float(tlwh[2])},{float(tlwh[3])},1,1,1\n"
                )
                results_file.write(formatted_line)

                # Write original result line to vis_results if needed
                if vis_results_file is not None:
                    result_line = f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    vis_results_file.write(result_line)

        return online_tlwhs, online_ids, online_scores

    def image_demo(self, predictor, vis_folder, current_time, args):
        # Don't use DataLoader - process images directly to save memory
        tracker = BYTETracker(args, frame_rate=args.fps)
        timer = Timer()

        (
            results_file,
            vis_results_file,
            results_file_path,
            vis_results_path,
        ) = self._open_result_files(vis_folder, current_time, args)

        crops_dir = os.path.join(self.save_path, "crops")
        use_llm_selection = (hasattr(self, "sentence") and self.sentence) or (
            hasattr(args, "prompt") and args.prompt
        )
        selected_target_ids = []

        seq_h, seq_w = None, None

        for frame_id, img_path in enumerate(tqdm(self.img_list)):
            # Read image directly without DataLoader buffering
            ori_img = cv2.imread(img_path)
            if ori_img is None:
                continue
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            if seq_h is None:
                seq_h, seq_w, _ = ori_img.shape

            outputs, img_info = predictor.inference(img_path, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(
                    outputs[0], [img_info["height"], img_info["width"]], exp.test_size
                )

                should_filter = self._should_filter(frame_id, args)

                if should_filter and use_llm_selection and len(online_targets) > 0:
                    frame_crops_dir = os.path.join(crops_dir, f"frame_{frame_id}")
                    selected_target_ids = self._run_llm_selection(
                        frame_crops_dir, frame_id, img_info, online_targets, args
                    )

                online_tlwhs, online_ids, online_scores = (
                    self._process_targets_and_write(
                        results_file,
                        vis_results_file,
                        frame_id,
                        img_info,
                        online_targets,
                        selected_target_ids,
                        args,
                    )
                )

                timer.toc()
                if args.save_result:
                    online_im = plot_tracking(
                        img_info["raw_img"],
                        online_tlwhs,
                        online_ids,
                        frame_id=frame_id,
                        fps=1.0 / max(1e-5, timer.average_time),
                    )
                else:
                    online_im = None
            else:
                timer.toc()
                online_im = img_info["raw_img"] if args.save_result else None

            if args.save_result and online_im is not None:
                timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                save_folder = osp.join(vis_folder, timestamp)
                os.makedirs(save_folder, exist_ok=True)
                cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

            # Clear image buffers to free memory immediately
            del ori_img
            if "raw_img" in img_info:
                del img_info["raw_img"]
            if "online_im" in locals() and online_im is not None:
                del online_im
            # Clear outputs from this frame
            if "outputs" in locals() and outputs[0] is not None:
                del outputs
            if "img_info" in locals():
                del img_info

            # Flush file buffer and run garbage collection more frequently
            if frame_id % 10 == 0:
                results_file.flush()
                if vis_results_file is not None:
                    vis_results_file.flush()
                gc.collect()  # Force garbage collection to free memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

        # Close the results files
        results_file.close()
        logger.info(f"Results written to {results_file_path}")

        if vis_results_file is not None:
            vis_results_file.close()
            logger.info(f"Vis results written to {vis_results_path}")
        gt_path = os.path.join(self.save_path, "gt.txt")
        self.write_gt(
            gt_path,
            self.json_path,
            os.path.join(
                self.args.rmot_path, "KITTI/labels_with_ids/image_02", self.seq_num[0]
            ),
            seq_h,
            seq_w,
        )

    def write_results_bytetrack(self, txt_path, results):
        save_format = "{frame},{id},{x1},{y1},{w},{h},1,1,1\n"
        # Results are formatted strings like "frame_id,tid,x,y,w,h,score,-1,-1,-1\n"
        # We need to parse and reformat to "frame,id,x1,y1,w,h,1,1,1\n"
        with open(txt_path, "w", encoding="utf-8") as f:
            for result in results:
                # Parse the CSV string
                parts = result.strip().split(",")
                if len(parts) >= 6:
                    frame_id, tid, x, y, w, h = parts[0:6]
                    line = save_format.format(
                        frame=int(frame_id) + 1,
                        id=int(tid),
                        x1=float(x),
                        y1=float(y),
                        w=float(w),
                        h=float(h),
                    )
                    f.write(line)

    # write ground-truth for each expression in a text. The text includes gt of all frames
    def write_gt(self, txt_path, json_file, gt_txt_file, im_height, im_width):
        save_format = "{frame},{id},{x1},{y1},{w},{h},1, 1, 1\n"

        with open(json_file) as f:
            json_info = json.load(f)

        with open(txt_path, "w") as f:
            for k in json_info["label"].keys():
                frame_id = int(k)
                if not os.path.isfile(
                    os.path.join(gt_txt_file, "{:06d}.txt".format(frame_id))
                ):
                    continue
                frame_gt = np.loadtxt(
                    os.path.join(gt_txt_file, "{:06d}.txt".format(frame_id))
                ).reshape(-1, 6)
                for frame_gt_line in frame_gt:
                    aa = json_info["label"][k]  # all gt from frame
                    aa = [int(a) for a in aa]
                    if int(frame_gt_line[1]) in aa:  # choose referent gt from all gt
                        track_id = int(frame_gt_line[1])
                        x1, y1, w, h = frame_gt_line[2:6]  # KITTI -> [x1, y1, w, h]
                        line = save_format.format(
                            frame=frame_id + 1,
                            id=track_id,
                            x1=x1 * im_width,
                            y1=y1 * im_height,
                            w=w * im_width,
                            h=h * im_height,
                        )
                        f.write(line)

        print("save gt to {}".format(txt_path))


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        ckpt_file = (
            args.ckpt
            if args.ckpt is not None
            else osp.join(output_dir, "best_ckpt.pth.tar")
        )
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # model.load_state_dict(ckpt["model"])
        # Fix state_dict loading error due to size mismatch
        # This handles cases where the number of classes in the pre-trained model differs from the current model.
        # Only load layers that match the current model's architecture.
        model_state_dict = model.state_dict()
        for k in ckpt["model"]:
            if (
                k in model_state_dict
                and ckpt["model"][k].shape == model_state_dict[k].shape
            ):
                model_state_dict[k] = ckpt["model"][k]
            else:
                logger.info(
                    f"Skipping loading of layer {k} due to size mismatch or not found."
                )
        model.load_state_dict(model_state_dict)
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()

    if args.trt:
        assert not args.fuse, "TensorRT model does not support fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(trt_file), "TensorRT model not found! Run tools/trt.py first."
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT for inference")
    else:
        trt_file = None
        decoder = None

    current_time = time.localtime()
    if args.demo == "image":
        for seq_num in seq_nums:
            predictor = Detector(
                args=args,
                model=model,
                exp=exp,
                trt_file=trt_file,
                decoder=decoder,
                device=args.device,
                fp16=args.fp16,
                seq_num=seq_num,
            )
            predictor.image_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = parser.make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    expressions_root = os.path.join(args.rmot_path, "expression")
    if "refer-kitti-v2" in args.rmot_path:
        video_ids = ["0005", "0011", "0013", "0019"]
    else:
        video_ids = ["0005", "0011", "0013"]

    seq_nums = []
    for video_id in video_ids:
        expression_jsons = sorted(os.listdir(os.path.join(expressions_root, video_id)))
        for expression_json in expression_jsons:
            seq_nums.append([video_id, expression_json])

    expression_num = len(seq_nums)

    print("Start inference")
    main(exp=exp, args=args)
