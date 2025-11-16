#!/bin/bash

# Specify custom output directory and FPS
# python draw_gt_boxes.py --gt_dir exps/bytetrack/results_epoch52/0011 --output_dir videos --fps 15

# Process with custom image directory
python draw_gt_boxes.py --gt_file exps/bytetrack/results_epoch52/0011/black-cars-in-the-left/gt.txt \
    --images_dir /home/seanachan/data/Dataset/refer-kitti/KITTI/training/image_02 \
    --output_dir videos