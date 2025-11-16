# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

python3 bytetrack_inference.py \
image \
--exp_file exps/default/yolox_x.py \
--ckpt pretrained/yolox_x.pth \
--resume exps/bytetrack/checkpoint0052.pth \
--fp16 \
--fuse \
--output_dir exps/bytetrack \
--device gpu \
--rmot_path "./datasets/refer-kitti"  \
--filter_every_n_frames 15
# --save_result \
 #<- or testing refer-kitti v2
#&> log.txt 

# --output_dir exps/default >"exps/bytetrack/test_log.txt" & echo $! >"exps/bytetrack/test_pid.txt"


#  0051 --> v2
#  0052 --> v1
