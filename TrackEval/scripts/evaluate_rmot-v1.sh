#python3 run_mot_challenge.py \
#--METRICS HOTA \
#--SEQMAP_FILE /data/wudongming/MOTR/seqmap_rmot_clean.txt \
#--SKIP_SPLIT_FOL True \
#--GT_FOLDER /data/Dataset/MOT17/images/train \
#--TRACKERS_FOLDER /data/wudongming/MOTR/exps/rmot_v2a/results_epoch249_ \
#--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
#--TRACKERS_TO_EVAL /data/wudongming/MOTR/exps/rmot_v2a/results_epoch249_ \
#--USE_PARALLEL True \
#--NUM_PARALLEL_CORES 2 \
#--SKIP_SPLIT_FOL True \
#--PLOT_CURVES False

python3 run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /home/seanachan/TempRMOT/datasets/data_path/seqmap-v1.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /datasets/refer-kitti/KITTI/training/image_02 \
--TRACKERS_FOLDER /home/seanachan/TempRMOT/exps/default_rk/results_epoch50 \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL /home/seanachan/TempRMOT/exps/default_rk/results_epoch50 \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False


#python3 run_mot_challenge.py \
#--METRICS HOTA \
#--SEQMAP_FILE /data/wudongming/MOTR/seqmap_kitti_clean.txt \
#--SKIP_SPLIT_FOL True \
#--GT_FOLDER /data/Dataset/KITTI/training/image_02 \
#--TRACKERS_FOLDER /data/wudongming/FairMOT/exp/fairmot_kitti_2/result_epoch100 \
#--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
#--TRACKERS_TO_EVAL /data/wudongming/FairMOT/exp/fairmot_kitti_2/result_epoch100 \
#--USE_PARALLEL True \
#--NUM_PARALLEL_CORES 2 \
#--SKIP_SPLIT_FOL True \
#--PLOT_CURVES False

