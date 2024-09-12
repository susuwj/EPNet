#!/usr/bin/env bash
MVS_TRAINING=""
LOG_DIR=''
GROUP_DIM="8,8,8,8,8"
DEPTH_INTER="4,4,2,0.5,0.125"
HYPO_NUM="32,16,8,8,8"
GPU_NUM="0"
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python train.py --logdir $LOG_DIR \
                --gpu_device=$GPU_NUM \
                --dataset 'dtu_yao' \
                --batch_size 4 \
                --trainpath=$MVS_TRAINING \
                --trainlist ./lists/dtu/train.txt \
                --testlist ./lists/dtu/test.txt \
                --num_views 5 \
                --interval_scale 1.605 \
                --ndepths=$HYPO_NUM \
                --depth_inter_r=$DEPTH_INTER \
                --num_groups=$GROUP_DIM \
                --numdepth 128 \
                --resize_wh 800,600 \
                --crop_wh 640,512 \
                --epochs 1 \
                --lrepochs "8,10,11:2"\
                --robust_train \
                --ph_w 20.0 \
                --backbone_only

python train.py --logdir $LOG_DIR \
                --gpu_device=$GPU_NUM \
                --dataset 'dtu_yao' \
                --batch_size 12 \
                --trainpath=$MVS_TRAINING \
                --trainlist ./lists/dtu/train.txt \
                --testlist ./lists/dtu/test.txt \
                --num_views 5 \
                --interval_scale 1.605 \
                --ndepths=$HYPO_NUM \
                --depth_inter_r=$DEPTH_INTER \
                --num_groups $GROUP_DIM \
                --numdepth 128 \
                --resize_wh 800,600 \
                --crop_wh 640,512 \
                --epochs 3 \
                --lrepochs "8,10,11:2"\
                --robust_train \
                --ph_w 20.0 \
                --offsetnet_only \
                --resume


python train.py --logdir $LOG_DIR \
                --gpu_device=$GPU_NUM \
                --dataset 'dtu_yao' \
                --batch_size 4 \
                --trainpath=$MVS_TRAINING \
                --trainlist ./lists/dtu/train.txt \
                --testlist ./lists/dtu/test.txt \
                --num_views 5 \
                --interval_scale 1.605 \
                --ndepths=$HYPO_NUM \
                --depth_inter_r=$DEPTH_INTER \
                --num_groups=$GROUP_DIM \
                --numdepth 128 \
                --resize_wh 800,600 \
                --crop_wh 640,512 \
                --epochs 12 \
                --lrepochs "8,10,11:2"\
                --robust_train \
                --ph_w 20.0 \
                --resume

