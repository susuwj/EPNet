#!/usr/bin/env bash
MVS_TRAINING=""

LOG_DIR=''
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi
LOAD_DIR=''

python train.py --logdir $LOG_DIR \
                --gpu_device '1' \
                --dataset 'blendedmvs' \
                --batch_size 2 \
                --trainpath=$MVS_TRAINING \
                --trainlist /mnt/B/BlendedMVS/BlendedMVS_training.txt \
                --testlist /mnt/B/BlendedMVS/validation_list.txt \
                --num_views 9 \
                --interval_scale 1.0 \
                --ndepths "32,16,8,8,8" \
                --depth_inter_r "4,4,2,0.5,0.125" \
                --num_groups "8,8,8,8,8" \
                --numdepth 128 \
                --resize_wh 768,576 \
                --crop_wh 640,512 \
                --robust_train \
                --epochs 10 \
                --lr 0.0001 \
                --lrepochs "6,8:2" \
                --ph_w 5.0 \
                --loadckpt $LOAD_DIR
