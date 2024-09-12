#!/usr/bin/env bash
TESTPATH=""
CKPT_FILE=''
L_SCAN=$(seq 0 5)
L_SCAN_NAME=( 'Auditorium' 'Ballroom' 'Courtroom' 'Museum' 'Palace' 'Temple' )
for S in ${L_SCAN[@]}; do
  python test.py --dataset=ad_eval \
                 --gpu_device '0' \
                 --batch_size=1 \
                 --testpath=$TESTPATH  \
                 --testlist=${L_SCAN_NAME[S]}  \
                 --interval_scale 1.0 \
                 --resize_wh 1920,1080 \
                 --crop_wh 1920,1024 \
                 --num_views 11 \
                 --ndepths "32,16,8,8,8" \
                 --depth_inter_r "4,4,2,0.5,0.125" \
                 --num_groups "8,8,8,8,8" \
                 --numdepth 128 \
                 --loadckpt $CKPT_FILE \
                 --outdir ""
done



