#!/usr/bin/env bash
TESTPATH=""
CKPT_FILE=''
L_SCAN=$(seq 0 21)
L_SCAN_NAME=( 'scan1' 'scan4' 'scan9' 'scan10' 'scan11' 'scan12' 'scan13' 'scan15'
        'scan23' 'scan24' 'scan29' 'scan32' 'scan33' 'scan34' 'scan48' 'scan49'
        'scan62' 'scan75' 'scan77' 'scan110' 'scan114' 'scan118' )
for S in ${L_SCAN[@]}; do
  python test.py --dataset=general_eval \
                 --gpu_device '0' \
                 --batch_size=1 \
                 --testpath=$TESTPATH  \
                 --testlist=${L_SCAN_NAME[S]}  \
                 --interval_scale 1.605 \
                 --resize_wh 1600,1200 \
                 --crop_wh 1600,1152 \
                 --num_view 5 \
                 --ndepths "32,16,8,8,8" \
                 --depth_inter_r "4,4,2,0.5,0.125" \
                 --num_groups "8,8,8,8,8" \
                 --numdepth 128 \
                 --loadckpt $CKPT_FILE \
                 --outdir ""
done
