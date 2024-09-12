#!/usr/bin/env bash
TESTPATH=""
CKPT_FILE=''
L_SCAN_NAME=( 'courtyard' 'delivery_area' 'electro' 'facade' 'kicker' 'meadow' 'office' 'pipes'
              'playground' 'relief' 'relief_2' 'terrace' 'terrains')
L_SCAN=$(seq 0 12)
for S in ${L_SCAN[@]}; do
  python test.py --dataset=eth_eval \
                 --gpu_device '0' \
                 --batch_size 1 \
                 --testpath=$TESTPATH  \
                 --testlist=${L_SCAN_NAME[S]}  \
                 --interval_scale 1.0 \
                 --resize_wh 2432,1600 \
                 --crop_wh 2432,1600 \
                 --num_views 7 \
                 --ndepths "32,16,8,8,8" \
                 --depth_inter_r "4,4,2,0.5,0.125" \
                 --num_groups "8,8,8,8,8" \
                 --numdepth 128  \
                 --loadckpt $CKPT_FILE \
                 --outdir ""
done

