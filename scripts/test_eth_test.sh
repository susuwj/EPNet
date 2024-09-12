#!/usr/bin/env bash
TESTPATH=""
CKPT_FILE='./ckpt/checkpoints_blendedmvs/model_000009.ckpt'
L_SCAN=$(seq 0 11)
L_SCAN_NAME=('botanical_garden' 'boulders' 'bridge' 'door' 'exhibition_hall' 'lecture_room' 'living_room' 'lounge' 'observatory' 'old_computer' 'statue' 'terrace_2')
for S in ${L_SCAN[@]}; do
  python test.py --dataset=eth_eval \
                 --gpu_device '0' \
                 --batch_size 1 \
                 --testpath=$TESTPATH  \
                 --testlist=${L_SCAN_NAME[S]}  \
                 --interval_scale 1.0 \
                 --resize_wh 2432,1600 \
                 --crop_wh 2432,1600 \
                 --num_views 10 \
                 --ndepths "32,16,8,8,8" \
                 --depth_inter_r "4,4,2,0.5,0.125" \
                 --num_groups "8,8,8,8,8" \
                 --numdepth 128 \
                 --loadckpt $CKPT_FILE \
                 --outdir "./results/outputs_bld_eth_test"
done
