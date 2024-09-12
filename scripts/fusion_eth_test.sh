#!/usr/bin/env bash
TESTPATH=""
L_SCAN_NAME=('botanical_garden' 'boulders' 'bridge' 'door' 'exhibition_hall' 'lecture_room' 'living_room' 'lounge' 'observatory' 'old_computer' 'statue' 'terrace_2')
L_SCAN=$(seq 0 11)
for S in ${L_SCAN[@]}; do
  python fusion.py --gpu_device '0' \
                   --outdir "" \
                   --filter_method "vis_fusion" \
                   --pthresh '.8,.7,.7,.7,.8' \
                   --vthresh 2 \
                   --testlist=${L_SCAN_NAME[S]} \
                   --testpath=$TESTPATH
done
