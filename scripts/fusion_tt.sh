#!/usr/bin/env bash
TESTPATH=""
L_SCAN_NAME=( 'Family' 'Francis' 'Horse' 'Lighthouse' 'M60' 'Panther' 'Playground' 'Train' )
L_SCAN=$(seq 0 7)
for S in ${L_SCAN[@]}; do
  python fusion_dynamic.py --gpu_device '0' \
                   --outdir "" \
                   --pthresh '.8,.7,.7,.7,.8' \
                   --vthresh 11 \
                   --dist_base 8 \
                   --rel_diff_base 1300.0 \
                   --testlist=${L_SCAN_NAME[S]} \
                   --testpath=$TESTPATH
done




