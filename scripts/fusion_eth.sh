#!/usr/bin/env bash
TESTPATH=""
L_SCAN_NAME=( 'courtyard' 'delivery_area' 'electro' 'facade' 'kicker' 'meadow' 'office' 'pipes'
              'playground' 'relief' 'relief_2' 'terrace' 'terrains')
L_SCAN=$(seq 0 12)
for S in ${L_SCAN[@]}; do
  python fusion.py --gpu_device '0' \
                   --outdir "" \
                   --pthresh '.8,.7,.7,.7,.8'\
                   --vthresh 2 \
                   --testlist=${L_SCAN_NAME[S]} \
                   --testpath=$TESTPATH
done

