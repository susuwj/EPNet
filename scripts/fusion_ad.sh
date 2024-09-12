#!/usr/bin/env bash
TESTPATH=""
L_SCAN=$(seq 0 5)
L_SCAN_NAME=( 'Auditorium' 'Ballroom' 'Courtroom' 'Museum' 'Palace' 'Temple' )
for S in ${L_SCAN[@]}; do
  python fusion_dynamic.py --gpu_device '0' \
                   --outdir "" \
                   --pthresh '.8,.1,.1,.1,.1' \
                   --vthresh 5 \
                   --dist_base 2.0 \
                   --rel_diff_base 1000.0 \
                   --testlist=${L_SCAN_NAME[S]} \
                   --testpath=$TESTPATH
done
