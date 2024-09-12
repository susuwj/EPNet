#!/usr/bin/env bash
TESTPATH=""
L_SCAN=$(seq 0 21)
L_SCAN_NAME=( 'scan1' 'scan4' 'scan9' 'scan10' 'scan11' 'scan12' 'scan13' 'scan15'
        'scan23' 'scan24' 'scan29' 'scan32' 'scan33' 'scan34' 'scan48' 'scan49'
        'scan62' 'scan75' 'scan77' 'scan110' 'scan114' 'scan118' )
for S in ${L_SCAN[@]}; do
  python fusion_dynamic.py --gpu_device '0' \
                   --outdir "" \
                   --pthresh '.6,.1,.1,.1,.1' \
                   --vthresh 11 \
                   --dist_base 26.0 \
                   --rel_diff_base 1300.0 \
                   --testlist=${L_SCAN_NAME[S]} \
                   --testpath=$TESTPATH
done

