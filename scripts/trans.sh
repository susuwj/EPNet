#!/usr/bin/env bash
TESTLIST="all"
L_SCAN_NAME=( 'courtyard' 'delivery_area' 'electro' 'facade' 'kicker' 'meadow' 'office' 'pipes'
              'playground' 'relief' 'relief_2' 'terrace' 'terrains')
L_SCAN=$(seq 0 12)
#L_SCAN_NAME=('botanical_garden' 'boulders' 'bridge' 'door' 'exhibition_hall' 'lecture_room' #'living_room' 'lounge' 'observatory' 'old_computer' 'statue' 'terrace_2')
#L_SCAN=$(seq 0 11)
for S in ${L_SCAN[@]}; do
  python ply_transfer.py  --outdir "" \
                          --testlist=${L_SCAN_NAME[S]}
done




