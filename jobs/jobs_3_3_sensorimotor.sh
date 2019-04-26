#!/usr/bin/env bash

DATA_LOC="/Volumes/Data/spreading activation model/Model output/sensorimotor component"
FILE_PREFIX="Category production traces [sensorimotor Minkowski-3] length 100"

python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 100 df 0.99; pt 0.05; rft 3000; bailout 3000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 150 df 0.99; pt 0.05; rft 3000; bailout 3000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 50 df 0.99; pt 0.05; rft 5000; bailout 5000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 100 df 0.99; pt 0.05; rft 5000; bailout 5000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 150 df 0.99; pt 0.05; rft 5000; bailout 5000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 198 df 0.99; pt 0.05; rft 5000; bailout 5000" &
