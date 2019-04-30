#!/usr/bin/env bash

DATA_LOC="/Volumes/Data/spreading activation model/Model output/sensorimotor component lognormal decay"
FILE_PREFIX="Category production traces [sensorimotor Minkowski-3] length 100"

python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 100 sigma 0.1; pt 0.05; rft 10000; bailout 1000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 100 sigma 0.2; pt 0.05; rft 10000; bailout 1000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 100 sigma 0.3; pt 0.05; rft 10000; bailout 1000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 150 sigma 0.1; pt 0.05; rft 10000; bailout 1000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 150 sigma 0.2; pt 0.05; rft 10000; bailout 1000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 150 sigma 0.3; pt 0.05; rft 10000; bailout 1000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 198 sigma 0.1; pt 0.05; rft 10000; bailout 1000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 198 sigma 0.2; pt 0.05; rft 10000; bailout 1000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 198 sigma 0.3; pt 0.05; rft 10000; bailout 1000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 200 sigma 0.1; pt 0.05; rft 10000; bailout 1000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 200 sigma 0.2; pt 0.05; rft 10000; bailout 1000" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, pruning at 200 sigma 0.3; pt 0.05; rft 10000; bailout 1000" &
