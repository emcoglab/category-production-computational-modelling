#!/usr/bin/env bash

DATA_LOC="/Volumes/Data/spreading activation model/Model output/sensorimotor component with buffer 2019-05-29"
FILE_PREFIX="Category production traces [sensorimotor Minkowski-3] length 100"

python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 250 sigma 1.0; bet 0.9; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 250 sigma 0.1; bet 0.9; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 250 sigma 1.0; bet 0.5; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 250 sigma 0.1; bet 0.5; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 200 sigma 1.0; bet 0.9; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 200 sigma 1.0; bet 0.5; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 200 sigma 0.1; bet 0.9; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 198 sigma 1.0; bet 0.9; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 200 sigma 0.1; bet 0.5; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 198 sigma 1.0; bet 0.5; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 198 sigma 0.1; bet 0.9; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 198 sigma 0.1; bet 0.5; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 150 sigma 1.0; bet 0.9; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 150 sigma 1.0; bet 0.5; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 150 sigma 0.1; bet 0.9; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 150 sigma 0.1; bet 0.5; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 100 sigma 1.0; bet 0.9; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 100 sigma 1.0; bet 0.5; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 100 sigma 0.1; bet 0.9; bpt 0.05; rft 10000; bailout None" &
python ../3_3_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX}, max r 100 sigma 0.1; bet 0.5; bpt 0.05; rft 10000; bailout None" &
