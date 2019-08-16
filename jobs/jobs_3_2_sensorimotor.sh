#!/usr/bin/env bash

DATA_LOC="/Volumes/Data/spreading activation model/Model output/sensorimotor component 2019-08-01 as-dampening, presynaptic cap"
FILE_PREFIX="Category production traces [sensorimotor Minkowski-3]"

pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 150 m 100.0; s 0.9; a 0.3; ac 3000; b 0.7; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 150 m 100.0; s 0.9; a 0.3; ac 3000; b 0.9; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 150 m 500.0; s 0.3; a 0.3; ac 3000; b 0.7; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 150 m 500.0; s 0.3; a 0.3; ac 3000; b 0.9; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 150 m 500.0; s 0.3; a 0.5; ac 3000; b 0.7; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 150 m 500.0; s 0.3; a 0.5; ac 3000; b 0.9; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 150 m 500.0; s 0.9; a 0.3; ac 3000; b 0.7; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 150 m 500.0; s 0.9; a 0.3; ac 3000; b 0.9; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 150 m 500.0; s 0.9; a 0.5; ac 3000; b 0.7; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 150 m 500.0; s 0.9; a 0.5; ac 3000; b 0.9; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 198 m 500.0; s 0.3; a 0.3; ac 3000; b 0.7; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 198 m 500.0; s 0.3; a 0.3; ac 3000; b 0.9; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 198 m 500.0; s 0.3; a 0.5; ac 3000; b 0.7; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 198 m 500.0; s 0.3; a 0.5; ac 3000; b 0.9; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 198 m 500.0; s 0.9; a 0.3; ac 3000; b 0.7; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 198 m 500.0; s 0.9; a 0.3; ac 3000; b 0.9; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 198 m 500.0; s 0.9; a 0.5; ac 3000; b 0.7; bc 10; attenuate Prevalence; rft 10000; bailout None" &
pythonw ../3_2_category_production_comparison_sensorimotor.py "${DATA_LOC}/${FILE_PREFIX} r 198 m 500.0; s 0.9; a 0.5; ac 3000; b 0.9; bc 10; attenuate Prevalence; rft 10000; bailout None" &
