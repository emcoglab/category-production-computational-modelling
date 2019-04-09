#!/usr/bin/env bash

DATA_LOC="/Volumes/Data/spreading activation model/Model output/CP n-gram param search"
PREFIX="Category production traces"

python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.2; sd_factor 8.0; length 10; model [PMI n-gram (BBC), r=5])" 0.2 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.2; sd_factor 10.0; length 10; model [PMI n-gram (BBC), r=5])" 0.2 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.2; sd_factor 15.0; length 10; model [PMI n-gram (BBC), r=5])" 0.2 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.2; sd_factor 20.0; length 10; model [PMI n-gram (BBC), r=5])" 0.2 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.2; sd_factor 25.0; length 10; model [PMI n-gram (BBC), r=5])" 0.2 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.3; sd_factor 8.0; length 10; model [PMI n-gram (BBC), r=5])" 0.3 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.3; sd_factor 10.0; length 10; model [PMI n-gram (BBC), r=5])" 0.3 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.3; sd_factor 15.0; length 10; model [PMI n-gram (BBC), r=5])" 0.3 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.3; sd_factor 20.0; length 10; model [PMI n-gram (BBC), r=5])" 0.3 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.3; sd_factor 25.0; length 10; model [PMI n-gram (BBC), r=5])" 0.3 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.4; sd_factor 8.0; length 10; model [PMI n-gram (BBC), r=5])" 0.4 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.4; sd_factor 10.0; length 10; model [PMI n-gram (BBC), r=5])" 0.4 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.4; sd_factor 15.0; length 10; model [PMI n-gram (BBC), r=5])" 0.4 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.4; sd_factor 20.0; length 10; model [PMI n-gram (BBC), r=5])" 0.4 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.4; sd_factor 25.0; length 10; model [PMI n-gram (BBC), r=5])" 0.4 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.5; sd_factor 8.0; length 10; model [PMI n-gram (BBC), r=5])" 0.5 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.5; sd_factor 10.0; length 10; model [PMI n-gram (BBC), r=5])" 0.5 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.5; sd_factor 15.0; length 10; model [PMI n-gram (BBC), r=5])" 0.5 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.5; sd_factor 20.0; length 10; model [PMI n-gram (BBC), r=5])" 0.5 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.5; sd_factor 25.0; length 10; model [PMI n-gram (BBC), r=5])" 0.5 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.6; sd_factor 8.0; length 10; model [PMI n-gram (BBC), r=5])" 0.6 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.6; sd_factor 10.0; length 10; model [PMI n-gram (BBC), r=5])" 0.6 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.6; sd_factor 15.0; length 10; model [PMI n-gram (BBC), r=5])" 0.6 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.6; sd_factor 20.0; length 10; model [PMI n-gram (BBC), r=5])" 0.6 &
python ../3_1_category_production_comparison.py "${DATA_LOC}/${PREFIX} (40,000 words; firing 0.6; sd_factor 25.0; length 10; model [PMI n-gram (BBC), r=5])" 0.6 &
