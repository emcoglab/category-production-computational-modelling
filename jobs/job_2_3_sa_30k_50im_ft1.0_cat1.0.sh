# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j23_30k_50im_ft1.0_cat1.0
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=90G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../2_3_category_production_parameter_search.py 30000 50 1.0 1.0
