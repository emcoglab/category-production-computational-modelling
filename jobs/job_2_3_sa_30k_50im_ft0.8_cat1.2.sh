# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j23_30k_50im_ft0.8_cat1.2
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=90G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../2_3_category_production_importance_pruned_tsa.py 30000 50 0.8 1.2
