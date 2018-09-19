# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j21_20k_20pc_sa
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=55G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../2_2_category_production_importance_pruned_tsa.py 20000 20
