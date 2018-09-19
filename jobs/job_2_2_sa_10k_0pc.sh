# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j21_10k_0pc_sa
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=20G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../2_2_category_production_importance_pruned_tsa.py 10000 0
