# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j21_3k_0pc_sa
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=3G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../2_1_category_production_pruned_tsa.py 3000 0