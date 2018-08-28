#@IgnoreInspection BashAddShebang
#$ -S /bin/bash
#$ -q serial
#$ -N j2_25k_sa
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=150G

source /etc/profile

echo Job running on compute node `uname -n` 

module add anaconda3

python3 ../3_category_production_tsa.py 25000
