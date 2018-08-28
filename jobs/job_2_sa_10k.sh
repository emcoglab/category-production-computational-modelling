#@IgnoreInspection BashAddShebang
#$ -S /bin/bash
#$ -q serial
#$ -N j2_10k_sa
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=30G

source /etc/profile

echo Job running on compute node `uname -n` 

module add anaconda3

python3 ../3_category_production_tsa.py 10000
