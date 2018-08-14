#$ -S /bin/bash
#$ -q serial
#$ -N cw_sa_10k
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=30G

source /etc/profile

echo Job running on compute node `uname -n` 

module add anaconda3

python3 ../1_category_production_tsa.py 10000
