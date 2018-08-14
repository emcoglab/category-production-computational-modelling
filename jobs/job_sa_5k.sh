#$ -S /bin/bash
#$ -q serial
#$ -N cw_sa_5k
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=10G

source /etc/profile

echo Job running on compute node `uname -n` 

module add anaconda3

python3 spreading_activation/1_category_production_tsa.py 5000
