#$ -S /bin/bash
#$ -q serial
#$ -N sa_3k
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=5G

source /etc/profile

echo Job running on compute node `uname -n` 

module add anaconda3

python3 ../1_category_production_tsa.py 3000