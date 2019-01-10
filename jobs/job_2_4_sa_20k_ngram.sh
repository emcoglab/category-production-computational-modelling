# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j24_20k_ngram_sa
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=10G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../2_4_category_production_ngram_tsa.py 20000
