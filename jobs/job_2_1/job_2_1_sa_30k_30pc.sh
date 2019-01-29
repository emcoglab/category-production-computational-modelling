# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j21_30k_30pc_sa
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=90G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../2_1_category_production_pruned_tsa.py \
           -b 2000 \
           -c bbc \
           -f 0.8 \
           -i 0.05 \
           -d cosine \
           -l 1000 \
           -m log_co-occurrence \
           -n 0.99 \
           -p 30 \
           -r 5 \
           -s 0.4 \
           -t 3000 \
           -w 30000 
