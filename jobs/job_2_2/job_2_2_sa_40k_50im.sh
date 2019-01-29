# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j22_40k_50im_sa
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=160G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../2_2_category_production_importance_pruned_tsa.py \
           -b 5000 \
           -c bbc \
           -f 0.8 \
           -i 0.05 \
           -d cosine \
           -l 1000 \
           -m log_co-occurrence \
           -n 0.99 \
           -p 50 \
           -r 5 \
           -s 0.4 \
           -t 3000 \
           -w 40000 
