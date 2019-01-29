# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j23_30k_ngram_sa
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=12G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../2_3_category_production_ngram_tsa.py \
           -b 15000 \
           -c bbc \
           -f 0.3 \
           -i 0.05 \
           -d cosine \
           -l 10 \
           -m log_ngram \
           -n 0.99 \
           -r 5 \
           -s 15 \
           -t 3000 \
           -w 30000 
