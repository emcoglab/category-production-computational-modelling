#@IgnoreInspection BashAddShebang
#$ -S /bin/bash
#$ -q serial
#$ -N j13_10k_length_quantiles
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=20G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../1_3_pruning_quantile_lengths.py \
           --corpus_name bbc \
           --distance_type cosine \
           --length_factor 1000 \
           --model_name log_co-occurrence \
           --radius 5 \
           --words 10000