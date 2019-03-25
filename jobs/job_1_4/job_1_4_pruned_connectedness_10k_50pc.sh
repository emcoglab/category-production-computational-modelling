#@IgnoreInspection BashAddShebang
#$ -S /bin/bash
#$ -q serial
#$ -N j14_10k_50pc_pruned_connectedness
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=20G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../1_4_disconnections_in_pruned_graphs.py \
           --corpus_name bbc \
           --distance_type cosine \
           --length_factor 1000 \
           --model_name log_co-occurrence \
           --radius 5 \
           --prune_percent 50 \
           --words 10000
