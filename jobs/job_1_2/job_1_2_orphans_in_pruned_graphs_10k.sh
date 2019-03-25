#@IgnoreInspection BashAddShebang
#$ -S /bin/bash
#$ -q serial
#$ -N j12_10k_pruned_orphans
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=30G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../1_2_orphans_in_pruned_graphs.py \
           --corpus_name bbc \
           --distance_type cosine \
           --length_factor 1000 \
           --model_name log_co-occurrence \
           --radius 5 \
           --words 10000
