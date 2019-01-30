# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j22_3k_100im_sa
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=5G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../2_2_category_production_importance_pruned_tsa.py \
           --bailout 5000 \
           --corpus_name bbc \
           --firing_threshold 0.8 \
           --impulse_pruning_threshold 0.05 \
           --distance_type cosine \
           --length_factor 1000 \
           --model_name log_co-occurrence \
           --node_decay_factor 0.99 \
           --prune_importance 100 \
           --radius 5 \
           --edge_decay_sd_factor 0.4 \
           --run_for_ticks 3000 \
           --words 3000 
