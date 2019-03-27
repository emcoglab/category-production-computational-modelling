# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j23_30k_f0.4_s15_ngram_sa
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=12G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../2_3_category_production_ngram_tsa.py \
           --bailout 15000 \
           --corpus_name bbc \
           --firing_threshold 0.4 \
           --impulse_pruning_threshold 0.05 \
           --distance_type cosine \
           --length_factor 10 \
           --model_name pmi_ngram \
           --node_decay_factor 0.99 \
           --radius 5 \
           --edge_decay_sd_factor 15 \
           --run_for_ticks 3000 \
           --words 30000 
