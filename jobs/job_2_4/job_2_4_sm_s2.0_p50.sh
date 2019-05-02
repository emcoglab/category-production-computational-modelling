# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j24_sm_s2.0_p50_sa
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=2G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3/2018.12

python3 ../2_4_sensorimotor_tsa.py \
           --distance_type Minkowski-3 \
           --pruning_length 50 \
           --impulse_pruning_threshold 0.05 \
           --length_factor 100 \
           --node_decay_sigma 2.0 \
           --run_for_ticks 10000 \