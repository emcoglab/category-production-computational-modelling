# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j24_sm_s0.1_b0.9_r200_sa
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=60G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3/2018.12

python3 ../2_4_sensorimotor_tsa.py \
           --distance_type Minkowski-3 \
           --max_sphere_radius 200 \
           --buffer_size_limit 10 \
           --buffer_entry_threshold 0.9 \
           --buffer_pruning_threshold 0.05 \
           --length_factor 100 \
           --node_decay_sigma 0.1 \
           --run_for_ticks 10000 \