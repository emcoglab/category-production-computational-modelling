# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j24_sm_s1.0_a0.0_b0.9_r198
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=45G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3/2018.12

python3 ../2_4_sensorimotor_tsp.py \
           --distance_type Minkowski-3 \
           --max_sphere_radius 198 \
           --buffer_size_limit 10 \
           --buffer_entry_threshold 0.9 \
           --buffer_pruning_threshold 0.2 \
           --activation_threshold 0.0 \
           --length_factor 100 \
           --node_decay_sigma 1.0 \
           --run_for_ticks 10000 \
