# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j24sm_r150_m300_s0.5_a0.5_b0.9
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=30G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3/2018.12

python3 ../2_4_sensorimotor_tsp.py \
           --accessible_set_capacity 10000 \
           --distance_type Minkowski-3 \
           --max_sphere_radius 150 \
           --buffer_capacity 10 \
           --buffer_threshold 0.9 \
           --activation_threshold 0.5 \
           --length_factor 100 \
           --node_decay_median 300 \
           --node_decay_sigma 0.5 \
           --run_for_ticks 10000 \
