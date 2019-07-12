# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j24sm_r250_m500_s0.3_a0.3_b0.7
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=120G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3/2018.12

python3 ../2_4_sensorimotor_tsp.py \
           --buffer_capacity 10000 \
           --distance_type Minkowski-3 \
           --max_sphere_radius 250 \
           --buffer_size_limit 10 \
           --buffer_threshold 0.7 \
           --activation_threshold 0.3 \
           --length_factor 100 \
           --node_decay_median 500 \
           --node_decay_sigma 0.3 \
           --run_for_ticks 10000 \
