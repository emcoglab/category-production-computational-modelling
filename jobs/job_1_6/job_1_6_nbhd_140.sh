# GENERATED CODE, CHANGES WILL BE OVERWRITTEN
#$ -S /bin/bash
#$ -q serial
#$ -N j16_sm_p140_nbhd
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=20G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3/2018.12

python3 ../1_6_sensorimotor_neighbourhood_densities.py \
           --distance_type Minkowski-3 \
           --pruning_length 140 \
           --length_factor 100 \