#@IgnoreInspection BashAddShebang
#$ -S /bin/bash
#$ -q serial
#$ -N n5k_length_quantiles
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=20G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../1_3_pruning_quantile_lengths.py 5000
