#@IgnoreInspection BashAddShebang
#$ -S /bin/bash
#$ -q serial
#$ -N n1k_pruned_connectedness
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=2G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../1_2_orphans_in_pruned_graphs.py 1000
