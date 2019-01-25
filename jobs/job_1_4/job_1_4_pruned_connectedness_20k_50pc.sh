#@IgnoreInspection BashAddShebang
#$ -S /bin/bash
#$ -q serial
#$ -N j14_20k_50pc_pruned_connectedness
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=60G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../1_4_disconnections_in_pruned_graphs.py 20000 50