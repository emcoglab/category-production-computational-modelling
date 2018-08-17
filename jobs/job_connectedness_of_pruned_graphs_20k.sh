#@IgnoreInspection BashAddShebang
#$ -S /bin/bash
#$ -q serial
#$ -N 20k_pruned_connectedness
#$ -l h_vmem=100G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../connectedness_of_pruned_graphs.py 20000
