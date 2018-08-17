#@IgnoreInspection BashAddShebang
#$ -S /bin/bash
#$ -q serial
#$ -N connectedness_of_pruned_graphs
#$ -l h_vmem=30G
#$ -t 1-10:1

source /etc/profile

echo Job running on compute node `uname -n`
echo Job task $SGE_TASK_ID

module add anaconda3

python3 ../connectedness_of_pruned_graphs.py $SGE_TASK_ID
