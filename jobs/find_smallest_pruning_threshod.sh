#$ -S /bin/bash
#$ -q serial
#$ -N find_pruning_thresholds
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=30G
#$ -t 0-9

source /etc/profile

echo Job running on compute node `uname -n`
echo Job task $SGE_TASK_ID

module add anaconda3

python3 ../find_smallest_pruning_threshold.py $SGE_TASK_ID
