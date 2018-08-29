#$ -S /bin/bash
#$ -q serial
#$ -N j22_5k_cp_comparison
#$ -t 0-50:10
#$ -l h_vmem=2G

source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3

python3 ../2_2_category_production_comparison.py 5000 $SGE_TASK_ID
