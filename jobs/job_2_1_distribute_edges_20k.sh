#$ -S /bin/bash
#$ -q serial
#$ -N dist_edges_20k
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk
#$ -l h_vmem=20G

source /etc/profile

echo Job running on compute node `uname -n` 

module add anaconda3

python3 ../2_1_distribute_edgelists.py 20000
