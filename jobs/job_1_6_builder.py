"""
Builds some template jobs
"""
from os import path, mkdir

job_name = 'job_1_6'
short_name = "j16"
script_name = "1_6_sensorimotor_neighbourhood_densities"

if not path.isdir(job_name):
    mkdir(job_name)

prune_ram = {
    20:  2,
    40:  2,
    60:  2,
    80:  2,
    100: 5,
    120: 10,
    140: 20,
    160: 30,
    180: 40,
    200: 60,
    220: 70,
    240: 100,
    260: 120,
}

# ---

names = []

for pruning_length, ram_amount in prune_ram.items():
    name = f"{job_name}_nbhd_{pruning_length}.sh"
    names.append(name)
    with open(path.join(job_name, name), mode="w", encoding="utf-8") as job_file:
        job_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
        job_file.write(f"#$ -S /bin/bash\n")
        job_file.write(f"#$ -q serial\n")
        job_file.write(f"#$ -N {short_name}_sm_p{pruning_length}_nbhd\n")
        job_file.write(f"#$ -m e\n")
        job_file.write(f"#$ -M c.wingfield@lancaster.ac.uk\n")
        job_file.write(f"#$ -l h_vmem={ram_amount}G\n")
        job_file.write(f"\n")
        job_file.write(f"source /etc/profile\n")
        job_file.write(f"\n")
        job_file.write(f"echo Job running on compute node `uname -n`\n")
        job_file.write(f"\n")
        job_file.write(f"module add anaconda3/2018.12\n")
        job_file.write(f"\n")
        job_file.write(f"python3 ../{script_name}.py \\\n")
        job_file.write(f"           --distance_type Minkowski-3 \\\n")
        job_file.write(f"           --pruning_length {pruning_length} \\\n")
        job_file.write(f"           --length_factor 100 \\\n")
with open(f"{job_name}_submit_ALL.sh", mode="w", encoding="utf-8") as batch_file:
    batch_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
    batch_file.write(f"#!/usr/bin/env bash\n")
    for name in names:
        batch_file.write(f"qsub {path.join(job_name, name)}\n")
