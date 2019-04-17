"""
Builds some template jobs
"""
from os import path, mkdir

job_name = 'job_2_4'
short_name = "j24"
script_name = "2_4_sensorimotor_tsa"

if not path.isdir(job_name):
    mkdir(job_name)

ram_amount = 60
names = []

name = f"{job_name}_sensorimotor.sh"
names.append(name)
with open(path.join(job_name, name), mode="w", encoding="utf-8") as job_file:
    job_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
    job_file.write(f"#$ -S /bin/bash\n")
    job_file.write(f"#$ -q serial\n")
    job_file.write(f"#$ -N {short_name}_sm_sa\n")
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
    job_file.write(f"           --bailout {3000} \\\n")
    job_file.write(f"           --distance_type Minkowski-3 \\\n")
    job_file.write(f"           --pruning_length {100} \\\n")
    job_file.write(f"           --impulse_pruning_threshold 0.05 \\\n")
    job_file.write(f"           --length_factor 100 \\\n")
    job_file.write(f"           --node_decay_factor 0.99 \\\n")
    job_file.write(f"           --run_for_ticks 3000 \\\n")
with open(f"{job_name}_submit_ALL.sh", mode="w", encoding="utf-8") as batch_file:
    batch_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
    batch_file.write(f"#!/usr/bin/env bash\n")
    for name in names:
        batch_file.write(f"qsub {path.join(job_name, name)}\n")
