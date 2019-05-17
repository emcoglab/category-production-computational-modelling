"""
Builds some template jobs
"""
from os import path, mkdir

from ldm.utils.maths import DistanceType

job_name = 'job_2_4'
short_name = "j24"
script_name = "2_4_sensorimotor_tsa"

if not path.isdir(job_name):
    mkdir(job_name)

prune_ram = {
    100: 5,
    150: 20,
    # 198 is the largest min edge length, so the threshold below which the graph becomes disconnected
    198: 45,
    200: 60,
    250: 120,
}
sigmas = [
    0.1,
    1.0,
]
buffer_entry_thresholds = [
    0.5,
    0.9,
]

run_for_ticks = 10_000

pruning_threshold = 0.05
length_factor = 100
buffer_size_limit = 10
distance_type = DistanceType.Minkowski3

# ---

names = []

for sphere_radius, ram_amount in prune_ram.items():
    for sigma in sigmas:
        for buffer_entry_threshold in buffer_entry_thresholds:
            name = f"{job_name}_sm_s{sigma}_b{buffer_entry_threshold}_r{sphere_radius}.sh"
            names.append(name)
            with open(path.join(job_name, name), mode="w", encoding="utf-8") as job_file:
                job_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
                job_file.write(f"#$ -S /bin/bash\n")
                job_file.write(f"#$ -q serial\n")
                job_file.write(f"#$ -N {short_name}_sm_s{sigma}_b{buffer_entry_threshold}_r{sphere_radius}_sa\n")
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
                job_file.write(f"           --distance_type {distance_type.name} \\\n")
                job_file.write(f"           --max_sphere_radius {sphere_radius} \\\n")
                job_file.write(f"           --buffer_size_limit {buffer_size_limit} \\\n")
                job_file.write(f"           --buffer_entry_threshold {buffer_entry_threshold} \\\n")
                job_file.write(f"           --buffer_pruning_threshold {pruning_threshold} \\\n")
                job_file.write(f"           --length_factor {length_factor} \\\n")
                job_file.write(f"           --node_decay_sigma {sigma} \\\n")
                job_file.write(f"           --run_for_ticks {run_for_ticks} \\\n")
with open(f"{job_name}_submit_ALL.sh", mode="w", encoding="utf-8") as batch_file:
    batch_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
    batch_file.write(f"#!/usr/bin/env bash\n")
    for name in names:
        batch_file.write(f"qsub {path.join(job_name, name)}\n")
