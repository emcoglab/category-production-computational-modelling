"""
Builds some template jobs
"""
from os import path, mkdir

from ldm.utils.maths import DistanceType

def main():

    job_name = 'job_2_4'
    short_name = "j24"
    script_name = "2_4_sensorimotor_tsp"

    if not path.isdir(job_name):
        mkdir(job_name)

    prune_ram = {
        # 100: 5,
        150: 20,
        # 198 is the largest min edge length, so the threshold below which the graph becomes disconnected
        198: 45,
        # 200: 60,
        250: 120,
    }
    sigmas = [
        # This very rough range taken from Mueller & Krawitz (2009)
        0.3,
        0.5,
    ]
    medians = [
        # This very rough range taken from Mueller & Krawitz (2009)
        3,
        5,
    ]
    buffer_thresholds = [
        0.7,
        0.9,
    ]
    activation_thresholds = [
        0.3,
        0.5,
    ]

    run_for_ticks = 10_000

    length_factor = 100
    buffer_size_limit = 10
    distance_type = DistanceType.Minkowski3

    # ---

    names = []

    for sphere_radius, ram_amount in prune_ram.items():
        for median in medians:
            for sigma in sigmas:
                for activation_threshold in activation_thresholds:
                    for buffer_threshold in buffer_thresholds:
                        if activation_threshold > buffer_threshold:
                            continue
                        name = f"{short_name}sm_" \
                            f"r{sphere_radius}_" \
                            f"m{median}_" \
                            f"s{sigma}_" \
                            f"a{activation_threshold}_" \
                            f"b{buffer_threshold}"
                        names.append(name)
                        with open(path.join(job_name, f"{name}.sh"), mode="w", encoding="utf-8") as job_file:
                            job_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
                            job_file.write(f"#$ -S /bin/bash\n")
                            job_file.write(f"#$ -q serial\n")
                            job_file.write(f"#$ -N {name}\n")
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
                            job_file.write(f"           --buffer_threshold {buffer_threshold} \\\n")
                            job_file.write(f"           --activation_threshold {activation_threshold} \\\n")
                            job_file.write(f"           --length_factor {length_factor} \\\n")
                            job_file.write(f"           --node_decay_median {median} \\\n")
                            job_file.write(f"           --node_decay_sigma {sigma} \\\n")
                            job_file.write(f"           --run_for_ticks {run_for_ticks} \\\n")
    with open(f"{job_name}_submit_ALL.sh", mode="w", encoding="utf-8") as batch_file:
        batch_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
        batch_file.write(f"#!/usr/bin/env bash\n")
        for name in names:
            batch_file.write(f"qsub {path.join(job_name, name)}.sh\n")

        
if __name__ == '__main__':
    main()
