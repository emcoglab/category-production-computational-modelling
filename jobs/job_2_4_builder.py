"""
Builds some template jobs
"""
from os import path, mkdir

from ldm.utils.maths import DistanceType


def main():

    job_name = 'job_2_4'
    short_name = "j24"
    script_name = "2_4_sensorimotor_tsp"

    length_factor = 100

    distance_type = DistanceType.Minkowski3
    buffer_capacity = 10
    accessible_set_capacity = 3_000

    if not path.isdir(job_name):
        mkdir(job_name)

    prune_ram = {
        # 100: 5,
        150: 30,
        198: 55,  # 198 is the largest min edge length, so the threshold below which the graph becomes disconnected
        # 200: 60,
        250: 120,
    }
    # These very rough ranges taken from Mueller & Krawitz (2009)
    sigmas = [
        0.3,
        0.5,
        0.9,
    ]
    # A short distance is like 100, a long distance is like 700
    medians = [
        100,
        300,
        500,
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

    # ---

    names = []

    for sphere_radius, ram_amount in prune_ram.items():
        for median in medians:
            for sigma in sigmas:
                for accessible_set_threshold in activation_thresholds:
                    for buffer_threshold in buffer_thresholds:
                        if accessible_set_threshold > buffer_threshold:
                            continue
                        name = f"{short_name}sm_" \
                            f"r{sphere_radius}_" \
                            f"m{median}_" \
                            f"s{sigma}_" \
                            f"a{accessible_set_threshold}_" \
                            f"ac{accessible_set_capacity}_" \
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
                            job_file.write(f"           --accessible_set_capacity {accessible_set_capacity} \\\n")
                            job_file.write(f"           --distance_type {distance_type.name} \\\n")
                            job_file.write(f"           --max_sphere_radius {sphere_radius} \\\n")
                            job_file.write(f"           --buffer_capacity {buffer_capacity} \\\n")
                            job_file.write(f"           --buffer_threshold {buffer_threshold} \\\n")
                            job_file.write(f"           --accessible_set_threshold {accessible_set_threshold} \\\n")
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
