"""
Builds some template jobs
"""
from dataclasses import dataclass
from os import path, mkdir

from ldm.utils.maths import DistanceType


@dataclass
class Spec:
    max_radius: int
    sigma: float
    median: int
    buffer_threshold: float
    accessible_set_threshold: float


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

    ram_requirement = {
        100: 5,
        150: 30,
        198: 55,  # 198 is the largest min edge length, so the threshold below which the graph becomes disconnected
        200: 60,
        250: 120,
    }

    run_for_ticks = 10_000

    # ---

    specs = [
        Spec(max_radius=198, buffer_threshold=0.9, accessible_set_threshold=0.5, median=500, sigma=0.3),
        Spec(max_radius=150, buffer_threshold=0.7, accessible_set_threshold=0.3, median=500, sigma=0.3),
        Spec(max_radius=198, buffer_threshold=0.9, accessible_set_threshold=0.3, median=500, sigma=0.9),
        Spec(max_radius=150, buffer_threshold=0.7, accessible_set_threshold=0.3, median=100, sigma=0.9),
    ]

    names = []

    for spec in specs:
        name = f"{short_name}sm_" \
            f"r{spec.max_radius}_" \
            f"m{spec.median}_" \
            f"s{spec.sigma}_" \
            f"a{spec.accessible_set_threshold}_" \
            f"ac{accessible_set_capacity}_" \
            f"b{spec.buffer_threshold}"
        names.append(name)
        with open(path.join(job_name, f"{name}.sh"), mode="w", encoding="utf-8") as job_file:
            job_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
            job_file.write(f"#$ -S /bin/bash\n")
            job_file.write(f"#$ -q serial\n")
            job_file.write(f"#$ -N {name}\n")
            job_file.write(f"#$ -m e\n")
            job_file.write(f"#$ -M c.wingfield@lancaster.ac.uk\n")
            job_file.write(f"#$ -l h_vmem={ram_requirement[spec.max_radius]}G\n")
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
            job_file.write(f"           --max_sphere_radius {spec.max_radius} \\\n")
            job_file.write(f"           --buffer_capacity {buffer_capacity} \\\n")
            job_file.write(f"           --buffer_threshold {spec.buffer_threshold} \\\n")
            job_file.write(f"           --accessible_set_threshold {spec.accessible_set_threshold} \\\n")
            job_file.write(f"           --length_factor {length_factor} \\\n")
            job_file.write(f"           --node_decay_median {spec.median} \\\n")
            job_file.write(f"           --node_decay_sigma {spec.sigma} \\\n")
            job_file.write(f"           --run_for_ticks {run_for_ticks} \\\n")
    with open(f"{job_name}_submit_ALL.sh", mode="w", encoding="utf-8") as batch_file:
        batch_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
        batch_file.write(f"#!/usr/bin/env bash\n")
        for name in names:
            batch_file.write(f"qsub {path.join(job_name, name)}.sh\n")

        
if __name__ == '__main__':
    main()
