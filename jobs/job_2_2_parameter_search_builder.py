"""
Builds some template jobs
"""
from os import path, mkdir

job_name = 'job_2_2_ps'
short_name = "j22ps"
script_name = "2_2_category_production_importance_pruned_tsa"

if not path.isdir(job_name):
    mkdir(job_name)

number_of_words = 30_000
importance_threshold = 50
ram_amount = 100
firing_thresholds = [
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
]

names = []
k = f"{int(number_of_words/1000)}k"
for firing_threshold in firing_thresholds:
    name = f"{job_name}_{k}_{importance_threshold}im_ft{firing_threshold}.sh"
    names.append(name)
    with open(path.join(job_name, name), mode="w", encoding="utf-8") as job_file:
        job_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
        job_file.write(f"#$ -S /bin/bash\n")
        job_file.write(f"#$ -q serial\n")
        job_file.write(f"#$ -N {short_name}_{k}_{importance_threshold}im_ft{firing_threshold}\n")
        job_file.write(f"#$ -m e\n")
        job_file.write(f"#$ -M c.wingfield@lancaster.ac.uk\n")
        job_file.write(f"#$ -l h_vmem={ram_amount}G\n")
        job_file.write(f"\n")
        job_file.write(f"source /etc/profile\n")
        job_file.write(f"\n")
        job_file.write(f"echo Job running on compute node `uname -n`\n")
        job_file.write(f"\n")
        job_file.write(f"module add anaconda3\n")
        job_file.write(f"\n")
        job_file.write(f"python3 ../{script_name}.py \\\n")
        job_file.write(f"           -b 2000 \\\n")
        job_file.write(f"           -c bbc \\\n")
        job_file.write(f"           -f {firing_threshold} \\\n")
        job_file.write(f"           -i 0.05 \\\n")
        job_file.write(f"           -d cosine \\\n")
        job_file.write(f"           -l 1000 \\\n")
        job_file.write(f"           -m log_co-occurrence \\\n")
        job_file.write(f"           -n 0.99 \\\n")
        job_file.write(f"           -p {importance_threshold} \\\n")
        job_file.write(f"           -r 5 \\\n")
        job_file.write(f"           -s 0.4 \\\n")
        job_file.write(f"           -t 3000 \\\n")
        job_file.write(f"           -w {int(number_of_words)} \\\n")
with open(f"{job_name}_submit_ALL.sh", mode="w", encoding="utf-8") as batch_file:
    batch_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
    batch_file.write(f"#!/usr/bin/env bash\n")
    for name in names:
        batch_file.write(f"qsub {path.join(job_name, name)}\n")
