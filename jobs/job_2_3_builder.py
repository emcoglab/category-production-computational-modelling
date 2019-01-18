"""
Builds some template jobs
"""
from os import path, mkdir

job_name = 'job_2_3'
if not path.isdir(job_name):
    mkdir(job_name)

number_of_words = 30_000
importance_threshold = 50
ram_amount = 100
thresholds = {
    # firing: conscious-access
    0.6: [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    0.7: [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    0.8: [0.8, 0.9, 1.0, 1.1, 1.2],
    0.9: [0.9, 1.0, 1.1, 1.2],
    1.0: [1.0, 1.1, 1.2],
    1.1: [1.1, 1.2],
    1.2: [1.2],
}
firing_thresholds = sorted(thresholds.keys())

names = []
k = f"{int(number_of_words/1000)}k"
for firing_threshold in firing_thresholds:
    conscious_access_thresholds = thresholds[firing_threshold]
    for conscious_access_threshold in conscious_access_thresholds:
        name = f"job_2_3_sa_{k}_{importance_threshold}im_ft{firing_threshold}_cat{conscious_access_threshold}.sh"
        names.append(name)
        with open(path.join(job_name, name), mode="w", encoding="utf-8") as job_file:
            job_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
            job_file.write(f"#$ -S /bin/bash\n")
            job_file.write(f"#$ -q serial\n")
            job_file.write(f"#$ -N j23_{k}_{importance_threshold}im_ft{firing_threshold}_cat{conscious_access_threshold}\n")
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
            job_file.write(f"python3 ../../2_3_category_production_parameter_search.py"
                           f" {int(number_of_words)}"
                           f" {int(importance_threshold)}"
                           f" {float(firing_threshold)}"
                           f" {float(conscious_access_threshold)}\n")
with open("job_2_3_submit_ALL.sh", mode="w", encoding="utf-8") as batch_file:
    batch_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
    batch_file.write(f"#!/usr/bin/env bash\n")
    for name in names:
        batch_file.write(f"qsub {path.join(job_name, name)}\n")
