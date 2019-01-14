"""
Builds some template jobs
"""

ram_amount = {
    1_000:  2,
    3_000:  3,
    10_000: 10,
    20_000: 10,
    30_000: 15,
    40_000: 20,
}
graph_sizes = sorted(ram_amount.keys())

names = []
for size in graph_sizes:
    k = f"{int(size/1000)}k"
    name = f"job_2_4_sa_{k}_ngram.sh"
    names.append(name)
    with open(name, mode="w", encoding="utf-8") as job_file:
        job_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
        job_file.write(f"#$ -S /bin/bash\n")
        job_file.write(f"#$ -q serial\n")
        job_file.write(f"#$ -N j24_{k}_ngram_sa\n")
        job_file.write(f"#$ -m e\n")
        job_file.write(f"#$ -M c.wingfield@lancaster.ac.uk\n")
        job_file.write(f"#$ -l h_vmem={ram_amount[size]}G\n")
        job_file.write(f"\n")
        job_file.write(f"source /etc/profile\n")
        job_file.write(f"\n")
        job_file.write(f"echo Job running on compute node `uname -n`\n")
        job_file.write(f"\n")
        job_file.write(f"module add anaconda3\n")
        job_file.write(f"\n")
        job_file.write(f"python3 ../2_4_category_production_ngram_tsa.py {int(size)}\n")
with open("submit_jobs_2_4_sa_ALL.sh", mode="w", encoding="utf-8") as batch_file:
    batch_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
    batch_file.write(f"#!/usr/bin/env bash\n")
    for name in names:
        batch_file.write(f"qsub {name}\n")
