"""
Builds some template jobs
"""

pruning_percents = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
]
ram_amount = {
    1_000:  {100: 2,   90: 2,   80: 2,   70: 2,  60: 2,  50: 2 , 40: 2 , 30: 2 , 20: 2 , 10: 2 , 0: 2  },
    3_000:  {100: 3,   90: 3,   80: 3,   70: 3,  60: 2,  50: 2 , 40: 2 , 30: 2 , 20: 2 , 10: 2 , 0: 2  },
    10_000: {100: 20,  90: 20,  80: 17,  70: 14, 60: 13, 50: 12, 40: 12, 30: 12, 20: 12, 10: 12, 0: 12 },
    20_000: {100: 65,  90: 60,  80: 55,  70: 50, 60: 40, 50: 35, 40: 35, 30: 30, 20: 30, 10: 25, 0: 25 },
    30_000: {100: 120, 90: 120, 80: 90,  70: 90, 60: 80, 50: 80, 40: 80, 30: 70, 20: 70, 10: 60, 0: 60 },
}
graph_sizes = sorted(ram_amount.keys())

names = []
for size in graph_sizes:
    for percent in pruning_percents:
        k = f"{int(size/1000)}k"
        name = f"job_2_2_sa_{k}_{percent}pc.sh"
        names.append(name)
        with open(name, mode="w", encoding="utf-8") as job_file:
            job_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
            job_file.write(f"#$ -S /bin/bash\n")
            job_file.write(f"#$ -q serial\n")
            job_file.write(f"#$ -N j22_{k}_{percent}pc_sa\n")
            job_file.write(f"#$ -m e\n")
            job_file.write(f"#$ -M c.wingfield@lancaster.ac.uk\n")
            job_file.write(f"#$ -l h_vmem={ram_amount[size][percent]}G\n")
            job_file.write(f"\n")
            job_file.write(f"source /etc/profile\n")
            job_file.write(f"\n")
            job_file.write(f"echo Job running on compute node `uname -n`\n")
            job_file.write(f"\n")
            job_file.write(f"module add anaconda3\n")
            job_file.write(f"\n")
            job_file.write(f"python3 ../2_2_category_production_importance_pruned_tsa.py {int(size)} {percent}\n")
with open("submit_jobs_2_2_sa_ALL.sh", mode="w", encoding="utf-8") as batch_file:
    batch_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
    batch_file.write(f"#!/usr/bin/env bash\n")
    for name in names:
        batch_file.write(f"qsub {name}\n")
