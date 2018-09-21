"""
Builds some template jobs
"""

importance_thresholds = [
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
    1_000:  {100: 2,   90: 2,   80: 2,   70: 2,   60: 2,  50: 2 , 40: 2 , 30: 2 , 20: 2 , 10: 2 , 0: 2  },
    3_000:  {100: 3,   90: 3,   80: 3,   70: 3,   60: 3,  50: 2 , 40: 2 , 30: 2 , 20: 2 , 10: 2 , 0: 2  },
    10_000: {100: 30,  90: 26,  80: 24,  70: 22,  60: 20, 50: 15, 40: 13, 30: 12, 20: 12, 10: 12, 0: 12 },
    20_000: {100: 70,  90: 65,  80: 60,  70: 55,  60: 50, 50: 45, 40: 35, 30: 30, 20: 30, 10: 25, 0: 25 },
    30_000: {100: 130, 90: 120, 80: 110, 70: 100, 60: 90, 50: 85, 40: 80, 30: 70, 20: 70, 10: 60, 0: 60 },
}
graph_sizes = sorted(ram_amount.keys())

names = []
for size in graph_sizes:
    for importance in importance_thresholds:
        k = f"{int(size/1000)}k"
        name = f"job_2_2_sa_{k}_{importance}im.sh"
        names.append(name)
        with open(name, mode="w", encoding="utf-8") as job_file:
            job_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
            job_file.write(f"#$ -S /bin/bash\n")
            job_file.write(f"#$ -q serial\n")
            job_file.write(f"#$ -N j22_{k}_{importance}im_sa\n")
            job_file.write(f"#$ -m e\n")
            job_file.write(f"#$ -M c.wingfield@lancaster.ac.uk\n")
            job_file.write(f"#$ -l h_vmem={ram_amount[size][importance]}G\n")
            job_file.write(f"\n")
            job_file.write(f"source /etc/profile\n")
            job_file.write(f"\n")
            job_file.write(f"echo Job running on compute node `uname -n`\n")
            job_file.write(f"\n")
            job_file.write(f"module add anaconda3\n")
            job_file.write(f"\n")
            job_file.write(f"python3 ../2_2_category_production_importance_pruned_tsa.py {int(size)} {importance}\n")
with open("submit_jobs_2_2_sa_ALL.sh", mode="w", encoding="utf-8") as batch_file:
    batch_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
    batch_file.write(f"#!/usr/bin/env bash\n")
    for name in names:
        batch_file.write(f"qsub {name}\n")
