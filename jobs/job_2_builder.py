"""
Builds some template jobs
"""
graph_sizes = [
    1_000,
    3_000,
    5_000,
    10_000,
    15_000,
    20_000,
    25_000,
]
pruning_percents = [
    0,
    10,
    20,
    30,
    40,
    50,
]
ram_amount = {
    1_000:  {0: 2,   10: 2,   20: 2,   30: 2,   40: 2,   50: 2,   },
    3_000:  {0: 3,   10: 3,   20: 3,   30: 3,   40: 2,   50: 2,   },
    5_000:  {0: 5,   10: 5,   20: 5,   30: 5,   40: 4,   50: 3,   },
    10_000: {0: 20,  10: 20,  20: 17,  30: 14,  40: 13,  50: 12,  },
    15_000: {0: 38,  10: 34,  20: 32,  30: 26,  40: 24,  50: 22,  },
    20_000: {0: 50,  10: 50,  20: 50,  30: 50,  40: 50,  50: 50,  },
    25_000: {0: 120, 10: 120, 20: 120, 30: 120, 40: 120, 50: 120, },
}
names = []
for size in graph_sizes:
    for percent in pruning_percents:
        k = f"{int(size/1000)}k"
        name = f"job_2_1_sa_{k}_{percent}pc.sh"
        names.append(name)
        with open(name, mode="w", encoding="utf-8") as job_file:
            job_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
            job_file.write(f"#$ -S /bin/bash\n")
            job_file.write(f"#$ -q serial\n")
            job_file.write(f"#$ -N j2_{k}_{percent}pc_sa\n")
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
            job_file.write(f"python3 ../2_1_category_production_pruned_tsa.py {int(size)} {percent}\n")
with open("submit_jobs_2_1_sa_ALL.sh", mode="w", encoding="utf-8") as batch_file:
    batch_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
    batch_file.write(f"#!/usr/bin/env bash\n")
    for name in names:
        batch_file.write(f"qsub {name}\n")
