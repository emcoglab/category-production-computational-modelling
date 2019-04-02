"""
Builds some template jobs
"""
from os import path, mkdir

job_name = 'job_2_3'
short_name = "j23"
script_name = "2_3_category_production_ngram_tsa"

if not path.isdir(job_name):
    mkdir(job_name)

ram_amount = {
    # 1_000:  2,
    # 3_000:  3,
    # 10_000: 7,
    30_000: 12,
    40_000: 15,
}
graph_sizes = sorted(ram_amount.keys())
names = []

for ft, sd in [
    (0.3, 10), (0.4, 10),
    (0.2, 15), (0.4, 15), (0.5, 15),
    (0.2, 20), (0.4, 20),
    (0.3, 25)
]:
    for model in ["pmi_ngram", "ppmi_ngram"]:
        for size in graph_sizes:
            k = f"{int(size / 1000)}k"
            name = f"{job_name}_{k}_ft{ft}_sd{sd}_{model}.sh"
            names.append(name)
            with open(path.join(job_name, name), mode="w", encoding="utf-8") as job_file:
                job_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
                job_file.write(f"#$ -S /bin/bash\n")
                job_file.write(f"#$ -q serial\n")
                job_file.write(f"#$ -N {short_name}_{k}_f{ft}_s{sd}_{model}_sa\n")
                job_file.write(f"#$ -m e\n")
                job_file.write(f"#$ -M c.wingfield@lancaster.ac.uk\n")
                job_file.write(f"#$ -l h_vmem={ram_amount[size]}G\n")
                job_file.write(f"\n")
                job_file.write(f"source /etc/profile\n")
                job_file.write(f"\n")
                job_file.write(f"echo Job running on compute node `uname -n`\n")
                job_file.write(f"\n")
                job_file.write(f"module add anaconda3/2018.12\n")
                job_file.write(f"\n")
                job_file.write(f"python3 ../{script_name}.py \\\n")
                job_file.write(f"           --bailout {int(size / 2)} \\\n")
                job_file.write(f"           --corpus_name bbc \\\n")
                job_file.write(f"           --firing_threshold {ft} \\\n")
                job_file.write(f"           --impulse_pruning_threshold 0.05 \\\n")
                job_file.write(f"           --length_factor 10 \\\n")
                job_file.write(f"           --model_name pmi_ngram \\\n")
                job_file.write(f"           --node_decay_factor 0.99 \\\n")
                job_file.write(f"           --radius 5 \\\n")
                job_file.write(f"           --edge_decay_sd_factor {sd} \\\n")
                job_file.write(f"           --run_for_ticks 3000 \\\n")
                job_file.write(f"           --words {int(size)} \n")
with open(f"{job_name}_submit_ALL.sh", mode="w", encoding="utf-8") as batch_file:
    batch_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
    batch_file.write(f"#!/usr/bin/env bash\n")
    for name in names:
        batch_file.write(f"qsub {path.join(job_name, name)}\n")
