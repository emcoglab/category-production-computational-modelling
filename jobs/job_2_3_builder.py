"""
Builds some template jobs
"""
from dataclasses import dataclass
from os import path, mkdir


@dataclass
class Spec:
    graph_size: int
    firing_threshold: float
    model_name: str
    edge_decay_sd: float


job_name = 'job_2_3'
short_name = "j23"
script_name = "2_3_category_production_ngram_tsa"

if not path.isdir(job_name):
    mkdir(job_name)


ram = {
    1_000: 2,
    3_000: 3,
    10_000: 7,
    30_000: 11,
    40_000: 15,
}

specs = [
    Spec(graph_size=40_000, model_name="pmi_ngram", firing_threshold=0.7, edge_decay_sd=35),
    Spec(graph_size=40_000, model_name="pmi_ngram", firing_threshold=0.6, edge_decay_sd=25),
    Spec(graph_size=40_000, model_name="pmi_ngram", firing_threshold=0.7, edge_decay_sd=30),
    Spec(graph_size=40_000, model_name="pmi_ngram", firing_threshold=0.6, edge_decay_sd=30),
    Spec(graph_size=40_000, model_name="ppmi_ngram", firing_threshold=0.7, edge_decay_sd=35),
    Spec(graph_size=40_000, model_name="ppmi_ngram", firing_threshold=0.5, edge_decay_sd=15),
    Spec(graph_size=40_000, model_name="ppmi_ngram", firing_threshold=0.7, edge_decay_sd=20),
    Spec(graph_size=40_000, model_name="ppmi_ngram", firing_threshold=0.5, edge_decay_sd=20),
    Spec(graph_size=40_000, model_name="ppmi_ngram", firing_threshold=0.5, edge_decay_sd=30),
]

names = []
for spec in specs:
    k = f"{int(spec.graph_size / 1000)}k"
    name = f"{job_name}_{k}_ft{spec.firing_threshold}_sd{spec.edge_decay_sd}_{spec.model_name}.sh"
    names.append(name)
    with open(path.join(job_name, name), mode="w", encoding="utf-8") as job_file:
        job_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
        job_file.write(f"#$ -S /bin/bash\n")
        job_file.write(f"#$ -q serial\n")
        job_file.write(f"#$ -N {short_name}_{k}_f{spec.firing_threshold}_s{spec.edge_decay_sd}_{spec.model_name}_sa\n")
        job_file.write(f"#$ -m e\n")
        job_file.write(f"#$ -M c.wingfield@lancaster.ac.uk\n")
        job_file.write(f"#$ -l h_vmem={ram[spec.graph_size]}G\n")
        job_file.write(f"\n")
        job_file.write(f"source /etc/profile\n")
        job_file.write(f"\n")
        job_file.write(f"echo Job running on compute node `uname -n`\n")
        job_file.write(f"\n")
        job_file.write(f"module add anaconda3/2018.12\n")
        job_file.write(f"\n")
        job_file.write(f"python3 ../{script_name}.py \\\n")
        job_file.write(f"           --bailout {int(spec.graph_size / 2)} \\\n")
        job_file.write(f"           --corpus_name bbc \\\n")
        job_file.write(f"           --firing_threshold {spec.firing_threshold} \\\n")
        job_file.write(f"           --impulse_pruning_threshold 0.05 \\\n")
        job_file.write(f"           --length_factor 10 \\\n")
        job_file.write(f"           --model_name {spec.model_name} \\\n")
        job_file.write(f"           --node_decay_factor 0.99 \\\n")
        job_file.write(f"           --radius 5 \\\n")
        job_file.write(f"           --edge_decay_sd_factor {spec.edge_decay_sd} \\\n")
        job_file.write(f"           --run_for_ticks 3000 \\\n")
        job_file.write(f"           --words {int(spec.graph_size)} \n")
with open(f"{job_name}_submit_ALL.sh", mode="w", encoding="utf-8") as batch_file:
    batch_file.write(f"# GENERATED CODE, CHANGES WILL BE OVERWRITTEN\n")
    batch_file.write(f"#!/usr/bin/env bash\n")
    for name in names:
        batch_file.write(f"qsub {path.join(job_name, name)}\n")
