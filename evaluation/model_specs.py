"""
===========================
Deal with model specifications
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""

from os import path

import yaml


def save_model_spec_linguistic(edge_decay_sd_factor, firing_threshold, length_factor, model_name, n_words,
                               response_dir):
    spec = {
        "Model name":       model_name,
        "Length factor":    length_factor,
        "SD factor":        edge_decay_sd_factor,
        "Firing threshold": firing_threshold,
        "Words":            n_words,
    }
    with open(path.join(response_dir, " model_spec.yaml"), mode="w", encoding="utf-8") as spec_file:
        yaml.dump(spec, spec_file, yaml.SafeDumper)


def save_model_spec_sensorimotor(length_factor, max_sphere_radius, run_for_ticks, bailout,
                                 response_dir):
    spec = {
        "Length factor":     length_factor,
        "Max sphere radius": max_sphere_radius,
        "Run for ticks":     run_for_ticks,
        "Bailout":           bailout,
    }
    with open(path.join(response_dir, " model_spec.yaml"), mode="w", encoding="utf-8") as spec_file:
        yaml.dump(spec, spec_file, yaml.SafeDumper)


def load_model_spec(response_dir) -> dict:
    with open(path.join(response_dir, " model_spec.yaml"), mode="r", encoding="utf-8") as spec_file:
        return yaml.load(spec_file, yaml.SafeLoader)
