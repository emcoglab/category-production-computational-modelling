#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Save graphs built from sensorimotor norms.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""

import argparse
import json
import sys
from os import path

from sklearn.metrics.pairwise import pairwise_distances

from cognitive_model.ldm.utils.maths import DistanceType
from cognitive_model.utils.logging import logger
from cognitive_model.sensorimotor_norms.sensorimotor_norms import SensorimotorNorms
from cognitive_model.graph import save_edgelist_from_distance_matrix
from cognitive_model.preferences import Preferences


def main(length_factor: int, distance_type_name: str, use_breng_translation: bool):

    distance_type = DistanceType.from_name(distance_type_name)

    if use_breng_translation:
        from cognitive_model.sensorimotor_norms.breng_translation.dictionary.version import VERSION as SM_BRENG_VERSION
        node_label_filename = f"sensorimotor words BrEng v{SM_BRENG_VERSION}.nodelabels"
    else:
        node_label_filename = "sensorimotor words.nodelabels"
    # Graph itself doesn't change with BrEng translation
    edgelist_filename = (f"sensorimotor"
                         f" {distance_type.name} distance"
                         f" length {length_factor}"
                         f".edgelist")
    edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)

    sm_norms = SensorimotorNorms(use_breng_translation=use_breng_translation)

    # i here ranges over range(0, len(sm_norms.iter_words))
    node_label_dict = {i: w for i, w in enumerate(sm_norms.iter_words())}

    if path.isfile(edgelist_path):
        logger.info(f"{edgelist_filename} already computed.")
        logger.info("Skipping")

    else:
        logger.info("Computing between-word distances")
        data_matrix = sm_norms.matrix_for_words(list(
            # Get words in idx order
            w for _, w in sorted(node_label_dict.items(), key=lambda iw: iw[0])))
        if distance_type is DistanceType.Minkowski3:
            # Need to pass p=3 param specifically when using Minkowski-3 distance
            distance_matrix = pairwise_distances(data_matrix, metric="minkowski", n_jobs=-1, p=3)
        else:
            distance_matrix = pairwise_distances(data_matrix, metric=distance_type.name, n_jobs=-1)

        save_edgelist_from_distance_matrix(
            file_path=edgelist_path,
            distance_matrix=distance_matrix,
            length_factor=length_factor)

    # Save node label dictionary
    node_label_filename = path.join(Preferences.graphs_dir, node_label_filename)
    if path.isfile(node_label_filename):
        logger.info(f"{path.basename(node_label_filename)} exists, skipping")
    else:
        logger.info(f"Saving node label dictionary to {path.basename(node_label_filename)}")
        with open(node_label_filename, mode="w", encoding="utf-8") as node_label_file:
            json.dump(node_label_dict, node_label_file)


if __name__ == '__main__':

    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Save graphs built from sensorimotor norms.")
    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("--use-breng-translation", action="store_true")
    args = parser.parse_args()

    main(length_factor=args.length_factor,
         distance_type_name=args.distance_type,
         use_breng_translation=args.use_breng_translation)

    logger.info("Done!")
