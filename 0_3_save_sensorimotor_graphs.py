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
import logging
import sys
from os import path

from sklearn.metrics.pairwise import pairwise_distances

from ldm.utils.logging import log_message, date_format
from ldm.utils.maths import DistanceType
from model.graph import save_edgelist_from_distance_matrix
from preferences import Preferences
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

logger = logging.getLogger(__name__)


def main(length_factor: int, distance_type_name: str):

    distance_type = DistanceType.from_name(distance_type_name)

    sm_norms = SensorimotorNorms()

    # i here ranges over range(0, len(sm_norms.iter_words))
    node_label_dict = {i: w for i, w in enumerate(sm_norms.iter_words())}

    edgelist_filename = f"sensorimotor {distance_type.name} distance length {length_factor}.edgelist"
    edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)

    if path.isfile(edgelist_path):
        logger.info(f"{edgelist_filename} already computed.")
        logger.info("Skipping")

    else:

        logger.info("Computing between-word distances")
        data_matrix = sm_norms.matrix_for_words(list(
            # Get words in idx order
            w for _, w in sorted(node_label_dict.items(), key=lambda iw: iw[0])))
        distance_matrix = pairwise_distances(data_matrix, metric=distance_type.name, n_jobs=-1)

        save_edgelist_from_distance_matrix(
            file_path=edgelist_path,
            distance_matrix=distance_matrix,
            length_factor=length_factor)

    # Save node label dictionary
    node_label_filename = f"sensorimotor words.nodelabels"
    node_label_filename = path.join(Preferences.graphs_dir, node_label_filename)
    if path.isfile(node_label_filename):
        logger.info(f"{path.basename(node_label_filename)} exists, skipping")
    else:
        logger.info(f"Saving node label dictionary to {path.basename(node_label_filename)}")
        with open(node_label_filename, mode="w", encoding="utf-8") as node_label_file:
            json.dump(node_label_dict, node_label_file)


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Save graphs built from sensorimotor norms.")
    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    args = parser.parse_args()

    main(length_factor=args.length_factor,
         distance_type_name=args.distance_type)

    logger.info("Done!")
