"""
===========================
Preprune some graphs
FOR TESTING PURPOSES ONLY.
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
import logging
import sys
from os import path

from ldm.utils.logging import log_message, date_format
from ldm.utils.maths import DistanceType
from model.graph import Graph
from preferences import Preferences

logger = logging.getLogger(__name__)


def main(length_factor: int, distance_type_name: str, pruning_length: int):

    distance_type = DistanceType.from_name(distance_type_name)

    edgelist_filename = f"sensorimotor {distance_type.name} distance length {length_factor}.edgelist"
    pruned_edgelist_filename = f"sensorimotor for testing only {distance_type.name} distance length {length_factor} pruned {pruning_length}.edgelist"
    edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)
    pruned_edgelist_path = path.join(Preferences.graphs_dir, pruned_edgelist_filename)

    # Load graph
    logger.info(f"Loading sensorimotor graph with pruning ({edgelist_filename})")
    sensorimotor_graph = Graph.load_from_edgelist(file_path=edgelist_path,
                                                  ignore_edges_longer_than=pruning_length,
                                                  with_feedback=True)

    logger.info(f"Saving pruned graph ({pruned_edgelist_filename})")
    sensorimotor_graph.save_as_edgelist(pruned_edgelist_path)


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Save pruned graphs.")
    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-p", "--pruning_length", required=True, type=int)
    args = parser.parse_args()

    main(length_factor=args.length_factor,
         distance_type_name=args.distance_type,
         pruning_length=args.pruning_length)

    logger.info("Done!")
