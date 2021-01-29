#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Preprune some graphs.
Can take upwards of an hour to run, depending on graph size.
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
import sys
from os import path, replace

from framework.cognitive_model.ldm.utils.maths import DistanceType
from framework.cognitive_model.utils.logging import logger

from framework.cognitive_model.graph import length_from_distance, iter_edges_from_edgelist, edgelist_line
from framework.cognitive_model.preferences.preferences import Preferences


def main(length_factor: int, distance_type_name: str, pruning_distance: float):

    distance_type = DistanceType.from_name(distance_type_name)

    pruning_length = length_from_distance(pruning_distance, length_factor)

    edgelist_filename = f"sensorimotor {distance_type.name} distance length {length_factor}.edgelist"
    pruned_edgelist_filename = f"sensorimotor for testing only {distance_type.name} distance length {length_factor} pruned {pruning_distance}.edgelist"
    pruned_edgelist_filename_incomplete = pruned_edgelist_filename + ".incomplete"
    edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)
    pruned_edgelist_path = path.join(Preferences.graphs_dir, pruned_edgelist_filename)
    pruned_edgelist_path_incomplete = path.join(Preferences.graphs_dir, pruned_edgelist_filename_incomplete)

    logger.info(f"Saving pruned graph ({pruned_edgelist_filename_incomplete})")
    with open(pruned_edgelist_path_incomplete, mode="w", encoding="utf-8") as pruned_file:
        written_counter = 0
        for read_counter, (edge, length) in enumerate(iter_edges_from_edgelist(edgelist_path)):
            if read_counter % 1_000_000 == 0:
                logger.info(f"{read_counter:,} edges read, {written_counter:,} edges written")
            if length > pruning_length:
                continue
            pruned_file.write(edgelist_line(from_edge=edge, with_length=length))
            written_counter += 1
    logger.info(f"Renaming to {pruned_edgelist_filename}")
    replace(pruned_edgelist_path_incomplete, pruned_edgelist_path)


if __name__ == '__main__':

    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Save pruned graphs.")
    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-p", "--pruning_distance", required=True, type=float)
    args = parser.parse_args()

    main(length_factor=args.length_factor,
         distance_type_name=args.distance_type,
         pruning_distance=args.pruning_distance)

    logger.info("Done!")
