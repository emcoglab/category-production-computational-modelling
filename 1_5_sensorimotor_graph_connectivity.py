#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Investigate connectivity of sensorimotor graphs.
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
import argparse
import sys
from os import path

from framework.cognitive_model.ldm.utils.maths import DistanceType
from framework.cognitive_model.graph import Graph, log_graph_topology
from framework.cognitive_model.utils.logging import logger
from framework.cognitive_model.preferences.preferences import Preferences


def main(pruning_length: int,
         length_factor: int,
         distance_type_name: str):

    distance_type = DistanceType.from_name(distance_type_name)

    edgelist_filename = f"sensorimotor {distance_type.name} distance length {length_factor}.edgelist"
    edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)

    # Load the full graph
    graph = Graph.load_from_edgelist(edgelist_path,
                                     ignore_edges_longer_than=pruning_length)

    log_graph_topology(graph)


if __name__ == '__main__':
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")

    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-p", "--prune_length", required=True, type=int, help="The length of the longest edges to prune from the graph.")

    args = parser.parse_args()

    main(length_factor=args.length_factor,
         distance_type_name=args.distance_type,
         pruning_length=args.prune_length)
    args = parser.parse_args()

    logger.info("Done!")
