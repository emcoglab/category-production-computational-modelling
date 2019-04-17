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
import logging
import sys
from os import path

from ldm.utils.maths import DistanceType
from model.graph import Graph, log_graph_topology
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


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
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")

    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-c", "--corpus_name", required=True, type=str)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-m", "--model_name", required=True, type=str)
    parser.add_argument("-r", "--radius", required=True, type=int)
    parser.add_argument("-w", "--words", type=int, required=True, help="The number of words to use from the corpus. (Top n words.)")
    parser.add_argument("-p", "--prune_percent", required=False, type=int, help="The percentage of longest edges to prune from the graph.", default=None)

    args = parser.parse_args()

    main(length_factor=args.length_factor,
         corpus_name=args.corpus_name,
         model_name=args.model_name,
         radius=args.radius,
         distance_type_name=args.distance_type,
         n_words=args.words,
         prune_top_percentile=args.prune_percent)
    args = parser.parse_args()

    logger.info("Done!")
