"""
===========================
Investigate how many items fall within sensorimotor spheres of different radii.
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

from matplotlib import pyplot
from seaborn import distplot

from ldm.utils.maths import DistanceType
from model.graph import Graph, log_graph_topology
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main(length_factor: int,
         distance_type_name: str,
         radius: int):

    distance_type = DistanceType.from_name(distance_type_name)

    edgelist_filename = f"sensorimotor {distance_type.name} distance length {length_factor}.edgelist"
    edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)

    logger.info(f"Computing neighbourhood distribution for {radius}...")

    graph = Graph.load_from_edgelist(edgelist_path, ignore_edges_longer_than=radius)

    log_graph_topology(graph)

    neighbour_counts = [
        len(list(graph.neighbourhood(n)))
        for n in graph.nodes
    ]

    f = pyplot.figure()
    distplot(neighbour_counts)
    f.savefig(path.join(Preferences.figures_dir,
                        "neighbourhood distributions",
                        f"points_within_radius_{radius}.png"))
    pyplot.close(f)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")

    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-p", "--pruning_length", required=True, type=int)

    args = parser.parse_args()

    main(length_factor=args.length_factor,
         distance_type_name=args.distance_type,
         radius=args.pruning_length)
    args = parser.parse_args()

    logger.info("Done!")
