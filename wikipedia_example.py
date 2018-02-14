"""
===========================
Runs the example of spreading activation from https://en.wikipedia.org/wiki/Spreading_activation#/media/File:Spreading-activation-graph-1.png.
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

import logging
import sys

from networkx import DiGraph

from spreading_activation import SpreadingActivationCleglowski

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main():
    graph = DiGraph()
    graph.add_edge(1, 2, weight=0.9)
    graph.add_edge(2, 3, weight=0.9)
    graph.add_edge(3, 4, weight=0.9)
    graph.add_edge(3, 11, weight=0.9)
    graph.add_edge(4, 5, weight=0.9)
    graph.add_edge(5, 6, weight=0.9)
    graph.add_edge(11, 12, weight=0.9)
    graph.add_edge(12, 13, weight=0.9)

    sa = SpreadingActivationCleglowski(graph, decay_factor=0.85, firing_threshold=0, energy_cap=1)

    sa.activate_node(1)

    sa.spread_n_times(5)
    sa.print_graph()


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
