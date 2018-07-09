"""
===========================
Runs the example of spreading activation on a toy language model.
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

from numpy import array

from model.graph import Graph
from model.temporal_spreading_activation import TemporalSpreadingActivation, \
    decay_function_exponential_with_decay_factor

logger = logging.getLogger()
logger_format = '%(asctime)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main():
    logger.info("Building graph...")

    tsa = TemporalSpreadingActivation(
        graph=Graph.from_distance_matrix(
            distance_matrix=array([
                [.0, .3, .6],  # Lion
                [.3, .0, .4],  # Tiger
                [.6, .4, .0],  # Stripes
            ]),
            weighted_graph=True,
            length_granularity=10
        ),
        node_relabelling_dictionary={0: "lion", 1: "tiger", 2: "stripes"},
        activation_threshold=0.3,
        impulse_pruning_threshold=0.1,
        node_decay_function=decay_function_exponential_with_decay_factor(
            decay_factor=0.9),
        edge_decay_function=decay_function_exponential_with_decay_factor(
            decay_factor=0.9),
    )

    logger.info("Activating node...")
    tsa.activate_node(tsa.label2node["lion"], 1)
    tsa.log_graph()

    logger.info("Running spreading activation...")
    for tick in range(1,100):
        tsa.tick()
        tsa.log_graph()


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
