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

from matplotlib.backends.backend_pdf import PdfPages
from networkx import convert_matrix, relabel_nodes
from numpy import array

from temporal_spreading_activation import TemporalSpreadingActivation

logger = logging.getLogger()
# logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_format = '%(asctime)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main():
    logger.info("Building graph...")

    distance_matrix = array([
        [.0, .3, .6],  # Lion
        [.3, .0, .4],  # Tiger
        [.6, .4, .0],  # Stripes
    ])
    graph = convert_matrix.from_numpy_array(distance_matrix)
    graph = relabel_nodes(graph, {0: "lion", 1: "tiger", 2: "stripes"}, copy=False)

    sa = TemporalSpreadingActivation(
        graph=graph,
        threshold=.2,
        weight_coefficient=1,
        granularity=100,
        node_decay_function=TemporalSpreadingActivation.create_decay_function_exponential_with_params(
            decay_factor=0.99),
        edge_decay_function=TemporalSpreadingActivation.create_decay_function_gaussian_with_params(
            sd=40),
    )

    with PdfPages("/Users/caiwingfield/Desktop/graph.pdf") as pdf:

        logger.info("Activating node...")
        sa.activate_node("lion", 1)
        # sa.log_graph()
        pos = sa.draw_graph(pdf=pdf, frame_label=str(0))

        logger.info("Running spreading activation...")
        for i in range(1, 200):
            logger.info(f"CLOCK = {i}")
            sa.tick()
            # sa.log_graph()
            sa.draw_graph(pdf=pdf, pos=pos, frame_label=str(i))


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
