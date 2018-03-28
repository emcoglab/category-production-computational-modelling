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
from numpy import array

from temporal_spreading_activation import TemporalSpreadingActivation
from tsa_visualisation import run_with_pdf_output

logger = logging.getLogger()
logger_format = '%(asctime)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main():
    logger.info("Building graph...")

    sa = TemporalSpreadingActivation(
        graph=TemporalSpreadingActivation.graph_from_distance_matrix(
            distance_matrix=array([
                [.0, .3, .6],  # Lion
                [.3, .0, .4],  # Tiger
                [.6, .4, .0],  # Stripes
            ]),
            length_granularity=10,
            relabelling_dict={0: "lion", 1: "tiger", 2: "stripes"}
        ),
        threshold=.2,
        node_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_decay_factor(
            decay_factor=0.9),
        edge_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_decay_factor(
            decay_factor=.9),
    )

    logger.info("Activating node...")
    sa.activate_node("lion", 1)
    sa.log_graph()

    logger.info("Running spreading activation...")
    for t in range(1,15):
        logger.info("")
        sa.tick()
        sa.log_graph()
    # run_with_pdf_output(sa, 200, "/Users/cai/Desktop/graph.pdf")


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
