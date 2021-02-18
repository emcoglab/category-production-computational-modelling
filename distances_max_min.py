"""
===========================
Maximum and minimum distances in different graphs.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2020
---------------------------
"""

import sys
from pathlib import Path

import numpy

from framework.cognitive_model.ldm.utils.maths import DistanceType
from framework.cognitive_model.graph import iter_edges_from_edgelist
from framework.cognitive_model.utils.logging import logger
from framework.cognitive_model.preferences.preferences import Preferences


def main(sensorimotor: bool):
    if sensorimotor:
        logger.info("Sensorimotor")
        distance_type = DistanceType.Minkowski3
        length_factor = 1589
        max_sphere_radius = 1.5
        edgelist_filename = f"sensorimotor for testing only {distance_type.name} distance length {length_factor} pruned {max_sphere_radius}.edgelist"
    else:
        logger.info("Linguistic")
        length_factor = 100
        words = 40_000
        edgelist_filename = f"PPMI n-gram (BBC), r=5 {words} words length {length_factor}.edgelist"

    min_length, max_length = numpy.Inf, -numpy.Inf
    min_edges = []
    max_edges = []
    for i, (edge, length) in enumerate(iter_edges_from_edgelist(Path(Preferences.graphs_dir, edgelist_filename))):
        if i % 1_000_000 == 0:
            logger.info(f"{i:,} edges checked")
        if length <= min_length:
            min_length = length
            min_edges.append(edge)
            continue
        if length >= max_length:
            max_length = length
            max_edges.append(edge)
            continue

    logger.info(f"Minimum: {min_edges}: {min_length}")
    logger.info(f"Maximum: {max_edges}: {max_length}")


if __name__ == '__main__':
    logger.info("Running %s" % " ".join(sys.argv))
    main(sensorimotor=True)
    main(sensorimotor=False)
    logger.info("Done!")
