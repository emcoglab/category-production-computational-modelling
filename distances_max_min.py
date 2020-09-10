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

from ldm.utils.maths import DistanceType
from model.graph import iter_edges_from_edgelist
from model.utils.logging import logger
from preferences import Preferences


def main():
    distance_type = DistanceType.Minkowski3
    length_factor = 100
    max_sphere_radius = 150
    # edgelist_filename = f"sensorimotor for testing only {distance_type.name} distance length {length_factor} pruned {max_sphere_radius}.edgelist"
    edgelist_filename = f"sensorimotor {distance_type.name} distance length {length_factor}.edgelist"
    min_length, max_length = numpy.Inf, -numpy.Inf
    min_edges = []
    max_edges = []
    for i, (edge, length) in enumerate(iter_edges_from_edgelist(Path(Preferences.graphs_dir, edgelist_filename))):
        if i % 1_000_000 == 0:
            logger.info(f"\t{i:,}")
        if length <= min_length:
            min_length = length
            min_edges.append(edge)
        elif length >= max_length:
            max_length = length
            max_edges.append(edge)
    logger.info(f"Minimum: {min_edges}: {min_length}")
    logger.info(f"Maximum: {max_edges}: {max_length}")


if __name__ == '__main__':
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")