"""
===========================
Tests for TemporalSpatialPropagation.
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

import unittest

from numpy import array, Infinity

from model.graph import Graph
from model.temporal_spatial_propagation import TemporalSpatialPropagation
from model.utils.maths import make_decay_function_exponential_with_decay_factor


class TestToyExample(unittest.TestCase):

    def test_worked_example_unweighted_node_values(self):
        distance_matrix = array([
            [.0, .3, .6],  # Lion
            [.3, .0, .4],  # Tiger
            [.6, .4, .0],  # Stripes
        ])
        graph = Graph.from_distance_matrix(
            distance_matrix=distance_matrix,
            length_granularity=10,
        )
        tsp = TemporalSpatialPropagation(
            underlying_graph=graph,
            idx2label={0: "lion", 1: "tiger", 2: "stripes"},
            impulse_pruning_threshold=.1,
            node_decay_function=make_decay_function_exponential_with_decay_factor(decay_factor=0.9),
            activation_cap=Infinity,
        )

        tsp.activate_item_with_label("lion", 1)

        for i in range(1, 16):
            tsp.tick()

        # WARNING!!!
        # These numbers not manually verified, just copied from the output for the purposes of refactoring!!!!!
        self.assertAlmostEqual(tsp.activation_of_item_with_label("lion"),    0.2059, places=4)
        self.assertAlmostEqual(tsp.activation_of_item_with_label("tiger"),   0.2824, places=4)
        self.assertAlmostEqual(tsp.activation_of_item_with_label("stripes"), 0.3874, places=4)


if __name__ == '__main__':
    unittest.main()
