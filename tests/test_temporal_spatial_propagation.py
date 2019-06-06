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

from numpy import array

from model.events import ItemFiredEvent
from model.graph import Graph
from model.temporal_spatial_propagation import TemporalSpatialPropagation
from model.utils.maths import make_decay_function_exponential_with_decay_factor


class TestTemporalSpatialPropagationToyExample(unittest.TestCase):

    # TODO: this test is failing
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
            node_decay_function=make_decay_function_exponential_with_decay_factor(decay_factor=0.9),
        )

        # t = 0
        e = tsp.activate_item_with_label("lion", 1)
        self.assertIsNotNone(e)
        self.assertEqual(e, ItemFiredEvent(time=0, item=0, activation=1.0))

        for t in range(1, 3):
            es = tsp.tick()
            self.assertEqual(len(es), 0)

        # t = 3
        es = tsp.tick()
        self.assertEqual(len(es), 1)
        self.assertTrue(ItemFiredEvent(time=3, item=1, activation=1.0) in es)

        for t in range(4, 6):
            es = tsp.tick()
            self.assertEqual(len(es), 0)

        # t = 6
        es = tsp.tick()
        self.assertEqual(len(es), 2)
        self.assertTrue(ItemFiredEvent(time=6, item=2, activation=1.0) in es)
        self.assertTrue(ItemFiredEvent(time=6, item=0, activation=1.5314409136772156) in es)

        # t = 7
        es = tsp.tick()
        self.assertEqual(len(es), 1)
        self.assertTrue(ItemFiredEvent(time=7, item=2, activation=1.899999976158142) in es)

        for t in range(8, 9):
            es = tsp.tick()
            self.assertEqual(len(es), 0)

        for t in range(9, 12):
            es = tsp.tick()
            self.assertEqual(len(es), 1)

        # t = 12
        es = tsp.tick()
        self.assertEqual(len(es), 2)
        self.assertTrue(ItemFiredEvent(time=12, item=0, activation=3.876752197742462) in es)
        self.assertTrue(ItemFiredEvent(time=12, item=2, activation=2.653371751308441) in es)


if __name__ == '__main__':
    unittest.main()
