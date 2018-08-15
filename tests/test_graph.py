"""
===========================
Tests for graph functions.
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

import unittest

from numpy import array

from model.graph import Graph


class TestGraph(unittest.TestCase):

    def test_connected(self):
        connected_graph = Graph.from_adjacency_matrix(
            adjacency_matrix=array([
                [.0, .3, .6],  # Lion
                [.3, .0, .4],  # Tiger
                [.6, .4, .0],  # Stripes
            ])
        )
        self.assertTrue(connected_graph.is_connected())

    def test_disconnected(self):
        disconnected_graph = Graph.from_adjacency_matrix(
            adjacency_matrix=array([
                [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 1, 1, 1]
            ])
        )
        self.assertFalse(disconnected_graph.is_connected())

    def test_orphan_detection(self):
        disconnected_graph = Graph.from_adjacency_matrix(
            adjacency_matrix=array([
                [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 1, 1, 1]
            ])
        )
        orphan_graph = Graph.from_adjacency_matrix(
            adjacency_matrix=array([
                [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ])
        )
        self.assertTrue(orphan_graph.has_orphaned_nodes())
        self.assertFalse(disconnected_graph.has_orphaned_nodes())


if __name__ == '__main__':
    unittest.main()
