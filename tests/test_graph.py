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

from numpy import array, ones, eye

from model.graph import Graph, Edge, Node


class TestGraphPruning(unittest.TestCase):

    def test_remove_edge_doesnt_affect_nodes(self):
        graph = Graph.from_adjacency_matrix(
            adjacency_matrix=array([
                [0, 1, 1],  # Lion
                [1, 0, 1],  # Tiger
                [1, 1, 0],  # Stripes
            ])
        )
        self.assertTrue(Node(0) in graph.nodes)
        self.assertTrue(Node(1) in graph.nodes)
        self.assertTrue(Node(2) in graph.nodes)
        graph.remove_edge(Edge((0, 1)))
        self.assertTrue(Node(0) in graph.nodes)
        self.assertTrue(Node(1) in graph.nodes)
        self.assertTrue(Node(2) in graph.nodes)

    def test_remove_edge_remove_it(self):
        graph = Graph.from_adjacency_matrix(
            adjacency_matrix=array([
                [0, 1, 1],  # Lion
                [1, 0, 1],  # Tiger
                [1, 1, 0],  # Stripes
            ])
        )
        edge_to_remove = Edge((0, 1))
        self.assertTrue(edge_to_remove in graph.edges)
        graph.remove_edge(edge_to_remove)
        self.assertFalse(edge_to_remove in graph.edges)

    def test_edge_pruning(self):
        graph = Graph.from_distance_matrix(
            length_granularity=1,
            distance_matrix=array([
                [0, 3, 5],  # Lion
                [3, 0, 4],  # Tiger
                [5, 4, 0],  # Stripes
            ])
        )
        # Check all present
        self.assertTrue(Edge((0, 1)) in graph.edges)
        self.assertTrue(Edge((1, 2)) in graph.edges)
        self.assertTrue(Edge((0, 2)) in graph.edges)
        graph.prune_longest_edges_by_length(4)
        # Check still present
        self.assertTrue(Edge((0, 1)) in graph.edges)
        self.assertTrue(Edge((1, 2)) in graph.edges)
        # Check absent
        self.assertFalse(Edge((0, 2)) in graph.edges)

    def test_edge_pruning_with_keeping(self):
        graph = Graph.from_distance_matrix(
            length_granularity=1,
            distance_matrix=array([
                [0, 3, 5],  # Lion
                [3, 0, 4],  # Tiger
                [5, 4, 0],  # Stripes
            ])
        )
        self.assertFalse(graph.has_orphaned_nodes())
        graph.prune_longest_edges_by_length(3, keep_at_least_n_edges=1)
        self.assertFalse(graph.has_orphaned_nodes())
        graph.prune_longest_edges_by_length(3, keep_at_least_n_edges=0)
        self.assertTrue(graph.has_orphaned_nodes())

    def test_distance_matrix_pruning(self):
        orphan_graph = Graph.from_distance_matrix(
            length_granularity=1,
            distance_matrix=array([
                [0, 3, 5],  # Lion
                [3, 0, 4],  # Tiger
                [5, 4, 0],  # Stripes
            ]),
            ignore_edges_longer_than=3,
            keep_at_least_n_edges=0
        )
        self.assertTrue(orphan_graph.has_orphaned_nodes())
        non_orphan_graph = Graph.from_distance_matrix(
            length_granularity=1,
            distance_matrix=array([
                [0, 3, 5],  # Lion
                [3, 0, 4],  # Tiger
                [5, 4, 0],  # Stripes
            ]),
            ignore_edges_longer_than=3,
            keep_at_least_n_edges=1
        )
        self.assertFalse(non_orphan_graph.has_orphaned_nodes())

    def test_keeping_enough_edges(self):
        wide_graph = Graph.from_distance_matrix(
            length_granularity=1,
            # everything 10 away from everything else
            distance_matrix=10*(ones((20, 20))-eye(20)),
            ignore_edges_longer_than=5,
            keep_at_least_n_edges=3
        )
        self.assertFalse(wide_graph.has_orphaned_nodes())
        for node in wide_graph.nodes:
            self.assertGreaterEqual(len(list(wide_graph.incident_edges(node))), 3)

    def test_keeping_the_right_number_of_edges(self):
        wide_graph = Graph.from_distance_matrix(
            length_granularity=1,
            # everything 10 away from everything else
            distance_matrix=10*(ones((20, 20))-eye(20)),
            ignore_edges_longer_than=5,
            keep_at_least_n_edges=3
        )
        self.assertFalse(wide_graph.has_orphaned_nodes())
        for node in wide_graph.nodes:
            self.assertEqual(len(list(wide_graph.incident_edges(node))), 3)


class TestGraphTopology(unittest.TestCase):

    def test_connected(self):
        connected_graph = Graph.from_adjacency_matrix(
            adjacency_matrix=array([
                [0, 1, 1],  # Lion
                [1, 0, 1],  # Tiger
                [1, 1, 0],  # Stripes
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
