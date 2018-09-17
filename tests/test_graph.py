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
from tests.test_materials.metadata import test_graph_file_path, test_graph_importance_file_path


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

    def test_distance_matrix_keeping_enough_edges(self):
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

    def test_pruning_keeping_enough_edges(self):
        wide_graph = Graph.from_distance_matrix(
            length_granularity=1,
            # everything 10 away from everything else
            distance_matrix=10*(ones((20, 20))-eye(20)),
        )
        wide_graph.prune_longest_edges_by_length(length_threshold=5, keep_at_least_n_edges=3)
        self.assertFalse(wide_graph.has_orphaned_nodes())
        for node in wide_graph.nodes:
            self.assertGreaterEqual(len(list(wide_graph.incident_edges(node))), 3)

    def test_loading_keeping_enough_edges(self):
        wide_graph: Graph = Graph.load_from_edgelist(file_path=test_graph_file_path, ignore_edges_longer_than=3, keep_at_least_n_edges=2)
        self.assertFalse(wide_graph.has_orphaned_nodes())
        for node in wide_graph.nodes:
            self.assertGreaterEqual(len(list(wide_graph.incident_edges(node))), 2)


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


class TestGraphImportancePruning(unittest.TestCase):

    def test_full_graph(self):
        full_graph: Graph = Graph.load_from_edgelist(file_path=test_graph_importance_file_path)
        n_edges = len(full_graph.edges)
        n_nodes = len(full_graph.nodes)
        # complete graph K5
        predict_edges = 0.5 * n_nodes * (n_nodes - 1)
        self.assertEqual(n_edges, predict_edges)

    def test_importance_pruned_graph_via_direct_importance_pruning(self):
        importance_pruned_graph = Graph.load_from_edgelist_with_importance_pruning(
            file_path=test_graph_importance_file_path,
            ignore_edges_with_importance_greater_than=50)
        n_edges = len(importance_pruned_graph.edges)
        self.assertEqual(n_edges, 5)


if __name__ == '__main__':
    unittest.main()