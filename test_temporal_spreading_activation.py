"""
===========================
Tests for TemporalSpreadingActivation.
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

from numpy import array, log

from temporal_spreading_activation import TemporalSpreadingActivation


class TestUnsummedCoOccurrenceModel(unittest.TestCase):

    def test_worked_example_weighted_node_values(self):
        distance_matrix = array([
            [.0, .3, .6],  # Lion
            [.3, .0, .4],  # Tiger
            [.6, .4, .0],  # Stripes
        ])
        graph = TemporalSpreadingActivation.graph_from_distance_matrix(
            distance_matrix=distance_matrix,
            weighted_graph=True,
            length_granularity=10,
            relabelling_dict={0: "lion", 1: "tiger", 2: "stripes"}
        )
        sa = TemporalSpreadingActivation(
            graph=graph,
            threshold=.2,
            node_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_decay_factor(
                decay_factor=0.9),
            edge_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_decay_factor(
                decay_factor=0.9),
            activation_cap=1
        )

        sa.activate_node("lion", 1)

        for i in range(1, 14):
            sa.tick()

        self.assertAlmostEqual(sa.graph.nodes(data=True)["lion"]["charge"].activation,    0.6888710581)
        self.assertAlmostEqual(sa.graph.nodes(data=True)["tiger"]["charge"].activation,   0.4430472139)
        self.assertAlmostEqual(sa.graph.nodes(data=True)["stripes"]["charge"].activation, 0.4742613262)

    def test_worked_example_unweighted_node_values(self):
        distance_matrix = array([
            [.0, .3, .6],  # Lion
            [.3, .0, .4],  # Tiger
            [.6, .4, .0],  # Stripes
        ])
        graph = TemporalSpreadingActivation.graph_from_distance_matrix(
            distance_matrix=distance_matrix,
            weighted_graph=False,
            length_granularity=10,
            relabelling_dict={0: "lion", 1: "tiger", 2: "stripes"}
        )
        sa = TemporalSpreadingActivation(
            graph=graph,
            threshold=.2,
            node_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_decay_factor(decay_factor=0.8),
            edge_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_decay_factor(decay_factor=0.8),
            activation_cap=1
        )

        sa.activate_node("lion", 1)

        for i in range(1, 14):
            sa.tick()

        self.assertAlmostEqual(sa.graph.nodes(data=True)["lion"]["charge"].activation,    0.2748779)
        self.assertAlmostEqual(sa.graph.nodes(data=True)["tiger"]["charge"].activation,   0.1649267)
        self.assertAlmostEqual(sa.graph.nodes(data=True)["stripes"]["charge"].activation, 0.1099512)

class TestDecayFunctions(unittest.TestCase):

    def test_exponential_decay_factor_1(self):
        d = 1
        t = 27
        a_0 = 0.64
        self.assertEqual(
            a_0,
            TemporalSpreadingActivation.decay_function_exponential_with_decay_factor(d)(t, a_0)
        )

    def test_two_form_of_exponential_decay_are_equal(self):
        d = 0.93
        λ = - log(d)
        hl = log(2) / λ
        t = 30
        a_0 = .7
        exponential_decay_via_d  = TemporalSpreadingActivation.decay_function_exponential_with_decay_factor(decay_factor=d)
        exponential_decay_via_hl = TemporalSpreadingActivation.decay_function_exponential_with_half_life(half_life=hl)
        self.assertAlmostEqual(
            exponential_decay_via_d(t, a_0),
            exponential_decay_via_hl(t, a_0)
        )


class TestGraphPruning(unittest.TestCase):

    def test_distance_pruning_sub_threshold_edges_gone(self):
        LION    = 0
        STRIPES = 2

        distance_matrix = array([
            [.0, .3, .6],  # Lion
            [.3, .0, .4],  # Tiger
            [.6, .4, .0],  # Stripes
        ])
        granularity = 10
        pruning_threshold = 5

        graph = TemporalSpreadingActivation.graph_from_distance_matrix(
            distance_matrix=distance_matrix,
            length_granularity=granularity,
            weighted_graph=False,
            prune_connections_longer_than=pruning_threshold
        )
        self.assertFalse((LION, STRIPES) in graph.edges)

    def test_distance_pruning_supra_threshold_edges_remain(self):
        LION    = 0
        TIGER   = 1
        STRIPES = 2

        distance_matrix = array([
            [.0, .3, .6],  # Lion
            [.3, .0, .4],  # Tiger
            [.6, .4, .0],  # Stripes
        ])
        granularity = 10
        pruning_threshold = 5

        graph = TemporalSpreadingActivation.graph_from_distance_matrix(
            distance_matrix=distance_matrix,
            length_granularity=granularity,
            weighted_graph=False,
            prune_connections_longer_than=pruning_threshold
        )
        self.assertTrue((LION, TIGER) in graph.edges)
        self.assertTrue((TIGER, STRIPES) in graph.edges)


if __name__ == '__main__':
    unittest.main()