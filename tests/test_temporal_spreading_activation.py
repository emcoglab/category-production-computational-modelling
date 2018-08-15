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

from approximate_comparator.approximate_comparator import is_almost_equal
from model.graph import Edge, WeightedGraph, UnweightedGraph
from model.temporal_spreading_activation import TemporalSpreadingActivation, \
    decay_function_exponential_with_decay_factor, decay_function_exponential_with_half_life, \
    decay_function_gaussian_with_sd_fraction, decay_function_gaussian_with_sd


class TestUnsummedCoOccurrenceModel(unittest.TestCase):

    def test_worked_example_weighted_node_values(self):
        distance_matrix = array([
            [.0, .3, .6],  # Lion
            [.3, .0, .4],  # Tiger
            [.6, .4, .0],  # Stripes
        ])
        graph = WeightedGraph.from_distance_matrix(
            distance_matrix=distance_matrix,
            length_granularity=10
        )
        tsa = TemporalSpreadingActivation(
            graph=graph,
            firing_threshold=0.3,
            conscious_access_threshold=0.3,
            impulse_pruning_threshold=.1,
            node_decay_function=decay_function_exponential_with_decay_factor(
                decay_factor=0.9),
            edge_decay_function=decay_function_exponential_with_decay_factor(
                decay_factor=0.9),
            node_relabelling_dictionary={0: "lion", 1: "tiger", 2: "stripes"}
        )

        tsa.activate_node_with_label("lion", 1)

        for i in range(1, 16):
            tsa.tick()

        self.assertAlmostEqual(tsa.activation_of_node_with_label("lion"),    0.377687, places=5)
        self.assertAlmostEqual(tsa.activation_of_node_with_label("tiger"),   0.245422, places=5)
        self.assertAlmostEqual(tsa.activation_of_node_with_label("stripes"), 0.316084, places=5)

    def test_worked_example_unweighted_node_values(self):
        distance_matrix = array([
            [.0, .3, .6],  # Lion
            [.3, .0, .4],  # Tiger
            [.6, .4, .0],  # Stripes
        ])
        graph = UnweightedGraph.from_distance_matrix(
            distance_matrix=distance_matrix,
            length_granularity=10,
        )
        tsa = TemporalSpreadingActivation(
            graph=graph,
            node_relabelling_dictionary={0: "lion", 1: "tiger", 2: "stripes"},
            firing_threshold=0.3,
            conscious_access_threshold=0.3,
            impulse_pruning_threshold=.1,
            node_decay_function=decay_function_exponential_with_decay_factor(decay_factor=0.9),
            edge_decay_function=decay_function_exponential_with_decay_factor(decay_factor=0.9),
        )

        tsa.activate_node_with_label("lion", 1)

        for i in range(1, 16):
            tsa.tick()

        self.assertAlmostEqual(tsa.activation_of_node_with_label("lion"),    0.4118, places=4)
        self.assertAlmostEqual(tsa.activation_of_node_with_label("tiger"),   0.6177, places=4)
        self.assertAlmostEqual(tsa.activation_of_node_with_label("stripes"), 0.2059, places=4)


class TestDecayFunctions(unittest.TestCase):

    def test_exponential_decay_factor_1(self):
        d = 1
        t = 27
        a_0 = 0.64
        self.assertEqual(
            a_0,
            decay_function_exponential_with_decay_factor(d)(t, a_0)
        )

    def test_two_form_of_exponential_decay_are_equal(self):
        d = 0.93
        λ = - log(d)
        hl = log(2) / λ
        t = 30
        a_0 = .7
        exponential_decay_via_d  = decay_function_exponential_with_decay_factor(decay_factor=d)
        exponential_decay_via_hl = decay_function_exponential_with_half_life(half_life=hl)
        self.assertAlmostEqual(
            exponential_decay_via_d(t, a_0),
            exponential_decay_via_hl(t, a_0)
        )

    def test_gaussian_decay_same_granularity_different_function_maker(self):
        """
        The values in this test haven't been manually verified, and has so far only been used to test that refactoring
        has no effect.
        """

        distance_matrix = array([
            [.0, .5],
            [.5, .0]
        ])
        tsa_frac = TemporalSpreadingActivation(
            graph=UnweightedGraph.from_distance_matrix(
                distance_matrix=distance_matrix,
                length_granularity=100,
            ),
            firing_threshold=0.3,
            conscious_access_threshold=0.3,
            impulse_pruning_threshold=0,
            node_decay_function=decay_function_exponential_with_half_life(50),
            edge_decay_function=decay_function_gaussian_with_sd_fraction(0.42, 100),
            node_relabelling_dictionary=dict()
        )
        tsa = TemporalSpreadingActivation(
            graph=UnweightedGraph.from_distance_matrix(
                distance_matrix=distance_matrix,
                length_granularity=100,
            ),
            firing_threshold=0.3,
            conscious_access_threshold=0.3,
            impulse_pruning_threshold=0,
            node_decay_function=decay_function_exponential_with_half_life(50),
            edge_decay_function=decay_function_gaussian_with_sd(42),
            node_relabelling_dictionary=dict()
        )

        tsa_frac.activate_node(n=0, activation=1.0)
        tsa.activate_node(n=0, activation=1.0)

        for tick in range(1, 50):
            tsa_frac.tick()
            tsa.tick()

        # Same granularity, different function-maker
        self.assertAlmostEqual(
            tsa_frac.activation_of_node(0),
            tsa.activation_of_node(0)
        )

    def test_gaussian_decay_different_granularity_same_function_maker(self):
        """
        The values in this test haven't been manually verified, and has so far only been used to test that refactoring
        has no effect.
        """

        distance_matrix = array([
            [.0, .5],
            [.5, .0]
        ])
        sd_frac = 0.42
        granularity = 390
        tsa_390 = TemporalSpreadingActivation(
            graph=UnweightedGraph.from_distance_matrix(
                distance_matrix=distance_matrix,
                length_granularity=granularity,
            ),
            impulse_pruning_threshold=0,
            firing_threshold=0.5,
            conscious_access_threshold=0.5,
            node_decay_function=decay_function_exponential_with_half_life(50),
            edge_decay_function=decay_function_gaussian_with_sd_fraction(sd_frac, granularity),
            node_relabelling_dictionary=dict()
        )
        granularity = 1000
        tsa_1000 = TemporalSpreadingActivation(
            graph=UnweightedGraph.from_distance_matrix(
                distance_matrix=distance_matrix,
                length_granularity=granularity,
            ),
            impulse_pruning_threshold=0,
            firing_threshold=0.5,
            conscious_access_threshold=0.5,
            node_decay_function=decay_function_exponential_with_half_life(50),
            edge_decay_function=decay_function_gaussian_with_sd_fraction(sd_frac, granularity),
            node_relabelling_dictionary=dict()
        )

        tsa_390.activate_node(n=0, activation=1.0)
        tsa_1000.activate_node(n=0, activation=1.0)

        # Different granularity, same function-maker
        almost_equal = is_almost_equal(
            # There will be only one impulse in each edge at this point so we can just grab it without worrying about
            # the lack of ordering in the set of impulses.
            set(float(v) for v in tsa_390.impulses_headed_for(1).values()),
            set(float(v) for v in tsa_1000.impulses_headed_for(1).values()),
            places=5
        )

        self.assertTrue(almost_equal)


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

        graph = UnweightedGraph.from_distance_matrix(
            distance_matrix=distance_matrix,
            length_granularity=granularity,
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

        graph = UnweightedGraph.from_distance_matrix(
            distance_matrix=distance_matrix,
            length_granularity=granularity,
            prune_connections_longer_than=pruning_threshold
        )
        self.assertTrue(Edge((LION, TIGER)) in graph.edges)
        self.assertTrue(Edge((TIGER, STRIPES)) in graph.edges)


if __name__ == '__main__':
    unittest.main()
