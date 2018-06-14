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
            pruning_threshold=.2,
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
            pruning_threshold=.2,
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

    def test_gaussian_decay_same_granularity_different_function_maker(self):
        """
        The values in this test haven't been manually verified, and has so far only been used to test that refactoring
        has no effect.
        """

        distance_matrix = array([
            [.0, .5],
            [.5, .0]
        ])
        sd_frac = 0.42
        tsa_100 = TemporalSpreadingActivation(
            graph=TemporalSpreadingActivation.graph_from_distance_matrix(
                distance_matrix=distance_matrix,
                length_granularity=100,
                weighted_graph=False
            ),
            pruning_threshold=0,
            node_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_half_life(50),
            edge_decay_function=TemporalSpreadingActivation.decay_function_gaussian_with_sd_fraction(sd_frac, 100)
        )
        tsa = TemporalSpreadingActivation(
            graph=TemporalSpreadingActivation.graph_from_distance_matrix(
                distance_matrix=distance_matrix,
                length_granularity=100,
                weighted_graph=False
            ),
            pruning_threshold=0,
            node_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_half_life(50),
            edge_decay_function=TemporalSpreadingActivation.decay_function_gaussian_with_sd(42)
        )

        tsa_100.activate_node(n=0, activation=1.0)
        tsa.activate_node(n=0, activation=1.0)

        for tick in range(1, 50):
            tsa_100.tick()
            tsa.tick()

        # Same granularity, different function-maker
        self.assertAlmostEqual(
            tsa_100.graph.edges[0, 1]["impulses"][0].activation_at_destination,
            tsa.graph.edges[0, 1]["impulses"][0].activation_at_destination
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
            graph=TemporalSpreadingActivation.graph_from_distance_matrix(
                distance_matrix=distance_matrix,
                length_granularity=granularity,
                weighted_graph=False
            ),
            pruning_threshold=0,
            node_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_half_life(50),
            edge_decay_function=TemporalSpreadingActivation.decay_function_gaussian_with_sd_fraction(sd_frac, granularity)
        )
        granularity = 1000
        tsa_1000 = TemporalSpreadingActivation(
            graph=TemporalSpreadingActivation.graph_from_distance_matrix(
                distance_matrix=distance_matrix,
                length_granularity=granularity,
                weighted_graph=False
            ),
            pruning_threshold=0,
            node_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_half_life(50),
            edge_decay_function=TemporalSpreadingActivation.decay_function_gaussian_with_sd_fraction(sd_frac, granularity)
        )

        tsa_390.activate_node(n=0, activation=1.0)
        tsa_1000.activate_node(n=0, activation=1.0)

        # Different granularity, same function-maker
        self.assertAlmostEqual(
            tsa_390.graph.edges[0, 1]["impulses"][0].activation_at_destination,
            tsa_1000.graph.edges[0, 1]["impulses"][0].activation_at_destination
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
