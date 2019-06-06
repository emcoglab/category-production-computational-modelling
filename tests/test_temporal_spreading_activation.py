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

from model.events import ItemFiredEvent
from .approximate_comparator.approximate_comparator import is_almost_equal

from model.graph import Graph
from model.temporal_spreading_activation import TemporalSpreadingActivation
from model.utils.maths import make_decay_function_exponential_with_decay_factor, \
    make_decay_function_exponential_with_half_life, make_decay_function_gaussian_with_sd


class TestTemporalSpreadingActivationToyExample(unittest.TestCase):

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
        tsa = TemporalSpreadingActivation(
            graph=graph,
            idx2label={0: "lion", 1: "tiger", 2: "stripes"},
            firing_threshold=0.3,
            impulse_pruning_threshold=.1,
            node_decay_function=make_decay_function_exponential_with_decay_factor(decay_factor=0.9),
            edge_decay_function=make_decay_function_exponential_with_decay_factor(decay_factor=0.9),
        )

        # t = 0
        e = tsa.activate_item_with_label("lion", 1)
        self.assertIsNotNone(e)
        self.assertEqual(e, ItemFiredEvent(time=0, item=0, activation=1.0))

        for t in range(1, 3):
            es = tsa.tick()
            self.assertEqual(len(es), 0)

        # t = 3
        es = tsa.tick()
        self.assertEqual(len(es), 1)
        self.assertTrue(ItemFiredEvent(time=3, item=1, activation=0.7289999127388) in es)

        for t in range(4, 6):
            es = tsa.tick()
            self.assertEqual(len(es), 0)

        # t = 6
        es = tsa.tick()
        self.assertEqual(len(es), 1)
        self.assertTrue(ItemFiredEvent(time=6, item=2, activation=0.5314409136772156) in es)

        for t in range(7, 12):
            es = tsa.tick()
            self.assertEqual(len(es), 0)

        # t = 12
        es = tsa.tick()
        self.assertEqual(len(es), 1)
        self.assertTrue(ItemFiredEvent(time=12, item=0, activation=0.5648589134216309) in es)

        for t in range(13, 15):
            es = tsa.tick()
            self.assertEqual(len(es), 0)

        # t = 15
        es = tsa.tick()
        self.assertEqual(len(es), 1)
        self.assertTrue(ItemFiredEvent(time=15, item=1, activation=0.6176731288433075) in es)

        for t in range(16, 17):
            es = tsa.tick()
            self.assertEqual(len(es), 0)


class TestDecayFunctions(unittest.TestCase):

    def test_exponential_decay_factor_1(self):
        d = 1
        t = 27
        a_0 = 0.64
        self.assertAlmostEqual(
            a_0,
            make_decay_function_exponential_with_decay_factor(d)(t, a_0)
        )

    def test_two_form_of_exponential_decay_are_equal(self):
        d = 0.93
        λ = - log(d)
        hl = log(2) / λ
        t = 30
        a_0 = .7
        exponential_decay_via_d  = make_decay_function_exponential_with_decay_factor(decay_factor=d)
        exponential_decay_via_hl = make_decay_function_exponential_with_half_life(half_life=hl)
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
            graph=Graph.from_distance_matrix(
                distance_matrix=distance_matrix,
                length_granularity=100,
            ),
            firing_threshold=0.3,
            impulse_pruning_threshold=0,
            node_decay_function=make_decay_function_exponential_with_half_life(50),
            edge_decay_function=make_decay_function_gaussian_with_sd(0.42 * 100),
            idx2label=dict()
        )
        tsa = TemporalSpreadingActivation(
            graph=Graph.from_distance_matrix(
                distance_matrix=distance_matrix,
                length_granularity=100,
            ),
            firing_threshold=0.3,
            impulse_pruning_threshold=0,
            node_decay_function=make_decay_function_exponential_with_half_life(50),
            edge_decay_function=make_decay_function_gaussian_with_sd(42),
            idx2label=dict()
        )

        tsa_frac.activate_item_with_idx(idx=0, activation=1.0)
        tsa.activate_item_with_idx(idx=0, activation=1.0)

        for tick in range(1, 50):
            tsa_frac.tick()
            tsa.tick()

        # Same granularity, different function-maker
        self.assertAlmostEqual(
            tsa_frac.activation_of_item_with_idx(0),
            tsa.activation_of_item_with_idx(0)
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
            graph=Graph.from_distance_matrix(
                distance_matrix=distance_matrix,
                length_granularity=granularity,
            ),
            impulse_pruning_threshold=0,
            firing_threshold=0.5,
            node_decay_function=make_decay_function_exponential_with_half_life(50),
            edge_decay_function=make_decay_function_gaussian_with_sd(sd_frac * granularity),
            idx2label=dict()
        )
        granularity = 1000
        tsa_1000 = TemporalSpreadingActivation(
            graph=Graph.from_distance_matrix(
                distance_matrix=distance_matrix,
                length_granularity=granularity,
            ),
            impulse_pruning_threshold=0,
            firing_threshold=0.5,
            node_decay_function=make_decay_function_exponential_with_half_life(50),
            edge_decay_function=make_decay_function_gaussian_with_sd(sd_frac * granularity),
            idx2label=dict()
        )

        tsa_390.activate_item_with_idx(idx=0, activation=1.0)
        tsa_1000.activate_item_with_idx(idx=0, activation=1.0)

        # Different granularity, same function-maker
        almost_equal = is_almost_equal(
            # There will be only one impulse in each edge at this point so we can just grab it without worrying about
            # the lack of ordering in the set of impulses.
            set(float(v) for v in tsa_390.impulses_headed_for(1).values()),
            set(float(v) for v in tsa_1000.impulses_headed_for(1).values()),
            places=5
        )

        self.assertTrue(almost_equal)


if __name__ == '__main__':
    unittest.main()
