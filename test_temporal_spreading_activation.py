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
    def test_v0_worked_example_node_values(self):
        distance_matrix = array([
            [.0, .3, .6],  # Lion
            [.3, .0, .4],  # Tiger
            [.6, .4, .0],  # Stripes
        ])
        graph = TemporalSpreadingActivation.graph_from_distance_matrix(
            distance_matrix=distance_matrix,
            length_granularity=10,
            relabelling_dict={0: "lion", 1: "tiger", 2: "stripes"}
        )
        sa = TemporalSpreadingActivation(
            graph=graph,
            threshold=.2,
            node_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_decay_factor(decay_factor=0.80),
            edge_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_decay_factor(decay_factor=0.80),
            activation_cap=1
        )

        sa.activate_node("lion", 1)

        for i in range(1, 14):
            sa.tick()

        self.assertAlmostEqual(sa.graph.nodes(data=True)["lion"]["charge"].activation,    0.2748779)
        self.assertAlmostEqual(sa.graph.nodes(data=True)["tiger"]["charge"].activation,   0.1649267)
        self.assertAlmostEqual(sa.graph.nodes(data=True)["stripes"]["charge"].activation, 0.1099512)

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


if __name__ == '__main__':
    unittest.main()
