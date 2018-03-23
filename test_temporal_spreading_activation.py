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

from numpy import array

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
            node_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_params(decay_factor=0.90),
            edge_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_params(decay_factor=0.90)
        )

        sa.activate_node("lion", 1)

        for i in range(1, 13):
            sa.tick()

        self.assertAlmostEqual(sa.graph.nodes(data=True)["lion"]["charge"].activation, 0.7654122868171584)
        self.assertAlmostEqual(sa.graph.nodes(data=True)["tiger"]["charge"].activation, 0.49227468208638314)
        self.assertAlmostEqual(sa.graph.nodes(data=True)["stripes"]["charge"].activation, 0.23159221991442014)


if __name__ == '__main__':
    unittest.main()
