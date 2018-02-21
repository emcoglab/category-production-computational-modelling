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
from networkx import convert_matrix, relabel_nodes

from temporal_spreading_activation import TemporalSpreadingActivation


class TestUnsummedCoOccurrenceModel(unittest.TestCase):
    def test_v0_worked_example_node_values(self):
        distance_matrix = array([
            [.0, .3, .6],  # Lion
            [.3, .0, .4],  # Tiger
            [.6, .4, .0],  # Stripes
        ])
        graph = convert_matrix.from_numpy_array(distance_matrix)
        graph = relabel_nodes(graph, {0: "lion", 1: "tiger", 2: "stripes"}, copy=False)

        sa = TemporalSpreadingActivation(
            graph=graph,
            threshold=.2,
            weight_coefficient=1,
            granularity=10,
            decay_function=TemporalSpreadingActivation.create_decay_function_exponential_with_params(decay_factor=0.90)
        )

        sa.activate_node("lion", 1)

        for i in range(1, 13):
            sa.tick()

        self.assertAlmostEqual(sa.graph.nodes(data=True)["lion"]["activation"], 0.7654122868171584)
        self.assertAlmostEqual(sa.graph.nodes(data=True)["tiger"]["activation"], 0.49227468208638314)
        self.assertAlmostEqual(sa.graph.nodes(data=True)["stripes"]["activation"], 0.23159221991442014)


if __name__ == '__main__':
    unittest.main()
