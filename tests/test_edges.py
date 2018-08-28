"""
===========================
Tests for Edges.
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

from model.graph import Edge, Node


class TestEdges(unittest.TestCase):

    def test_edge_node_order_does_not_matter(self):
        edge1 = Edge((Node(0), Node(1)))
        edge2 = Edge((Node(1), Node(0)))
        self.assertEqual(edge1, edge2)


if __name__ == '__main__':
    unittest.main()
