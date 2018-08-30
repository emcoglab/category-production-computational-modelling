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

from numpy import array, vstack

from model.points_in_space import Point, PointsInSpace


class TestGraphPruning(unittest.TestCase):

    def test_dimensions(self):

        p1 = Point(array([1, 2, 3]), "p1")
        p2 = Point(array([4, 5, 6]), "p2")

        pis = PointsInSpace(vstack((p1.vector, p2.vector)), {i: p.label for i, p in enumerate([p1, p2])})

        self.assertEqual(pis.n_points, 2)
        self.assertEqual(pis.n_dims, 3)


if __name__ == '__main__':
    unittest.main()
