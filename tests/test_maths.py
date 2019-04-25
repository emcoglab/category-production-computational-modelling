"""
===========================
Tests for maths functions.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""

import unittest

from model.utils.maths_core import lognormal_sf, lognormal_pdf, lognormal_cdf
from scipy.stats import lognorm


class TestCython(unittest.TestCase):

    def test_lognormal_pdf(self):

        x = 3.2
        sigma = 2.2

        self.assertAlmostEqual(
            # Library version
            lognorm.pdf(x, s=sigma),
            # cythonised version
            lognormal_pdf(x, sigma)
        )

    def test_lognormal_cdf(self):

        x = 3.2
        sigma = 2.2

        self.assertAlmostEqual(
            # Library version
            lognorm.cdf(x, s=sigma),
            # cythonised version
            lognormal_cdf(x, sigma)
        )

    def test_lognormal_sf(self):

        x = 3.2
        sigma = 2.2

        self.assertAlmostEqual(
            # Library version
            lognorm.sf(x, s=sigma),
            # cythonised version
            lognormal_sf(x, sigma)
        )


if __name__ == '__main__':
    unittest.main()
