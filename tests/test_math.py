"""
===========================
Test math functions.
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

from model.utils.math import _lognormal_pdf, decay_function_lognormal_with_sd, _lognormal_mode


class TestDecayFunctions(unittest.TestCase):

    def test_unaligned_unscaled_lognormal(self):
        x = 2
        sigma = 3
        mu = 1

        self.assertAlmostEqual(_lognormal_pdf(x, mu, sigma), 0.066143474)

    def test_normalised_lognormal(self):

        sd = 3
        mu = 0

        x = _lognormal_mode(mu, sd)

        normalised_decay = decay_function_lognormal_with_sd(sd, False)

        self.assertAlmostEqual(normalised_decay(x, 1), 1)

    def test_realigned_normalised_lognormal(self):
        sd = 6

        x = 0

        realigned_decay = decay_function_lognormal_with_sd(sd, True)

        self.assertAlmostEqual(realigned_decay(x, 1), 1)


if __name__ == '__main__':
    unittest.main()
