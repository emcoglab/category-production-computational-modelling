"""
===========================
Math utils.
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
from typing import Sequence

from numpy import percentile
from numpy.core.umath import float_power, exp, pi, sqrt
from scipy.stats import lognorm

from model.utils.maths_core import gaussian_decay, exponential_decay

TAU: float = 2 * pi


def decay_function_constant() -> callable:
    """
    Constant decay function (i.e. does not decay).
    :return:
    """
    def decay_function(_age, original_activation):
        return original_activation

    return decay_function


def decay_function_exponential_with_decay_factor(decay_factor) -> callable:
    # Decay formula for activation a, original activation a_0, decay factor d, time t:
    #   a = a_0 d^t
    #
    # In traditional formulation of exponential decay, this is equivalent to:
    #   a = a_0 e^(-λt)
    # where λ is the decay constant.
    #
    # I.e.
    #   d = e^(-λ)
    #   λ = - ln d
    assert 0 < decay_factor <= 1

    def decay_function(age, original_activation):
        return exponential_decay(age=age, original_activation=original_activation, decay_factor=decay_factor)

    return decay_function


def decay_function_exponential_with_half_life(half_life) -> callable:
    assert half_life > 0
    # Using notation from above, with half-life hl
    #   λ = ln 2 / ln hl
    #   d = 2 ^ (- 1 / hl)
    decay_factor = float_power(2, - 1 / half_life)
    return decay_function_exponential_with_decay_factor(decay_factor)


def decay_function_gaussian_with_sd(sd, height_coef=1, centre=0) -> callable:
    """
    Gaussian decay function with sd specifying the number of ticks.
    :param sd:
    :param height_coef:
    :param centre:
    :return:
    """
    assert height_coef > 0
    assert sd > 0

    # The actual normal pdf has height 1/sqrt(2 pi sd^2). We want its natural height to be 1 (which is then scaled
    # by the original activation), so we force that here.
    reset_height = sqrt(TAU * sd * sd)

    def decay_function(age, original_activation):
        return gaussian_decay(age=age,
                              original_activation=original_activation,
                              height_coef=height_coef,
                              reset_height=reset_height,
                              centre=centre,
                              sd=sd)

    return decay_function


def decay_function_lognormal_median(median: float, shape: float) -> callable:
    """
    Lognormal survival decay function, parameterised by the median and the shape.
    :param median:
    :param shape:
        The spread or shape
    :return:
        Decay function
    """
    def decay_function(age, original_activation):
        return original_activation * lognorm.sf(age, s=shape, scale=median)

    return decay_function


def decay_function_lognormal_mean(mu: float, shape: float) -> callable:
    """
    Lognormal survival decay function, parameterised by the mean and the shape.

    (Use of median is preferred.)
    :param mu:
    :param shape:
    :return:
    """
    return decay_function_lognormal_median(exp(mu), shape)


def mean(*items):
    return sum(items) / len(items)


def nearest_value_at_quantile(values, quantile):
    """
    Return items(s) at specified quantile.
    :param values
        Sequence of values from which to form a distribution.
    :param quantile:
        float in range of [0,1] (or sequence of floats)
    :return:
        value (or sequence of values) marking specified quantile.
        Returned values will not be interpolated - nearest values to the quantile will be given.
    """
    # If one quantile provided
    if isinstance(quantile, float):
        centile = 100 * quantile
    # If sequence of quantiles provided
    elif isinstance(quantile, Sequence):
        centile = [100 * q for q in quantile]
    else:
        raise TypeError()
    # noinspection PyTypeChecker
    value = percentile(values,
                       # I don't know why Pycharm thinks its expecting an int here; it shouldn't be
                       centile, interpolation="nearest")
    return value