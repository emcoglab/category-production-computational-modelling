import scipy.stats

import numpy
from numpy import linspace

from matplotlib import pyplot

from model.graph import Graph

xs = numpy.linspace(0, 10, 1000)

gs = scipy.stats.norm.pdf(xs, loc=0, scale=5)
ss = scipy.stats.norm.sf(xs, loc=0, scale=5)


print("Done")
