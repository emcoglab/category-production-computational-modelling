from matplotlib import pyplot
from numpy import linspace, log, exp
from scipy.stats import lognorm, norm

x = linspace(0, 10, 1000)
sigma = 0.35
median = 3.75
mu = log(median)
y = 1-lognorm.cdf(x, s=sigma, scale=median)
y = lognorm.sf(x, s=sigma, scale=median)
# y = 1-norm.cdf(x, loc=mu, scale=sigma)
pyplot.plot(x, y)
pyplot.show()
