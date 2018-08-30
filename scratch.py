import numpy

from model.temporal_spatial_expansion import PointsInSpace, Point

p1 = Point(numpy.array([1, 2, 3]), "p1")
p2 = Point(numpy.array([4, 5, 6]), "p2")

pis = PointsInSpace(numpy.vstack((p1.vector, p2.vector)), {i: p.label for i, p in enumerate([p1, p2])})

pass
