"""
===========================
Proof of concept for linguistic spreading activation model.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2017
---------------------------
"""

import logging
import sys
from typing import Set

from networkx import Graph

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class SpreadingActivation(object):
    def __init__(self,
                 decay_factor: float,
                 firing_threshold: float):

        self.decay_factor = decay_factor
        self.firing_threshold = firing_threshold

        # The a weighted, undirected graph
        self.graph: Graph() = Graph()

        # The activations of each node
        self.activations = dict()
        # The additional activations gained in a cycle
        self._delta_activations = dict()
        # Nodes which have has_fired
        self.has_fired = dict()

        # Whether the graph has been is_frozen yet
        self.is_frozen = False

    def add_edge(self, n1, n2, weight: float):
        if self.is_frozen:
            raise Exception("Graph is_frozen, can't update structure!")
        self.graph.add_edge(n1, n2, weight=weight)

        for n in [n1, n2]:
            if n not in self.activations:
                self.activations[n] = 0
                self._delta_activations[n] = 0
                self.has_fired[n] = False

    def freeze(self):
        self.is_frozen = True

    def activate_node(self, n):
        if not self.is_frozen:
            raise Exception("Freeze graph before activating a node")
        if n not in self.activations:
            raise Exception(f"{n} not found in graph!")
        self.activations[n] = 1

    def spread_once(self):
        # Spread activations to unfired neighbours of unfired nodes
        for n, neighbours in self.graph.adjacency():
            for neighbour, edge_attributes in neighbours.items():
                if self.has_fired[neighbour]:
                    continue
                weight = edge_attributes["weight"]
                self._delta_activations[neighbour] += self._clamp01(
                    self.activations[n] * weight * self.decay_factor
                )

        # Update activations of unfired nodes
        for n in self.graph.nodes:
            if not self.has_fired[n]:
                self.activations[n] += self._delta_activations[n]
                self._delta_activations[n] = 0

        # Fire unfired nodes
        for n in self.graph.nodes:
            if not self.has_fired[n]:
                if self.activations[n] > self.firing_threshold:
                    self.has_fired[n] = True

    def spread_n_times(self, n):
        for i in range(n):
            self.spread_once()

    def print_graph(self):
        for n in self.graph.nodes:
            logger.info(f"{n}, {self.activations[n]}")

    @staticmethod
    def _clamp01(x):
        return max(0, min(1, x))


def main():
    sa = SpreadingActivation(decay_factor=0.85, firing_threshold=0.01)
    sa.add_edge(1, 2, weight=0.9)
    sa.add_edge(2, 3, weight=0.9)
    sa.add_edge(3, 4, weight=0.9)
    sa.add_edge(3, 11, weight=0.9)
    sa.add_edge(4, 5, weight=0.9)
    sa.add_edge(5, 6, weight=0.9)
    sa.add_edge(11, 12, weight=0.9)
    sa.add_edge(12, 13, weight=0.9)
    sa.freeze()

    sa.activate_node(1)

    sa.spread_n_times(4)
    sa.print_graph()


def get_vocabulary() -> Set[str]:
    pass


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
