"""
===========================
Temporal spreading activation.
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

import logging
from math import ceil
from typing import List
from numpy import exp

from networkx import Graph

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class EdgeDataKey(object):
    WEIGHT = "weight"
    LENGTH = "length"
    IMPULSES = "impulses"


class NodeDataKey(object):
    ACTIVATION = "activation"


class Impulse(object):
    def __init__(self, target_node, activation: float, decay_function: callable, age: int = 0):
        self.age: int = age
        self.target_node = target_node
        self._original_activation: float = activation

        self._decay_function: callable = decay_function

        self._expired: bool = False

    @property
    def is_expired(self):
        return self._expired

    @property
    def activation(self):
        return self._decay_function(self.age, self._original_activation)

    def expire(self):
        self._expired = True

    def propagate(self):
        self.age += 1

    def __str__(self) -> str:
        return f"Impulse: {self.activation} -> {self.target_node} ({self.age})"


class TemporalSpreadingActivation(object):

    def __init__(self,
                 graph: Graph,
                 threshold: float,
                 weight_coefficient: float,
                 granularity: int,
                 decay_function: callable):

        # Parameters
        self.threshold = threshold
        self.weight_coefficient = weight_coefficient
        self.granularity = granularity
        self.decay_function: callable = decay_function

        # Underlying graph: weighted, undirected
        self.graph: Graph = graph

        self._initialise_graph()

        self.reset()

    @staticmethod
    def create_decay_function_exponential_with_params(decay_factor) -> callable:
        def decay_function(age, original_activation):
            return original_activation * (decay_factor ** age)
        return decay_function

    @staticmethod
    def create_decay_function_gaussian_with_params(sd, height=1, centre=0) -> callable:
        def decay_function(age, original_activation):
            return height * original_activation * exp((-1) * ((age - centre) ** 2) / (2 * sd * sd))
        return decay_function

    def _initialise_graph(self):
        for _n1, _n2, e_data in self.graph.edges(data=True):
            # Compute lengths
            e_data[EdgeDataKey.LENGTH] = ceil(e_data[EdgeDataKey.WEIGHT] * self.granularity)
            # Convert distances to weights
            e_data[EdgeDataKey.WEIGHT] = (1 - e_data[EdgeDataKey.WEIGHT]) * self.weight_coefficient

    def reset(self):
        # Set all node activations to zero
        for _n, n_data in self.graph.nodes(data=True):
            n_data[NodeDataKey.ACTIVATION] = 0
        # Delete all activations
        for _n1, _n2, e_data in self.graph.edges(data=True):
            e_data[EdgeDataKey.IMPULSES]: List[Impulse] = []

    def _node_decay(self):
        for n, n_data in self.graph.nodes(data=True):
            # TODO: how to apply activation function here? Do nodes need ages too?
            n_data[NodeDataKey.ACTIVATION] *= self.decay

    def _propagate_impulses(self):
        impulses_at_destination_nodes = []
        for _n1, _n2, e_data in self.graph.edges(data=True):
            for impulse in e_data[EdgeDataKey.IMPULSES]:

                impulse.propagate()

                # Expire if below threshold
                if impulse.activation < self.threshold:
                    impulse.expire()
                    continue

                # Apply to node if destination reached
                if impulse.age >= e_data[EdgeDataKey.LENGTH]:
                    impulses_at_destination_nodes.append(impulse)
                    impulse.expire()
                    continue
            # Remove expired impulses
            e_data[EdgeDataKey.IMPULSES] = [i for i in e_data[EdgeDataKey.IMPULSES]
                                            if not i.is_expired]

        # Apply new activations to nodes
        for impulse in impulses_at_destination_nodes:
            self.activate_node(impulse.target_node, impulse.activation)

    def activate_node(self, n, activation: float):
        # Accumulate activation
        self.graph.nodes[n][NodeDataKey.ACTIVATION] += activation
        # Rebroadcast
        for n1, n2, e_data in self.graph.edges(n, data=True):
            if n == n1:
                target_node = n2
            elif n == n2:
                target_node = n1
            else:
                raise ValueError()

            # TODO: This still doesn't seem like the most efficient way to do this, as we may make the same check
            # TODO: multiple times.
            # TODO: Perhaps instead have a separate place to accumulate new emissions, and do the agglomeration there
            # TODO: before putting them in the pipes.
            new_impulse = Impulse(target_node=target_node,
                                  activation=e_data[EdgeDataKey.WEIGHT] * self.graph.nodes[n][NodeDataKey.ACTIVATION],
                                  decay_function=self.decay_function)
            # Check if another impulse was released this tick
            existing_impulses = [i for i in e_data[EdgeDataKey.IMPULSES]
                                 if i.age == new_impulse.age
                                 and i.target_node == new_impulse.target_node
                                 and not i.is_expired]
            if len(existing_impulses) == 1:
                existing_impulses[0].activation += new_impulse.activation
            # If there are no existing ones, add the new one
            elif len(existing_impulses) == 0:
                e_data[EdgeDataKey.IMPULSES].append(new_impulse)
            else:
                raise Exception("There should only ever be 0 or 1 existing impulse released this turn")

    def __str__(self):
        string_builder = ""
        string_builder += "Nodes:\n"
        for node, n_data in self.graph.nodes(data=True):
            string_builder += f"\t{node}: {n_data[NodeDataKey.ACTIVATION]}\n"
        string_builder += "Edges:\n"
        for n1, n2, e_data in self.graph.edges(data=True):
            impulse_list = [str(i) for i in e_data[EdgeDataKey.IMPULSES]]
            string_builder += f"\t{n1}–{n2}:\n"
            for impulse in impulse_list:
                string_builder += f"\t\t{impulse}\n"
        return string_builder

    def log_graph(self):
        [logger.info(f"{line}") for line in str(self).strip().split('\n')]

    def tick(self):
        self._node_decay()
        self._propagate_impulses()
