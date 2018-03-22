"""
===========================
Temporal spreading activation.

Terminology:

`Node` and `Edge` refer to the underlying graph.
`Charge` and `Impulse` refer to activations within nodes and edges respectively.
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
import networkx
from networkx import Graph
from matplotlib import pyplot

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class EdgeDataKey(object):
    WEIGHT   = "weight"
    LENGTH   = "length"
    IMPULSES = "impulses"


class NodeDataKey(object):
    CHARGE   = "charge"


class Charge(object):
    """An activation living in a node."""
    def __init__(self, node, initial_activation: float, decay_function: callable):
        self.node = node
        self._original_activation = initial_activation
        self._decay_function = decay_function

        self.time_since_last_activation = 0

    @property
    def activation(self):
        """Current activation."""
        return self._decay_function(self.time_since_last_activation, self._original_activation)

    def decay(self):
        # Decay is handled by the `self.activation` property; all we need to do is increment the age.
        self.time_since_last_activation += 1

    def __str__(self):
        return f"{self.node}: {self.activation} ({self.time_since_last_activation})"


class Impulse(object):
    """An activation travelling through an edge."""
    def __init__(self, source_node, target_node, initial_activation: float, decay_function: callable):
        self.source_node = source_node
        self.target_node = target_node
        self._original_activation = initial_activation
        self._decay_function = decay_function

        self._expired: bool = False
        self.age: int = 0

    @property
    def is_expired(self):
        return self._expired

    @property
    def activation(self):
        """Current activation."""
        return self._decay_function(self.age, self._original_activation)

    def propagate_and_decay(self):
        # Decay is handled by the `self.activation` property; all we need to do is increment the age.
        self.age += 1

    def expire(self):
        self._expired = True

    def __str__(self) -> str:
        return f"Impulse: {self.source_node} → {self.activation} → {self.target_node} ({self.age})"


class TemporalSpreadingActivation(object):

    def __init__(self,
                 graph: Graph,
                 threshold: float,
                 weight_coefficient: float,
                 granularity: int,
                 node_decay_function: callable,
                 edge_decay_function: callable,
                 ):

        # Parameters
        self.threshold = threshold
        self.weight_coefficient = weight_coefficient
        self.granularity = granularity
        # These decay functions should be stateless, and convert an original activation and an age into a current
        # activation.
        self.node_decay_function: callable = node_decay_function
        self.edge_decay_function: callable = edge_decay_function

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
    def create_decay_function_gaussian_with_params(sd, height_coef=1, centre=0) -> callable:
        def decay_function(age, original_activation):
            return original_activation * height_coef * exp((-1) * ((age - centre) ** 2) / (2 * sd * sd))
        return decay_function

    def iter_impulses(self):
        for v1, v2, e_data in self.graph.edges(data=True):
            for impulse in e_data[EdgeDataKey.IMPULSES]:
                yield impulse

    def _initialise_graph(self):
        for _n1, _n2, e_data in self.graph.edges(data=True):
            # Compute lengths
            e_data[EdgeDataKey.LENGTH] = ceil(e_data[EdgeDataKey.WEIGHT] * self.granularity)
            # Convert distances to weights
            e_data[EdgeDataKey.WEIGHT] = (1 - e_data[EdgeDataKey.WEIGHT]) * self.weight_coefficient

    def reset(self):
        """Reset SA graph so it looks as if it's just been built."""
        # Set all node activations to zero
        for n, n_data in self.graph.nodes(data=True):
            n_data[NodeDataKey.CHARGE] = Charge(n, 0, self.node_decay_function)
        # Delete all activations
        for _n1, _n2, e_data in self.graph.edges(data=True):
            e_data[EdgeDataKey.IMPULSES]: List[Impulse] = []

    def _decay_nodes(self):
        """Decays activation in each node."""
        for n, n_data in self.graph.nodes(data=True):
            charge = n_data[NodeDataKey.CHARGE]
            if charge is not None:
                charge.decay()

    def _propagate_impulses(self):
        """Propagates impulses along connections."""
        impulses_at_destination_nodes = []
        for _n1, _n2, e_data in self.graph.edges(data=True):
            for impulse in e_data[EdgeDataKey.IMPULSES]:

                impulse.propagate_and_decay()

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

    def tick(self):
        self._decay_nodes()
        self._propagate_impulses()

    def activate_node(self, n, activation: float):
        # Accumulate activation
        existing_charge: Charge = self.graph.nodes[n][NodeDataKey.CHARGE]
        new_charge = Charge(n, activation + existing_charge.activation, self.node_decay_function)
        self.graph.nodes[n][NodeDataKey.CHARGE] = new_charge

        # Rebroadcast

        # For each incident edge
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

            initial_activation = e_data[EdgeDataKey.WEIGHT] * new_charge.activation

            # Check if another impulse has been released into this edge this tick
            existing_impulses = [i for i in e_data[EdgeDataKey.IMPULSES]
                                 if i.age == 0
                                 and i.target_node == target_node
                                 and not i.is_expired]

            # If there are no existing ones, we don't need to do anything
            if len(existing_impulses) == 0:
                pass
            # If one has been released this tick, we add its activation to the existing one and delete it
            elif len(existing_impulses) == 1:
                initial_activation += existing_impulses[0].activation
                existing_impulses[0].expire()
            # We are performing this check for each impulse released, so there should only ever be 0 or 1 existing
            # impulse (if there were more, we already would have combined them.
            elif len(existing_impulses) > 1:
                raise Exception("There should only ever be 0 or 1 existing impulse released this turn")

            new_impulse = Impulse(source_node=n,
                                  target_node=target_node,
                                  initial_activation=initial_activation,
                                  decay_function=self.edge_decay_function)

            e_data[EdgeDataKey.IMPULSES].append(new_impulse)

    def __str__(self):
        string_builder = ""
        string_builder += "Nodes:\n"
        for node, n_data in self.graph.nodes(data=True):
            string_builder += f"\t{str(n_data[NodeDataKey.CHARGE])}\n"
        string_builder += "Edges:\n"
        for n1, n2, e_data in self.graph.edges(data=True):
            impulse_list = [str(i) for i in e_data[EdgeDataKey.IMPULSES]]
            string_builder += f"\t{n1}–{n2}:\n"
            for impulse in impulse_list:
                string_builder += f"\t\t{impulse}\n"
        return string_builder

    def log_graph(self):
        [logger.info(f"{line}") for line in str(self).strip().split('\n')]

    def draw_graph(self, pdf=None, pos=None, frame_label=None):
        """Draws and saves or shows the graph."""

        # Use supplied position, or recompute
        if pos is None:
            pos = networkx.spring_layout(self.graph, iterations=500)

        cmap = pyplot.get_cmap("autumn")

        # Prepare labels

        node_labels = {}
        for n, n_data in self.graph.nodes(data=True):
            node_labels[n] = f"{n}\n{n_data[NodeDataKey.CHARGE].activation:.3g}"

        edge_labels = {}
        for v1, v2, e_data in self.graph.edges(data=True):
            weight = e_data[EdgeDataKey.WEIGHT]
            length = e_data[EdgeDataKey.LENGTH]
            impulses = e_data[EdgeDataKey.IMPULSES]
            edge_labels[(v1, v2)] = f"w={weight:.3g}; l={length}"

        # Prepare impulse points and labels
        impulse_data = []
        for v1, v2, e_data in self.graph.edges(data=True):
            length = e_data[EdgeDataKey.LENGTH]
            impulses = e_data[EdgeDataKey.IMPULSES]
            if len(impulses) == 0:
                continue
            x1, y1 = pos[v1]
            x2, y2 = pos[v2]
            for i in impulses:

                if i.age == 0:
                    continue

                if i.target_node == v2:
                    # Travelling v1 → v2
                    fraction = i.age / length
                elif i.target_node == v1:
                    # Travelling v2 → v1
                    fraction = 1 - (i.age / length)
                else:
                    raise Exception(f"Inappropriate target node {i.target_node}")
                x = x1 + (fraction * (x2 - x1))
                y = y1 + (fraction * (y2 - y1))

                c = cmap(i.activation)

                impulse_data.append([x, y, c, i, length])

        pyplot.figure()

        # Draw the nodes
        networkx.draw_networkx_nodes(
            self.graph, pos=pos, with_labels=False,
            node_color=[n_data[NodeDataKey.CHARGE].activation for n, n_data in self.graph.nodes(data=True)],
            cmap=cmap, vmin=0, vmax=1,
            node_size=400)
        networkx.draw_networkx_labels(self.graph, pos=pos, labels=node_labels)

        # Draw the edges
        networkx.draw_networkx_edges(
            self.graph, pos=pos, with_labels=False,
        )
        networkx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=edge_labels, font_size=6)

        # Draw impulses
        for x, y, c, i, l in impulse_data:
            pyplot.plot(x, y, marker='o', markersize=5, color=c)
            pyplot.text(x, y, f"{i.activation:.3g} ({i.age}/{l})")

        # Draw frame_label
        if frame_label is not None:
            pyplot.annotate(frame_label, horizontalalignment='left', verticalalignment='bottom', xy=(1, 0), xycoords='axes fraction')

        # Style figure
        pyplot.axis('off')

        # Save or show graph
        if pdf is not None:
            pdf.savefig()
            pyplot.close()
        else:
            pyplot.show()

        return pos
