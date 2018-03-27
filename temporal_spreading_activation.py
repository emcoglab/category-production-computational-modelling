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
from typing import List, Dict

from numpy import exp, ndarray, ones_like, ceil, float_power
import networkx
from networkx import Graph, from_numpy_matrix, relabel_nodes, selfloop_edges
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
                 node_decay_function: callable,
                 edge_decay_function: callable,
                 activation_cap=1,
                 ):
        """
        :param graph:
        Should be an undirected weighted graph with the following data:
            On Nodes:
                (no data required)
            On Edges:
                weight
                length
        To this the following data fields will be added by the constructor:
            On Nodes:
                charge
            On Edges:
                impulses
        :param threshold:
        Activation threshold.
        Impulses which drop (strictly) below this threshold will be deleted.
        :param node_decay_function:
        A function governing the decay of activations on nodes.
        Use the decay_function_*_with_params methods to create these.
        :param edge_decay_function:
        A function governing the decay of activations in connections.
        Use the decay_function_*_with_* methods to create these.
        :param activation_cap:
        (Optional, default 1) Caps node activations at this value. If None is given, activations will not be capped.
        """

        if activation_cap is not None:
            assert activation_cap > 0
            assert threshold <= activation_cap

        # Parameters
        self.threshold = threshold
        self.activation_cap = activation_cap

        # These decay functions should be stateless, and convert an original activation and an age into a current
        # activation.
        self.node_decay_function: callable = node_decay_function
        self.edge_decay_function: callable = edge_decay_function

        # Underlying graph: weighted, undirected
        self.graph: Graph = graph

        # Initialise graph
        self.reset()

    @staticmethod
    def graph_from_distance_matrix(distance_matrix: ndarray,
                                   length_granularity: int,
                                   weight_factor: float = 1,
                                   relabelling_dict=None) -> Graph:
        """
        Produces a Graph of the correct format to underlie a TemporalSpreadingActivation.

        Nodes will be numbered according to the row/column indices of weight_matrix (and so can
        be relabelled accordingly).

        Distances will be converted to weights using x ↦ 1-x.

        Distances will be converted to integer lengths using the supplied scaling factor.

        :param distance_matrix:
        A symmetric distance matrix in numpy format.
        :param length_granularity:
        Distances will be scaled into integer connection lengths using this granularity scaling factor.
        :param weight_factor:
        (Optional, default 1.) Linearly scale weights by this factor
        :param relabelling_dict:
        (Optional.) If provided and not None: A dictionary which maps the integer indices of nodes to
        their desired labels.
        :return:
        A Graph of the correct format.
        """

        weight_matrix = weight_factor * (ones_like(distance_matrix) - distance_matrix)
        length_matrix = ceil(distance_matrix * length_granularity)

        graph = from_numpy_matrix(weight_matrix)

        # Converting from a distance matrix creates self-loop edges, which we have to remove
        graph.remove_edges_from(selfloop_edges(graph))

        # Add lengths to graph data
        for n1, n2, e_data in graph.edges(data=True):
            e_data[EdgeDataKey.LENGTH] = length_matrix[n1][n2]

        # Relabel nodes if dictionary was provided
        if relabelling_dict is not None:
            graph = relabel_nodes(graph, relabelling_dict, copy=False)

        return graph

    @staticmethod
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
            return original_activation * (decay_factor ** age)
        return decay_function

    @staticmethod
    def decay_function_exponential_with_half_life(half_life) -> callable:
        assert half_life > 0
        # Using notation from above, with half-life hl
        #   λ = ln 2 / ln hl
        #   d = 2 ^ (- 1 / hl)
        decay_factor = float_power(2, - 1 / half_life)
        return TemporalSpreadingActivation.decay_function_exponential_with_decay_factor(decay_factor)

    @staticmethod
    def decay_function_gaussian_with_sd(sd, height_coef=1, centre=0) -> callable:
        assert height_coef > 0
        assert sd > 0

        def decay_function(age, original_activation):
            height = original_activation * height_coef
            return height * exp((-1) * (((age - centre) ** 2) / (2 * sd * sd)))
        return decay_function

    def iter_impulses(self):
        for v1, v2, e_data in self.graph.edges(data=True):
            for impulse in e_data[EdgeDataKey.IMPULSES]:
                yield impulse

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
        existing_activation = self.graph.nodes[n][NodeDataKey.CHARGE].activation
        new_activation = existing_activation + activation
        if self.activation_cap is not None:
            new_activation = min(new_activation, self.activation_cap)
        new_charge = Charge(n, new_activation, self.node_decay_function)
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

            # If another impulse was released from this node to the same target this tick, it gets replaced by this one
            for i in e_data[EdgeDataKey.IMPULSES]:
                if i.age == 0 and i.target_node == target_node and i.source_node == n:
                    i.expire()

            new_impulse = Impulse(source_node=n,
                                  target_node=target_node,
                                  initial_activation=e_data[EdgeDataKey.WEIGHT] * new_charge.activation,
                                  decay_function=self.edge_decay_function)

            e_data[EdgeDataKey.IMPULSES].append(new_impulse)

    def __str__(self):
        string_builder = ""
        string_builder += "Nodes:\n"
        for node, n_data in self.graph.nodes(data=True):
            # Skip unactivated nodes
            if n_data[NodeDataKey.CHARGE].activation == 0:
                continue
            string_builder += f"\t{str(n_data[NodeDataKey.CHARGE])}\n"
        string_builder += "Edges:\n"
        for n1, n2, e_data in self.graph.edges(data=True):
            impulse_list = [str(i) for i in e_data[EdgeDataKey.IMPULSES]]
            # Skip empty edges
            if len(impulse_list) == 0:
                continue
            string_builder += f"\t{n1}–{n2}:\n"
            for impulse in impulse_list:
                string_builder += f"\t\t{impulse}\n"
        return string_builder

    def activation_snapshot(self) -> Dict:
        """
        Returns a dictionary of activations for each node.
        """
        return {
            n: n_data[NodeDataKey.CHARGE].activation
            for n, n_data in self.graph.nodes(data=True)
        }

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
            pyplot.annotate(frame_label,
                            horizontalalignment='left', verticalalignment='bottom',
                            xy=(1, 0), xycoords='axes fraction')

        # Style figure
        pyplot.axis('off')

        # Save or show graph
        if pdf is not None:
            pdf.savefig()
            pyplot.close()
        else:
            pyplot.show()

        return pos
