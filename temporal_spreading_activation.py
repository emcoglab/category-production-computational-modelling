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
import time
from typing import List, Dict, Set

from networkx import Graph, from_numpy_matrix, relabel_nodes, selfloop_edges
from numpy import exp, ndarray, ones_like, ceil, float_power
from pandas import DataFrame

from utils import set_partition

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class NodeDataKey(object):
    """Column names for node data."""
    CHARGE         = "charge"


class EdgeDataKey(object):
    """Column names for edge data."""
    WEIGHT         = "weight"
    LENGTH         = "length"


class HistoryDataKey(object):
    """Column names for results history."""
    CLOCK          = "clock"
    N_ACTIVATED    = "n_activated"
    JUST_ACTIVATED = "just_activated"


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

    def __init__(self,
                 source_node,
                 target_node,
                 time_at_creation: int,
                 time_at_destination: int,
                 initial_activation: float,
                 final_activation: float):
        self.source_node = source_node
        self.target_node = target_node
        self.time_at_creation: int = time_at_creation
        self.time_at_destination: int = time_at_destination
        self.initial_activation: float = initial_activation
        self.activation_at_destination: float = final_activation

    def __str__(self):
        return f"{self.source_node} → {self.target_node}: {self.activation_at_destination:.4g} @ {str(self.time_at_destination)}"

    def age_at_time(self, t: int):
        """
        The integer age of this impulse at specified time, or None if impulse will not exist at that time.
        """
        if t > self.time_at_destination:
            return None
        if t < self.time_at_creation:
            return None
        return t - self.time_at_creation

    # region Implement Hashable

    def __key(self):
        return (
            self.source_node,
            self.target_node,
            self.time_at_creation,
            self.time_at_destination,
            self.initial_activation,
            self.activation_at_destination
        )

    def __eq__(self, other):
        return type(self) == type(other) and self.__key() == other.__key()

    def __hash__(self):
        return hash(self.__key())

    # endregion


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
        ¡ Calling this constructor or modifying this object WILL modify the underlying graph !
        Should be an undirected weighted graph with the following data:
            On Nodes:
                (no data required)
            On Edges:
                weight
                length
        To this the following data fields will be added by the constructor:
            On Nodes:
                charge
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
        self.threshold = threshold  # Use `< self.threshold` and `>= self.threshold` to test for below/above threshold.
        self.activation_cap = activation_cap

        # These decay functions should be stateless, and convert an original activation and an age into a current
        # activation.
        # Each should be of the form (age, initial_activation) ↦ current_activation
        self.node_decay_function: callable = node_decay_function
        self.edge_decay_function: callable = edge_decay_function

        # Underlying graph: weighted, undirected
        self.graph: Graph = graph

        # Graph data:
        self.impulses: Set = set()

        # Zero-indexed tick counter.
        self.clock: int = 0

        # Stores a clock-indexed list of dictionaries for the relevant data
        # Backing for self.activation_history, which converts this into a pandas.DataFrame
        self._activation_history: List[Dict] = []

        # Nodes which received an activation this tick
        self.nodes_activated_this_tick: Set = set()

        # Initialise graph
        self.reset()

    @staticmethod
    def graph_from_distance_matrix(distance_matrix: ndarray,
                                   length_granularity: int,
                                   weighted_graph: bool,
                                   weight_factor: float = 1,
                                   relabelling_dict=None,
                                   prune_connections_longer_than: int = None) -> Graph:
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
        :param weighted_graph:
        Whether to use weights on the edges.
        If True, distances will be converted to weights using x ↦ 1-x.
        If False, all edges get the same weight.
        :param weight_factor:
        (Default 1.)
        If `weighted_graph` is True, this factor is multiplied by all weights.
        If `weighted_graph` is false, this fixed weight given to each edge in the graph.
        :param relabelling_dict:
        (Optional.)
        If provided and not None: A dictionary which maps the integer indices of nodes to
        their desired labels.
        :param prune_connections_longer_than:
        (Optional.) If provided and not None: Any connections with lengths (strictly) longer than this will be severed.
        :return:
        A Graph of the correct format.
        """

        length_matrix = ceil(distance_matrix * length_granularity)

        if weighted_graph:
            weight_matrix = weight_factor * (ones_like(distance_matrix) - distance_matrix)
        else:
            weight_matrix = weight_factor * ones_like(distance_matrix)

        graph = from_numpy_matrix(weight_matrix)

        # Converting from a distance matrix creates self-loop edges, which we have to remove
        graph.remove_edges_from(selfloop_edges(graph))

        # Add lengths to graph data
        for n1, n2, e_data in graph.edges(data=True):
            e_data[EdgeDataKey.LENGTH] = length_matrix[n1][n2]

        # Prune long connections
        if prune_connections_longer_than is not None:
            long_edges = [
                (n1, n2)
                for n1, n2, e_data in graph.edges(data=True)
                if e_data[EdgeDataKey.LENGTH] > prune_connections_longer_than
            ]
            graph.remove_edges_from(long_edges)

        # Relabel nodes if dictionary was provided
        if relabelling_dict is not None:
            graph = relabel_nodes(graph, relabelling_dict, copy=False)

        return graph

    @property
    def activation_history(self) -> DataFrame:
        return DataFrame(self._activation_history)

    @property
    def node_labels(self) -> List:
        return [node
                for node in self.graph.nodes]

    @property
    def charges(self) -> List:
        return [n_data[NodeDataKey.CHARGE]
                for n, n_data in self.graph.nodes(data=True)]

    @property
    def edge_labels(self) -> List:
        return [edge
                for edge in self.graph.edges]

    @property
    def n_suprathreshold_nodes(self) -> int:
        """The number of nodes which are above the threshold."""
        return len([
            charge for charge in self.charges
            if charge.activation >= self.threshold
        ])

    def activation_of_node(self, n) -> float:
        """Returns the current activation of a node."""
        return self.graph.nodes(data=True)[n][NodeDataKey.CHARGE].activation

    def activate_node(self, n, activation: float):
        """Activates a node."""

        self.nodes_activated_this_tick.add(n)

        # Accumulate activation
        existing_activation = self.graph.nodes[n][NodeDataKey.CHARGE].activation
        new_activation = existing_activation + activation
        if self.activation_cap is not None:
            new_activation = min(new_activation, self.activation_cap)
        new_charge = Charge(n, new_activation, self.node_decay_function)
        self.graph.nodes[n][NodeDataKey.CHARGE] = new_charge

        source_node = n

        # Rebroadcast

        # For each incident edge
        for n1, n2, e_data in self.graph.edges(source_node, data=True):
            if source_node == n1:
                target_node = n2
            elif source_node == n2:
                target_node = n1
            else:
                raise ValueError()

            edge_length = e_data[EdgeDataKey.LENGTH]

            # We pre-compute the impulses now rather than decaying them over time.
            # Intermediate activates can be computed for display purposes if necessary.
            initial_activation = e_data[EdgeDataKey.WEIGHT] * new_charge.activation
            final_activation = self.edge_decay_function(edge_length, initial_activation)

            # Only create impulses that will reach the destination before decaying below threshold
            if final_activation < self.threshold:
                continue

            new_impulse = Impulse(
                source_node=source_node, target_node=target_node,
                time_at_creation=self.clock, time_at_destination=self.clock + edge_length,
                initial_activation=initial_activation,
                final_activation=final_activation
            )

            # If another impulse was released from this node to the same target this tick, it should replaced by this
            # one, so remove it
            existing_impulses = set(impulse
                                    for impulse in self.impulses
                                    if impulse.target_node == target_node
                                    and impulse.source_node == source_node
                                    and impulse.time_at_creation == self.clock)
            if len(existing_impulses) > 0:
                self.impulses -= existing_impulses

            self.impulses.add(new_impulse)

        self._update_history()

    def _decay_nodes(self):
        """Decays activation in each node."""
        for n, n_data in self.graph.nodes(data=True):
            charge = n_data[NodeDataKey.CHARGE]
            if charge is not None:
                charge.decay()

    def _propagate_impulses(self):
        """Propagates impulses along connections."""

        # "Propagation" happens by just incrementing the global clock.

        # But we have to check if any impulses have reached their destination.
        impulses_at_destination = set(i for i in self.impulses if i.time_at_destination == self.clock)

        if len(impulses_at_destination) > 0:

            # Remove those that have reached the destination
            self.impulses -= impulses_at_destination

            # And have them activated their target nodes
            for impulse in impulses_at_destination:
                self.activate_node(impulse.target_node, impulse.activation_at_destination)

    def tick(self):
        """Performs the spreading activation algorithm for one tick of the clock."""
        self.clock += 1

        # Empty the list of activated nodes, it will be refilled as nodes become activated
        self.nodes_activated_this_tick = set()

        self._decay_nodes()
        self._propagate_impulses()

        self._update_history()

    def tick_timed(self) -> float:
        """
        Performs the spreading activation algorithm for one tick of the clock.
        Returns the number of seconds which elapsed computing this tick.
        """
        start = time.time()
        self.tick()
        duration = time.time() - start

        return duration

    def reset(self):
        """Reset SA graph so it looks as if it's just been built."""
        # Set all node activations to zero
        for n, n_data in self.graph.nodes(data=True):
            n_data[NodeDataKey.CHARGE] = Charge(n, 0, self.node_decay_function)
        # Delete all activations
        self.impulses = set()
        self.nodes_activated_this_tick = set()
        # Reset clock and history
        self.clock = 0
        self._activation_history = []
        self._update_history()

    def _update_history(self):
        entry = {
            HistoryDataKey.CLOCK:          self.clock,
            HistoryDataKey.N_ACTIVATED:    self.n_suprathreshold_nodes,
            # Include activations with just-activated nodes, and sort descending
            HistoryDataKey.JUST_ACTIVATED: sorted(
                [(n, self.activation_of_node(n))
                 for n in self.nodes_activated_this_tick],
                # Sort by activation, descending
                key=lambda n_a_pair: n_a_pair[1], reverse=True),
        }
        # Check if a history entry exists for this clock time
        # Use a > check because when the clock is 0, and there IS an entry, the len will be 1 > 0
        if len(self._activation_history) > self.clock:
            # Entry exists
            self._activation_history[self.clock] = entry
        else:
            # Entry doesn't exist yet
            self._activation_history.append(entry)

    def __str__(self):
        string_builder = f"CLOCK = {self.clock}\n"
        string_builder += "Nodes:\n"
        for node, n_data in self.graph.nodes(data=True):
            # Skip unactivated nodes
            if n_data[NodeDataKey.CHARGE].activation == 0:
                continue
            string_builder += f"\t{str(n_data[NodeDataKey.CHARGE])}\n"
        string_builder += "Edges:\n"
        for n1, n2 in self.graph.edges():
            impulse_list = [str(i)
                            for i in self.impulses
                            if {i.source_node, i.target_node} == {n1, n2}]
            # Skip empty edges
            if len(impulse_list) == 0:
                continue
            string_builder += f"\t{n1}–{n2}:\n"
            for impulse in impulse_list:
                string_builder += f"\t\t{impulse}\n"
        return string_builder

    def log_graph(self):
        [logger.info(f"{line}") for line in str(self).strip().split('\n')]

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
        """Gaussian decay with sd specifying the number of ticks."""
        assert height_coef > 0
        assert sd > 0

        def decay_function(age, original_activation):
            height = original_activation * height_coef
            return height * exp((-1) * (((age - centre) ** 2) / (2 * sd * sd)))
        return decay_function

    @staticmethod
    def decay_function_gaussian_with_sd_fraction(sd_frac: float, granularity: int, height_coef=1, centre=0) -> callable:
        """Gaussian decay with sd as a fraction of the granularity."""
        sd = sd_frac * granularity
        return TemporalSpreadingActivation.decay_function_gaussian_with_sd(
            sd=sd,
            height_coef=height_coef,
            centre=centre)
