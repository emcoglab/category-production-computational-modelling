from numpy import array

from model.graph import Graph
from model.temporal_spatial_propagation import TemporalSpatialPropagation
from model.temporal_spreading_activation import TemporalSpreadingActivation
from model.utils.maths import make_decay_function_exponential_with_decay_factor


def test_worked_example_unweighted_node_values():
    distance_matrix = array([
        [.0, .3, .6],  # Lion
        [.3, .0, .4],  # Tiger
        [.6, .4, .0],  # Stripes
    ])
    graph = Graph.from_distance_matrix(
        distance_matrix=distance_matrix,
        length_granularity=10,
    )
    tsa = TemporalSpreadingActivation(
        graph=graph,
        idx2label={0: "lion", 1: "tiger", 2: "stripes"},
        impulse_pruning_threshold=.1,
        firing_threshold=0.3,
        node_decay_function=make_decay_function_exponential_with_decay_factor(decay_factor=0.9),
        edge_decay_function=make_decay_function_exponential_with_decay_factor(decay_factor=0.9),
    )

    tsa.activate_item_with_label("lion", 1)

    for i in range(1, 16):
        tsa.tick()
        print(tsa.clock)
        print(tsa.activation_of_item_with_label("lion"), 0.4118)
        print(tsa.activation_of_item_with_label("tiger"), 0.6177)
        print(tsa.activation_of_item_with_label("stripes"), 0.2059)
        print()


if __name__ == '__main__':
    test_worked_example_unweighted_node_values()
