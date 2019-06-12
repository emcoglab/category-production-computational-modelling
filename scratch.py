from numpy import array

from model.events import ItemActivatedEvent
from model.graph import Graph
from model.temporal_spatial_propagation import TemporalSpatialPropagation
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
    tsp = TemporalSpatialPropagation(
        underlying_graph=graph,
        idx2label={0: "lion", 1: "tiger", 2: "stripes"},
        node_decay_function=make_decay_function_exponential_with_decay_factor(decay_factor=0.9),
    )

    e = tsp.activate_item_with_label("lion", 1)

    print(e == ItemActivatedEvent(time=0, item=0, activation=1.0, fired=True))

    for t in range(1, 17):
        es = tsp.tick()
        print(tsp.clock)
        print(es)


if __name__ == '__main__':
    test_worked_example_unweighted_node_values()
