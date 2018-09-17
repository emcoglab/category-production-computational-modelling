from numpy import array

from model.graph import Graph
from model.temporal_spreading_activation import TemporalSpreadingActivation, decay_function_exponential_with_half_life, \
    decay_function_gaussian_with_sd_fraction

distance_matrix = array([
    [.0, .5],
    [.5, .0]
])
sd_frac = 0.42
granularity = 390
tsa_390 = TemporalSpreadingActivation(
    graph=Graph.from_distance_matrix(
        distance_matrix=distance_matrix,
        length_granularity=granularity,
    ),
    impulse_pruning_threshold=0,
    firing_threshold=0.5,
    conscious_access_threshold=0.5,
    node_decay_function=decay_function_exponential_with_half_life(50),
    edge_decay_function=decay_function_gaussian_with_sd_fraction(sd_frac, granularity),
    node_relabelling_dictionary=dict()
)
granularity = 1000
tsa_1000 = TemporalSpreadingActivation(
    graph=Graph.from_distance_matrix(
        distance_matrix=distance_matrix,
        length_granularity=granularity,
    ),
    impulse_pruning_threshold=0,
    firing_threshold=0.5,
    conscious_access_threshold=0.5,
    node_decay_function=decay_function_exponential_with_half_life(50),
    edge_decay_function=decay_function_gaussian_with_sd_fraction(sd_frac, granularity),
    node_relabelling_dictionary=dict()
)

tsa_390.activate_node(n=0, activation=1.0)
tsa_1000.activate_node(n=0, activation=1.0)

print(set(float(v) for v in tsa_390.impulses_headed_for(1).values()))
print(set(float(v) for v in tsa_1000.impulses_headed_for(1).values()))

print("Done")
