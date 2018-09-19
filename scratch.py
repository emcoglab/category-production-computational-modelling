import json
from os import path

from pandas import DataFrame

from ldm.core.corpus.indexing import FreqDist
from ldm.core.model.count import LogCoOccurrenceCountModel
from ldm.core.utils.maths import DistanceType
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.graph import Graph
from model.temporal_spreading_activation import TemporalSpreadingActivation, \
    decay_function_exponential_with_decay_factor, decay_function_gaussian_with_sd_fraction
from preferences import Preferences


def main():

    n_words = 1000
    length_factor = 1_000
    impulse_pruning_threshold = 0.05
    firing_threshold = 0.8
    conscious_access_threshold = 0.9
    node_decay_factor = 0.99
    edge_decay_sd_frac = 0.4
    prune_percent = 50

    corpus = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distance_type = DistanceType.cosine

    distributional_model = LogCoOccurrenceCountModel(corpus, window_radius=5, freq_dist=freq_dist)

    quantile_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor} edge length quantiles.csv"
    quantile_data = DataFrame.from_csv(path.join(Preferences.graphs_dir, quantile_file_name), header=0, index_col=None)
    pruning_length = quantile_data[
        # Use 1 - so that smallest top quantiles get converted to longest edges
        quantile_data["Top quantile"] == 1 - (prune_percent / 100)
        ]["Pruning length"].iloc[0]

    # Load node relabelling dictionary
    print(f"Loading node labels")
    with open(path.join(Preferences.graphs_dir, f"{corpus.name} {n_words} words.nodelabels"), mode="r", encoding="utf-8") as nrd_file:
        node_relabelling_dictionary_json = json.load(nrd_file)
    # TODO: this isn't a great way to do this
    node_relabelling_dictionary = dict()
    for k, v in node_relabelling_dictionary_json.items():
        node_relabelling_dictionary[int(k)] = v
    node_lookup = {v: k for k, v in node_relabelling_dictionary.items()}

    graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"

    print("Loading unpruned graph")
    unpruned_graph = Graph.load_from_edgelist_with_importance_pruning(file_path=path.join(Preferences.graphs_dir, graph_file_name))
    print("Loading pruned graph")
    pruned_graph = Graph.load_from_edgelist_with_importance_pruning(file_path=path.join(Preferences.graphs_dir, graph_file_name),
                                                                    ignore_edges_with_importance_greater_than=pruning_length,
                                                                    keep_at_least_n_edges=Preferences.min_edges_per_node)

    unpruned_tsa = TemporalSpreadingActivation(
        graph=unpruned_graph,
        node_relabelling_dictionary=node_relabelling_dictionary,
        firing_threshold=firing_threshold,
        conscious_access_threshold=conscious_access_threshold,
        impulse_pruning_threshold=impulse_pruning_threshold,
        node_decay_function=decay_function_exponential_with_decay_factor(decay_factor=node_decay_factor),
        edge_decay_function=decay_function_gaussian_with_sd_fraction(sd_frac=edge_decay_sd_frac, granularity=length_factor)
    )
    pruned_tsa = TemporalSpreadingActivation(
        graph=pruned_graph,
        node_relabelling_dictionary=node_relabelling_dictionary,
        firing_threshold=firing_threshold,
        conscious_access_threshold=conscious_access_threshold,
        impulse_pruning_threshold=impulse_pruning_threshold,
        node_decay_function=decay_function_exponential_with_decay_factor(decay_factor=node_decay_factor),
        edge_decay_function=decay_function_gaussian_with_sd_fraction(sd_frac=edge_decay_sd_frac,
                                                                     granularity=length_factor)
    )

    unpruned_tsa.activate_node_with_label("building", 1)
    pruned_tsa.activate_node_with_label("building", 1)

    print("Running TSA")


if __name__ == '__main__':
    main()
    print("Done")
