from os import path

from pandas import DataFrame

from ldm.core.corpus.indexing import FreqDist, TokenIndex
from ldm.core.model.count import LogCoOccurrenceCountModel
from ldm.core.utils.maths import DistanceType
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.graph import Graph
from preferences import Preferences

length_factor = 1_000
n_words = 1_000
prune_percent = 50

corpus = CorpusPreferences.source_corpus_metas.bbc
distance_type = DistanceType.cosine
freq_dist = FreqDist.load(corpus.freq_dist_path)
token_index = TokenIndex.from_freqdist_ranks(freq_dist)
distributional_model = LogCoOccurrenceCountModel(corpus, window_radius=5, freq_dist=freq_dist)

graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"
quantile_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor} edge length quantiles.csv"

quantile_data = DataFrame.from_csv(path.join(Preferences.graphs_dir, quantile_file_name), header=0, index_col=None)
pruning_length = quantile_data[
    # Use 1 - so that smallest top quantiles get converted to longest edges
    quantile_data["Top quantile"] == 1 - (prune_percent / 100)
    ]["Pruning length"].iloc[0]

graph = Graph.load_from_edgelist(file_path=path.join(Preferences.graphs_dir, graph_file_name),
                                 ignore_edges_longer_than=pruning_length,
                                 keep_at_least_n_edges=Preferences.min_edges_per_node)

print("Done")
