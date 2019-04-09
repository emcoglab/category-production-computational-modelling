"""
===========================
Example of tandem model function
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""
from logging import getLogger
from os import path

from cli.lookups import get_corpus_from_name, get_model_from_params
from ldm.corpus.indexing import FreqDist, TokenIndex
from ldm.model.base import DistributionalSemanticModel
from ldm.utils.maths import DistanceType
from model.graph import Graph
from model.temporal_spatial_expansion import TemporalSpatialExpansion
from model.temporal_spreading_activation import TemporalSpreadingActivation, load_labels
from model.utils.indexing import list_index_dictionaries
from model.utils.math import decay_function_exponential_with_decay_factor, decay_function_gaussian_with_sd
from preferences import Preferences

logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main(n_words, corpus_name, model_name, radius, length_factor, firing_threshold, node_decay_factor,
         edge_decay_sd_factor, impulse_pruning_threshold, run_for_ticks, bailout):

    corpus = get_corpus_from_name(corpus_name)
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    token_index = TokenIndex.from_freqdist_ranks(freq_dist)
    distributional_model: DistributionalSemanticModel = get_model_from_params(corpus, freq_dist, model_name, radius)

    filtered_words = set(freq_dist.most_common_tokens(n_words))
    filtered_ldm_ids = sorted([token_index.token2id[w] for w in filtered_words])

    # These dictionaries translate between matrix-row/column indices (after filtering) and token indices within the LDM.
    _, matrix_to_ldm = list_index_dictionaries(filtered_ldm_ids)

    # Load distance matrix
    graph_file_name = f"{distributional_model.name} {n_words} words length {length_factor}.edgelist"

    logger.info(f"Loading graph from {graph_file_name}")

    # Load graph
    graph = Graph.load_from_edgelist(file_path=path.join(Preferences.graphs_dir, graph_file_name))

    linguistic_component: TemporalSpreadingActivation = TemporalSpreadingActivation(
        graph=graph,
        item_labelling_dictionary=load_labels(corpus, n_words),
        firing_threshold=firing_threshold,
        impulse_pruning_threshold=impulse_pruning_threshold,
        node_decay_function=decay_function_exponential_with_decay_factor(
            decay_factor=node_decay_factor),
        edge_decay_function=decay_function_gaussian_with_sd(
            sd=edge_decay_sd_factor*length_factor))
    sensorimotor_component: TemporalSpatialExpansion = TemporalSpatialExpansion(
        points_in_space=,
        item_labelling_dictionary=,
        expansion_rate=,
        max_radius=,
        distance_type=DistanceType.cosine,
        decay_median=,
        decay_shape=,
        decay_threshold=,
    )


if __name__ == '__main__':

    main(n_words=3_000,
         corpus_name="bbc",
         model_name="ppmi_ngram",
         radius=5,
         length_factor=10,
         firing_threshold=0.5,
         node_decay_factor=0.99,
         edge_decay_sd_factor=25,
         impulse_pruning_threshold=0.05,
         run_for_ticks=1_000,
         bailout=1_000)
    logger.info("Done!")
