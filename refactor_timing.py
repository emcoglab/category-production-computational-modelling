"""
===========================
A simple example to run to test timings with different model implementations.
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
import sys
import time
from os import path

from sklearn.metrics.pairwise import pairwise_distances

from corpus_analysis.core.corpus.indexing import TokenIndexDictionary, FreqDist
from corpus_analysis.core.model.count import ConditionalProbabilityModel
from corpus_analysis.preferences.preferences import Preferences as CorpusPreferences
from model.temporal_spreading_activation import TemporalSpreadingActivation

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main():

    # Strings describing what's going on to keep track of results.
    # There's probably a better way to do this lol.
    current_refactor_state = "charge objects stored in graph node data; impulse objects stored in set"
    machine = "iMac"

    # Parameters
    top_n_words = 3_000
    n_ticks = 100
    initial_word = "school"
    granularity = 100
    node_decay_rate = 0.9
    edge_decay_sd = 15
    activation_threshold = 0.75

    box_root = "/Users/caiwingfield/Box Sync/WIP/"
    csv_location = path.join(box_root, "refactor_activations.csv")

    logger.info("Training distributional model")
    corpus_meta = CorpusPreferences.source_corpus_metas[1]  # 1 = bbc
    freq_dist = FreqDist.load(corpus_meta.freq_dist_path)
    distributional_model_index = TokenIndexDictionary.from_freqdist(freq_dist)
    distributional_model = ConditionalProbabilityModel(corpus_meta, window_radius=1, token_indices=distributional_model_index, freq_dist=freq_dist)
    distributional_model.train(memory_map=True)

    # Work with subset of words
    filtered_words = freq_dist.most_common_tokens(top_n_words)

    # Build distance matrix for subset of words
    logger.info("Constructing distance matrix")
    t_before_distance_matrix = time.monotonic()
    filtered_indices = sorted([distributional_model_index.token2id[w] for w in filtered_words])
    # These dictionaries translate between matrix-row/column indices (after filtering) and token indices within the LDM.
    ldm_to_matrix, matrix_to_ldm = filtering_dictionaries([distributional_model_index.token2id[w] for w in filtered_words])
    # First coordinate (row index) points to target words
    embedding_matrix = distributional_model.matrix.tocsr()[filtered_indices, :]
    distance_matrix = pairwise_distances(embedding_matrix, metric="cosine", n_jobs=-1)
    duration_distance_matrix = time.monotonic() - t_before_distance_matrix

    # Build the graph
    logger.info("Building graph")
    t_before_graph = time.monotonic()
    graph = TemporalSpreadingActivation.graph_from_distance_matrix(
        distance_matrix=distance_matrix,
        weighted_graph=False,
        length_granularity=granularity,
        # Relabel nodes with words rather than indices
        relabelling_dict=build_relabelling_dictionary(ldm_to_matrix, distributional_model_index))
    duration_graph = time.monotonic() - t_before_graph

    # Set up spreading activation model
    logger.info(f"Building SA model with values: θ={activation_threshold}, δ={node_decay_rate}, sd={edge_decay_sd}")
    t_before_sa_build = time.monotonic()
    tsa = TemporalSpreadingActivation(
        graph=graph,
        activation_threshold=activation_threshold,
        node_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_decay_factor(
            decay_factor=node_decay_rate),
        edge_decay_function=TemporalSpreadingActivation.decay_function_gaussian_with_sd(
            sd=edge_decay_sd))
    duration_sa_build = time.monotonic() - t_before_sa_build

    logger.info(f"Spreading activation from {initial_word} for {n_ticks} ticks")
    t_before_sa = time.monotonic()
    tsa.activate_node(initial_word, 1)
    for tick in range(1, n_ticks):
        logger.info(f"\tClock = {tick}")
        tsa.tick()
    duration_sa = time.monotonic() - t_before_sa

    # Prepare results
    results_df = tsa.activation_history
    results_df["Activation threshold"] = activation_threshold
    results_df["Node decay factor"] = node_decay_rate
    results_df["Edge decay SD"] = edge_decay_sd

    results_df.to_csv(csv_location, header=True, index=False)

    logger.info(f"")
    logger.info(f"Timings for \"{current_refactor_state}\" running on {machine}:")
    logger.info(f"\tBuild distance matrix: {duration_distance_matrix}")
    logger.info(f"\tBuild graph: {duration_graph}")
    logger.info(f"\tBuild SA model: {duration_sa_build}")
    logger.info(f"\tSpread activation from {initial_word} for {n_ticks} ticks: {duration_sa}")


def filtering_dictionaries(filtered_indices):
    ldm_to_matrix_index = {}
    matrix_to_ldm_index = {}
    for i, filtered_index in enumerate(filtered_indices):
        ldm_to_matrix_index[filtered_index] = i
        matrix_to_ldm_index[i] = filtered_index
    return ldm_to_matrix_index, matrix_to_ldm_index


def build_relabelling_dictionary(ldm_to_matrix, distributional_model_index: TokenIndexDictionary):
    relabelling_dictinoary = dict()
    for token_index, matrix_index in ldm_to_matrix.items():
        relabelling_dictinoary[matrix_index] = distributional_model_index.id2token[token_index]
    return relabelling_dictinoary


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
