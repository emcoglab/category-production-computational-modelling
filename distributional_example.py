"""
===========================
Runs the example of spreading activation on a language model.
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
import random
import sys
from os import path

from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances

from ldm.core.corpus.indexing import FreqDistIndex
from ldm.core.model.count import LogCoOccurrenceCountModel
from ldm.core.utils.logging import log_message, date_format
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.temporal_spreading_activation import TemporalSpreadingActivation, graph_from_distance_matrix, \
    decay_function_exponential_with_decay_factor, decay_function_gaussian_with_sd


logger = logging.getLogger()


def main():

    box_root = "/Users/caiwingfield/Box Sync/WIP/"
    csv_location = path.join(box_root, "activated node counts.csv")

    logger.info("Training distributional model")

    corpus_meta = CorpusPreferences.source_corpus_metas[0]
    freq_dist = FreqDistIndex.load(corpus_meta.freq_dist_path)
    distributional_model = LogCoOccurrenceCountModel(corpus_meta, window_radius=1, freq_dist=freq_dist)
    distributional_model.train(memory_map=True)

    # Words 101–400
    filtered_words = set(freq_dist.most_common_tokens(400)) - set(freq_dist.most_common_tokens(100))  # school is in the top 300

    filtered_indices = sorted([freq_dist.token2id[w] for w in filtered_words])
    # These dictionaries translate between matrix-row/column indices (after filtering) and token indices within the LDM.
    ldm_to_matrix, matrix_to_ldm = filtering_dictionaries([freq_dist.token2id[w] for w in filtered_words])

    logger.info("Constructing weight matrix")

    # First coordinate (row index) points to target words
    embedding_matrix = distributional_model.matrix.tocsr()[filtered_indices, :]

    # Convert to distance matrix
    distance_matrix = pairwise_distances(embedding_matrix, metric="cosine", n_jobs=-1)

    logger.info("Building graph")

    graph = graph_from_distance_matrix(
        distance_matrix=distance_matrix,
        weighted_graph=False,
        length_granularity=100,
        weight_factor=20)

    n_ticks = 200
    n_runs = 5
    bailout = 200

    # Run multiple times with different parameters

    results_df = DataFrame()

    for run in range(n_runs):

        initial_word = random.choice(tuple(filtered_words))

        for activation_threshold in [0.4, 0.6, 0.8, 1.0]:
            for node_decay_factor in [0.85, 0.9, 0.99]:
                for edge_decay_sd in [10, 15, 20]:

                    logger.info(f"")
                    logger.info(f"\t(run {run})")
                    logger.info(f"Setting up spreading output")
                    logger.info(f"Using values: θ={activation_threshold}, δ={node_decay_factor}, sd={edge_decay_sd}")

                    tsa = TemporalSpreadingActivation(
                        graph=graph,
                        activation_threshold=activation_threshold,
                        node_relabelling_dictionary=build_relabelling_dictionary(ldm_to_matrix, distributional_model_index),
                        node_decay_function=decay_function_exponential_with_decay_factor(
                            decay_factor=node_decay_factor),
                        edge_decay_function=decay_function_gaussian_with_sd(
                            sd=edge_decay_sd))

                    logger.info(f"Initial node {initial_word}")
                    tsa.activate_node(initial_word, 1)

                    logger.info("Running spreading output")
                    for tick in range(1, n_ticks):
                        logger.info(f"Clock = {tick}")
                        tsa.tick()

                        # Break early if we've got a probable explosion
                        # if tsa.n_suprathreshold_nodes > bailout:
                        #     logger.warning(f"{tsa.n_suprathreshold_nodes} nodes active... Bailout!!")
                        #     break

                    # Prepare results
                    results_these_params = tsa.activation_history
                    results_these_params["Activation threshold"] = activation_threshold
                    results_these_params["Node decay factor"] = node_decay_factor
                    results_these_params["Edge decay SD"] = edge_decay_sd
                    results_these_params["Run"] = run

                    results_df = results_df.append(results_these_params)

    results_df.to_csv(csv_location, header=True, index=False)


def filtering_dictionaries(filtered_indices):
    ldm_to_matrix_index = {}
    matrix_to_ldm_index = {}
    for i, filtered_index in enumerate(filtered_indices):
        ldm_to_matrix_index[filtered_index] = i
        matrix_to_ldm_index[i] = filtered_index
    return ldm_to_matrix_index, matrix_to_ldm_index


def build_relabelling_dictionary(ldm_to_matrix, freq_dist: FreqDistIndex):
    relabelling_dictinoary = dict()
    for token_index, matrix_index in ldm_to_matrix.items():
        relabelling_dictinoary[matrix_index] = freq_dist.id2token[token_index]
    return relabelling_dictinoary


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
