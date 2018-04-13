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
from typing import Set

from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances

from corpus_analysis.core.corpus.indexing import TokenIndexDictionary, FreqDist
from corpus_analysis.core.model.count import LogCoOccurrenceCountModel
from corpus_analysis.preferences.preferences import Preferences as CorpusPreferences
from temporal_spreading_activation import TemporalSpreadingActivation

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main():

    box_root = "/Users/caiwingfield/Box Sync/WIP/"
    csv_location = path.join(box_root, "activated node counts.csv")

    logger.info("Training distributional model")

    corpus_meta = CorpusPreferences.source_corpus_metas[0]
    freq_dist = FreqDist.load(corpus_meta.freq_dist_path)
    distributional_model_index = TokenIndexDictionary.from_freqdist(freq_dist)
    distributional_model = LogCoOccurrenceCountModel(corpus_meta, window_radius=1, token_indices=distributional_model_index)
    distributional_model.train(memory_map=True)

    # Words 101–400
    filtered_words = get_word_list(freq_dist, top_n=400) - get_word_list(freq_dist, top_n=100)  # school is in the top 300

    filtered_indices = sorted([distributional_model_index.token2id[w] for w in filtered_words])
    # TODO: explain what these dictionaries are
    ldm_to_matrix, matrix_to_ldm = filtering_dictionaries([distributional_model_index.token2id[w] for w in filtered_words])

    logger.info("Constructing weight matrix")

    # First coordinate (row index) points to TODO: what?
    embedding_matrix = distributional_model.matrix.tocsr()[filtered_indices, :].copy()

    # Convert to distance matrix
    distance_matrix = pairwise_distances(embedding_matrix, metric="cosine", n_jobs=-1)

    logger.info("Building graph")

    graph = TemporalSpreadingActivation.graph_from_distance_matrix(
        distance_matrix=distance_matrix.copy(),
        weighted_graph=False,
        length_granularity=100,
        weight_factor=20,
        # Relabel nodes with words rather than indices
        relabelling_dict=build_relabelling_dictionary(ldm_to_matrix, distributional_model_index))

    n_ticks = 100
    n_runs = 10

    # Run multiple times with different parameters

    results = []

    for run in range(n_runs):

        initial_word = random.choice(tuple(filtered_words))

        for threshold in [0.1, 0.2, 0.3]:
            for node_decay_factor in [0.99, 0.9, 0.8]:
                for edge_decay_sd in [10, 15, 20]:

                    logger.info(f"")
                    logger.info(f"\t(run {run})")
                    logger.info(f"Setting up spreading output")
                    logger.info(f"Using values: θ={threshold}, δ={node_decay_factor}, sd={edge_decay_sd}")

                    tsa = TemporalSpreadingActivation(
                        graph=graph,
                        threshold=threshold,
                        node_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_decay_factor(
                            decay_factor=node_decay_factor),
                        edge_decay_function=TemporalSpreadingActivation.decay_function_gaussian_with_sd(
                            sd=edge_decay_sd))

                    logger.info(f"Initial node {initial_word}")
                    tsa.activate_node(initial_word, 1)

                    results.append(
                        [run, 0, tsa.n_suprathreshold_nodes(), threshold, node_decay_factor, edge_decay_sd])

                    logger.info("Running spreading output")
                    for tick in range(1, n_ticks):
                        logger.info(f"Clock = {tick}")
                        tsa.tick()
                        results.append(
                            [run, tick, tsa.n_suprathreshold_nodes(), threshold, node_decay_factor, edge_decay_sd])

    results_df = DataFrame(data=results,
                           columns=["Run", "Tick", "Activated nodes", "Threshold", "Node decay factor", "Edge decay SD"])

    results_df.to_csv(csv_location, header=True, index=False)


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


def get_word_list(freq_dist, top_n) -> Set:
    return {word for word, _ in freq_dist.most_common(top_n)}


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
