"""
===========================
Model responses to Briony's category production data.
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


class BrionyCategoryProductionData(object):

    _box_root = "/Users/cai/Box Sync/LANGBOOT Project/Experiments/Phase 1 - Categorisation/Experiment 1.4 - Category production/Data & Analysis"
    _csv_path = path.join(_box_root, "Data for proximity measures/Data for linguistic proximity measure BrEng.csv")

    def __init__(self):
        self.data = DataFrame()
        self._load()

        # TODO: Build vocab lists
        

    def _load(self):
        self.data = DataFrame.from_csv(BrionyCategoryProductionData._csv_path, index_col=0, header=0)


def main():

    # SA parameters
    granularity = 100
    node_decay_factors = [0.99, 0.9, 0.8]
    edge_decay_sd_fracs = [0.1, 0.15, 0.2]
    thresholds = [0.1, 0.2, 0.3]

    n_ticks = 200
    bailout = 1000

    # Where to save data
    box_root = "/Users/caiwingfield/Box Sync/WIP/"
    csv_location = path.join(box_root, "activated node counts.csv")

    logger.info("Training distributional model")
    corpus_meta = CorpusPreferences.source_corpus_metas[0]
    freq_dist = FreqDist.load(corpus_meta.freq_dist_path)
    distributional_model_index = TokenIndexDictionary.from_freqdist(freq_dist)
    distributional_model = LogCoOccurrenceCountModel(corpus_meta, window_radius=1, token_indices=distributional_model_index)
    distributional_model.train(memory_map=True)

    # Use most frequent words, excluding the *very* most frequent ones
    top_n_words = 3000
    top_n_function_words = 10
    filtered_words = get_word_list(freq_dist, top_n=top_n_words)
    function_words = get_word_list(freq_dist, top_n=top_n_function_words)
    filtered_words -= function_words
    logger.info(f"Building graph from top {top_n_words} words, except these: {', '.join(function_words)}")
    logger.info(f"This will include a total of {len(filtered_words)} words, "
                f"and a maximum of {int(0.5 * len(filtered_words) * (len(filtered_words) - 1))} connections")

    # Build index-lookup dictionaries
    filtered_indices = sorted([distributional_model_index.token2id[w] for w in filtered_words])
    ldm_to_matrix, matrix_to_ldm = filtering_dictionaries([distributional_model_index.token2id[w] for w in filtered_words])

    # Build distance matrix
    logger.info("Constructing distance matrix")
    embedding_matrix = distributional_model.matrix.tocsr()[filtered_indices, :].copy()
    distance_matrix = pairwise_distances(embedding_matrix, metric="cosine", n_jobs=-1)

    # Build graph
    logger.info("Building graph")
    graph = TemporalSpreadingActivation.graph_from_distance_matrix(
        distance_matrix=distance_matrix.copy(),
        weighted_graph=False,
        length_granularity=granularity,
        # Relabel nodes with words rather than indices
        relabelling_dict=build_relabelling_dictionary(ldm_to_matrix, distributional_model_index))

    # Run multiple times with different parameters

    results = []

    initial_word = random.choice(tuple(filtered_words))

    for threshold in thresholds:
        for node_decay_factor in node_decay_factors:
            for edge_decay_sd_frac in edge_decay_sd_fracs:

                logger.info(f"")
                logger.info(f"Setting up spreading output")
                logger.info(f"Using values: θ={threshold}, δ={node_decay_factor}, sd_frac={edge_decay_sd_fracs}")

                tsa = TemporalSpreadingActivation(
                    graph=graph,
                    threshold=threshold,
                    node_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_decay_factor(
                        decay_factor=node_decay_factor),
                    edge_decay_function=TemporalSpreadingActivation.decay_function_gaussian_with_sd_fraction(
                        sd_frac=edge_decay_sd_frac, granularity=granularity))

                logger.info(f"Initial node {initial_word}")
                tsa.activate_node(initial_word, 1)

                # TODO: This could be stored internal to the TSA object.
                results.append(
                    [0, tsa.n_suprathreshold_nodes, threshold, node_decay_factor, edge_decay_sd_fracs])

                logger.info("Running spreading output")
                for tick in range(1, n_ticks):
                    logger.info(f"Clock = {tick}")
                    tsa.tick()
                    results.append(
                        [tick, tsa.n_suprathreshold_nodes, threshold, node_decay_factor, edge_decay_sd_fracs])

                    # Break early if we've got a probable explosion
                    if tsa.n_suprathreshold_nodes > bailout:
                        break

    results_df = DataFrame(data=results,
                           columns=["Tick", "Activated nodes", "Threshold", "Node decay factor", "Edge decay SD"])

    results_df.to_csv(csv_location, header=True, index=False)


def filtering_dictionaries(filtered_indices):
    # A lookup dictionary which converts the index of a word in the LDM to its index in the specific filtered matrix
    ldm_to_matrix_index = {}
    # A lookup dictionary which converts the index of a word in the specific filtered distance matrix to its index in the LDM
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
