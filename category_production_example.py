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
import sys
from os import path

from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances

from category_production.category_production import CategoryProduction
from ldm.core.corpus.indexing import FreqDist, TokenIndex
from ldm.core.model.count import LogCoOccurrenceCountModel
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.temporal_spreading_activation import TemporalSpreadingActivation, graph_from_distance_matrix, \
    decay_function_exponential_with_decay_factor, decay_function_gaussian_with_sd_fraction

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def briony_vocab_overlap(top_n_words):
    """Coverage of vocabulary in Briony's Category Production dataset."""
    # Category production words
    category_production = CategoryProduction()
    category_production_words = category_production.vocabulary_single_word

    # Frequent words in corpus
    corpus_meta = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus_meta.freq_dist_path)
    corpus_words = freq_dist.most_common_tokens(top_n_words)

    # Useful numbers to report
    n_cat_prod_words = len(category_production_words)
    n_intersection = len(set(category_production_words).intersection(set(corpus_words)))
    n_missing = len(set(category_production_words) - set(corpus_words))
    n_unused = len(set(corpus_words) - set(category_production_words))

    logger.info(f"Category production dataset contains {n_cat_prod_words} words.")
    logger.info(f"This includes {n_intersection} of the top {top_n_words} words in the {corpus_meta.name} corpus.")
    logger.info(f"{n_missing} category production words are not present.")
    logger.info(f"{n_unused} graph words are not used in the category production set.")


def briony_categories_overlap(top_n_words):
    """Coverage of individual category responses in Briony's Category Production dataset."""
    # Category production words
    category_production = CategoryProduction()
    categories = category_production.category_labels

    # Frequent words in corpus
    corpus_meta = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus_meta.freq_dist_path)
    corpus_words = set(freq_dist.most_common_tokens(top_n_words))

    logger.info(f"Top {top_n_words} words:")

    for category in categories:
        if " " in category:
            continue
        if category in corpus_words:
            responses = category_production.responses_for_category(category, single_word_only=True)
            n_present_responses = len(set(responses).intersection(set(corpus_words)))
            percent_present_responses = 100 * n_present_responses / len(responses)
            logger.info(f"{category}:\t{percent_present_responses:0.2f}%")
        else:
            logger.info(f"{category}:\t(not present)")


def main():
    # SA parameters
    granularity = 100
    node_decay_factors = [0.99, 0.9, 0.8]
    edge_decay_sd_fracs = [0.1, 0.15, 0.2]
    activation_thresholds = [0.2, 0.3, 0.4]
    impulse_pruning_threshold = 0.1

    # Use most frequent words, excluding the *very* most frequent ones
    top_n_words = 3_000
    top_n_function_words = 10

    n_ticks = 200
    # Bail if more than 50% of words activated
    explosion_bailout = int(top_n_words * 0.5)

    # Look for only the most frequent category response
    top_n_responses = 1

    # Where to save data
    box_root = "/Users/caiwingfield/Box Sync/WIP/"
    csv_location = path.join(box_root, "activated node counts.csv")

    logger.info("Training distributional model")
    corpus_meta = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus_meta.freq_dist_path)
    distributional_model = LogCoOccurrenceCountModel(corpus_meta, window_radius=5, freq_dist=freq_dist)
    distributional_model.train(memory_map=True)

    filtered_words = set(freq_dist.most_common_tokens(top_n_words))
    function_words = set(freq_dist.most_common_tokens(top_n_function_words))
    filtered_words -= function_words
    filtered_words = sorted(filtered_words)

    logger.info(f"Building graph from top {top_n_words:,} words, except these:")
    logger.info(f"\t{', '.join(function_words)}")
    logger.info(f"This will include a total of {len(filtered_words):,} words, "
                f"and a maximum of {int(0.5 * len(filtered_words) * (len(filtered_words) - 1)):,} connections")

    # Build index-lookup dictionaries
    filtered_indices = sorted([freq_dist.token2id[w] for w in filtered_words])
    ldm_to_matrix, matrix_to_ldm = filtering_dictionaries(filtered_indices)

    # Build distance matrix
    logger.info("Constructing distance matrix")
    embedding_matrix = distributional_model.matrix.tocsr()[filtered_indices, :].copy()
    distance_matrix = pairwise_distances(embedding_matrix, metric="cosine", n_jobs=-1)

    # Build graph
    logger.info("Building graph")
    word_graph = graph_from_distance_matrix(
        distance_matrix=distance_matrix.copy(),
        weighted_graph=False,
        length_granularity=granularity)

    # Run multiple times with different parameters

    category_production = CategoryProduction()

    results_df = DataFrame()

    for category in category_production.category_labels:

        # Skip the check if the category won't be in the network
        if category not in filtered_words:
            continue

        logger.info(f"Category: {category}")

        for activation_threshold in activation_thresholds:
            for node_decay_factor in node_decay_factors:
                for edge_decay_sd_frac in edge_decay_sd_fracs:

                    ordered_word_list = []

                    logger.info(f"")
                    logger.info(f"Setting up spreading output")
                    logger.info(f"Using values: θ={activation_threshold}, δ={node_decay_factor}, sd_frac={edge_decay_sd_frac}")

                    tsa = TemporalSpreadingActivation(graph=word_graph,
                                                      node_relabelling_dictionary=build_relabelling_dictionary(
                                                          ldm_to_matrix, freq_dist),
                                                      activation_threshold=activation_threshold,
                                                      impulse_pruning_threshold=impulse_pruning_threshold,
                                                      node_decay_function=decay_function_exponential_with_decay_factor(
                                                          decay_factor=node_decay_factor),
                                                      edge_decay_function=decay_function_gaussian_with_sd_fraction(
                                                          sd_frac=edge_decay_sd_frac, granularity=granularity))

                    logger.info(f"Initial node {category}")
                    tsa.activate_node(category, 1)
                    logger.info(f"\tNodes: {tsa.n_suprathreshold_nodes:,}, impulses: {len(tsa.impulses):,}.")

                    logger.info("Running spreading output")
                    for tick in range(1, n_ticks):

                        # Spread the activation
                        logger.info(f"Clock = {tick}")
                        tsa.tick()
                        logger.info(f"\tNodes: {tsa.n_suprathreshold_nodes:,}, impulses: {len(tsa.impulses):,}.")

                        ordered_word_list.extend(tsa.nodes_activated_this_tick)

                        # Break early if we've got a probable explosion
                        if tsa.n_suprathreshold_nodes() > explosion_bailout:
                            logger.info("EXPLOSION BAILOUT!")
                            break

                    response_indices = []
                    # Look for indices of overlap
                    responses = category_production.responses_for_category(category)[:top_n_responses]
                    for response in responses:
                        try:
                            index_of_response_in_activated_words = ordered_word_list.index(response)
                            response_indices.append(index_of_response_in_activated_words)
                        # TODO: what is this doing?
                        except ValueError:
                            pass

                    # Prepare results
                    results_these_params = tsa.activation_history
                    results_these_params["Activation threshold"] = activation_threshold
                    results_these_params["Node decay factor"] = node_decay_factor
                    results_these_params["Edge decay SD"] = edge_decay_sd_frac
                    results_these_params["Category"] = category
                    results_these_params[f"Indices of top {top_n_responses} responses"] = ",".join([str(i) for i in response_indices])

                    results_df = results_df.append(results_these_params, ignore_index=True)

        results_df.to_csv(csv_location, header=True, index=False)


def filtering_dictionaries(filtered_indices):
    # A lookup dictionary which converts the index of a word in the LDM to its index in the specific filtered matrix
    ldm_to_matrix_index = {}
    # A lookup dictionary which converts the index of a word in the specific filtered distance matrix to its index in
    # the LDM
    matrix_to_ldm_index = {}
    for i, filtered_index in enumerate(filtered_indices):
        ldm_to_matrix_index[filtered_index] = i
        matrix_to_ldm_index[i] = filtered_index
    return ldm_to_matrix_index, matrix_to_ldm_index


def build_relabelling_dictionary(ldm_to_matrix, freq_dist: FreqDist):
    ti = TokenIndex.from_freqdist_ranks(freq_dist)
    relabelling_dictionary = dict()
    for token_index, matrix_index in ldm_to_matrix.items():
        relabelling_dictionary[matrix_index] = ti.id2token[token_index]
    return relabelling_dictionary


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
