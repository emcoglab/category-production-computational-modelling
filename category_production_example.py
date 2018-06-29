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

import sys
import logging
import json
from os import path
from typing import Set

from category_production.category_production import CategoryProduction
from ldm.core.corpus.indexing import FreqDist
from ldm.core.model.count import LogCoOccurrenceCountModel
from ldm.core.utils.maths import DistanceType
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.graph import load_graph
from model.temporal_spreading_activation import TemporalSpreadingActivation, \
    decay_function_exponential_with_decay_factor, decay_function_gaussian_with_sd_fraction
from model.utils.indexing import list_index_dictionaries
from preferences import Preferences

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main():

    n_words = 10_000
    n_ticks = 1_000
    length_factor = 1_000
    impulse_pruning_threshold = 0.05
    activation_threshold = 0.8
    node_decay_factor = 0.99
    edge_decay_sd_frac = 0.4

    # Bail if more than 50% of words activated
    bailout = n_words * 0.5

    logger.info("Training distributional model")
    corpus = CorpusPreferences.source_corpus_metas.bbc
    distance_type = DistanceType.cosine
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distributional_model = LogCoOccurrenceCountModel(corpus, window_radius=5, freq_dist=freq_dist)

    filtered_words = set(freq_dist.most_common_tokens(n_words))
    filtered_ldm_ids = sorted([freq_dist.token2id[w] for w in filtered_words])

    # These dictionaries translate between matrix-row/column indices (after filtering) and token indices within the LDM.
    _, matrix_to_ldm = list_index_dictionaries(filtered_ldm_ids)

    # Load distance matrix
    graph = load_graph(path.join(Preferences.graphs_dir, f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.graph"))
    node_relabelling_dictionary = json.load(path.join(Preferences.graphs_dir, f"{corpus.name} {n_words} words.nodelabels"))

    logger.info(f"Using values: θ={activation_threshold}, δ={node_decay_factor}, sd_frac={edge_decay_sd_frac}")

    # Run multiple times with different parameters

    category_production = CategoryProduction()

    for category in category_production.category_labels:

        actual_responses = category_production.responses_for_category(category)

        # Skip the check if the category won't be in the network
        if category not in filtered_words:
            continue

        logger.info(f"Category: {category}")

        tsa = TemporalSpreadingActivation(
            graph=graph,
            node_relabelling_dictionary=node_relabelling_dictionary,
            activation_threshold=activation_threshold,
            impulse_pruning_threshold=impulse_pruning_threshold,
            node_decay_function=decay_function_exponential_with_decay_factor(
                decay_factor=node_decay_factor),
            edge_decay_function=decay_function_gaussian_with_sd_fraction(
                sd_frac=edge_decay_sd_frac, granularity=length_factor))

        tsa.activate_node(category, 1)

        activated_nodes = []
        for tick in range(1, n_ticks):

            logger.info(f"Clock = {tick}")
            nodes_activated_this_tick: Set = tsa.tick()

            activated_nodes.extend(list(nodes_activated_this_tick))

            # Break early if we've got a probable explosion
            if tsa.n_suprathreshold_nodes() > bailout:
                logger.info("Bailout")
                break

        model_responses = [response
                           for response in activated_nodes
                           if response in actual_responses]

        logger.info("Actual responses:")
        logger.info(f"\t{', '.join(actual_responses)}")

        logger.info("Model reponses:")
        logger.info(f"\t{', '.join(model_responses)}")

        response_indices = []
        # Look for indices of overlap
        for response in actual_responses:
            index_of_response_in_activated_words = activated_nodes.index(response)
            response_indices.append(index_of_response_in_activated_words)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
