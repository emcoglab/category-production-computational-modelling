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
import sys
from os import path

from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances

from ldm.core.corpus.indexing import FreqDist, TokenIndex
from ldm.core.model.count import LogCoOccurrenceCountModel
from ldm.core.utils.logging import log_message, date_format
from ldm.core.utils.maths import DistanceType
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.graph import Graph
from model.temporal_spreading_activation import TemporalSpreadingActivation, \
    decay_function_exponential_with_decay_factor, decay_function_gaussian_with_sd
from model.utils.indexing import list_index_dictionaries
from preferences import Preferences

logger = logging.getLogger()


def main():

    n_words = 10_000
    n_ticks = 1_000
    length_factor = 1_000
    initial_word = "colour"
    impulse_pruning_threshold = 0.05

    # Bail on computation if too many nodes get activated
    bailout = n_words * 0.5

    corpus = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    token_index = TokenIndex.from_freqdist_ranks(freq_dist)
    distance_type = DistanceType.cosine
    distributional_model = LogCoOccurrenceCountModel(corpus, window_radius=5, freq_dist=freq_dist)

    filtered_ldm_ids = sorted([token_index.token2id[token]
                               for token in freq_dist.most_common_tokens(n_words)])

    # These dictionaries translate between matrix-row/column indices (after filtering) and token indices within the LDM.
    _, matrix_to_ldm = list_index_dictionaries(filtered_ldm_ids)

    # A dictionary whose keys are nodes (i.e. row-ids for the distance matrix) and whose values are labels for those
    # nodes (i.e. the word for the LDM-id corresponding to that row-id).
    node_relabelling_dictionary = {node_id: token_index.id2token[ldm_id]
                                   for (node_id, ldm_id) in matrix_to_ldm.items()}

    edgelist_filename = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"
    edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)

    if path.isfile(edgelist_path):
        logger.info(f"Loading graph with {n_words:,} nodes")
        graph: Graph = Graph.load_from_edgelist(edgelist_path)
    else:
        logger.info("Training distributional model")
        distributional_model.train(memory_map=True)

        logger.info("Constructing weight matrix")

        # First coordinate (row index) points to target words
        embedding_matrix = distributional_model.matrix.tocsr()[filtered_ldm_ids, :]

        # Convert to distance matrix
        distance_matrix = pairwise_distances(embedding_matrix, metric=distance_type.name, n_jobs=-1)
        # free ram
        del embedding_matrix

        logger.info(f"Building graph with {n_words:,} nodes")
        graph: Graph = Graph.from_distance_matrix(
            distance_matrix=distance_matrix,
            weighted_graph=False,
            length_granularity=length_factor)
        # free ram
        del distance_matrix

        logger.info("Saving graph")
        graph.save_as_edgelist(edgelist_path)

    # Run spreading activation
    d = []
    for activation_threshold in [0.8]:
        for node_decay_factor in [0.99]:
            for edge_decay_sd in [400]:

                logger.info(f"Setting up spreading output")
                logger.info(f"Using values: l={length_factor}, θ={activation_threshold}, δ={node_decay_factor}, sd={edge_decay_sd}")

                tsa = TemporalSpreadingActivation(
                    graph=graph,
                    activation_threshold=activation_threshold,
                    impulse_pruning_threshold=impulse_pruning_threshold,
                    node_relabelling_dictionary=node_relabelling_dictionary,
                    node_decay_function=decay_function_exponential_with_decay_factor(
                        decay_factor=node_decay_factor),
                    edge_decay_function=decay_function_gaussian_with_sd(
                        sd=edge_decay_sd))

                logger.info(f"Initial node {initial_word}")
                tsa.activate_node_with_label(initial_word, 1)

                logger.info("Running spreading output")
                for tick in range(1, n_ticks):
                    logger.info(f"Clock = {tick}")
                    nodes_fired = tsa.tick()
                    nodes_fired_str = ", ".join([f"{tsa.node2label[n]} ({tsa.activation_of_node(n):.3})" for n in nodes_fired])

                    if len(nodes_fired) > 0:
                        logger.info("\t" + nodes_fired_str)

                    # Record results
                    d.append({
                        'Tick': tick,
                        'Nodes fired': nodes_fired_str,
                        "Activation threshold": activation_threshold,
                        "Node decay factor": node_decay_factor,
                        "Edge decay SD": edge_decay_sd,
                        "Activated nodes": tsa.n_suprathreshold_nodes()
                    })

                    # Every so often, check if we've got explosive behaviour
                    if tick % 10 == 0:
                        if tsa.n_suprathreshold_nodes() >= bailout:
                            logger.warning("Bailout!")
                            break

    csv_location = path.join(Preferences.output_dir, "activated node counts.csv")
    DataFrame(d).to_csv(csv_location, header=True, index=False)


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
