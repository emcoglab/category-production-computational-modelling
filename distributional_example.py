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
    logger.info("Training distributional model")

    corpus_meta = CorpusPreferences.source_corpus_metas[0]
    freq_dist = FreqDist.load(corpus_meta.freq_dist_path)
    distributional_model_index = TokenIndexDictionary.from_freqdist(freq_dist)
    distributional_model = LogCoOccurrenceCountModel(corpus_meta, window_radius=1, token_indices=distributional_model_index)
    distributional_model.train(memory_map=True)

    filtered_words = get_word_list(freq_dist, top_n=300) # school is in the top 300
    # filtered_words = ["lion", "tiger", "jungle"]
    filtered_indices = [distributional_model_index.token2id[w] for w in filtered_words]
    # TODO: explain what these dictionaries are
    ldm_to_matrix, matrix_to_ldm = filtering_dictionaries([distributional_model_index.token2id[w] for w in filtered_words])

    logger.info("Constructing weight matrix")

    # First coordinate (row index) points to TODO: what?
    embedding_matrix = distributional_model.matrix.tocsr()[filtered_indices, :]

    # Convert to distance matrix
    distance_matrix = pairwise_distances(embedding_matrix, metric="cosine", n_jobs=-1)

    logger.info("Building graph")

    graph = TemporalSpreadingActivation.graph_from_distance_matrix(
        distance_matrix=distance_matrix.copy(),
        length_granularity=100,
        weight_factor=20,
        # Relabel nodes with words rather than indices
        relabelling_dict=build_relabelling_dictionary(ldm_to_matrix, distributional_model_index))

    # TODO: Does this make a difference somehow?
    del distributional_model, embedding_matrix, distance_matrix

    logger.info("Setting up spreading output")

    sa = TemporalSpreadingActivation(
        graph=graph,
        threshold=0.25,
        node_decay_function=TemporalSpreadingActivation.decay_function_exponential_with_params(
            decay_factor=0.99),
        edge_decay_function=TemporalSpreadingActivation.decay_function_gaussian_with_params(
            sd=15),
        )

    activation_trace = []

    initial_word = "school"

    # sa.activate_node(initial_word, 1)
    # run_with_pdf_output(sa, 100, "/Users/cai/Desktop/graph.pdf")
    # sa.reset()

    logger.info(f"Activating initial node {initial_word}")
    sa.activate_node(initial_word, 1)
    activation_trace.append(sa.activation_snapshot())

    logger.info("Running spreading output")
    for i in range(1, 100):
        logger.info(f"Clock = {i}")
        sa.tick()
        # sa.log_graph()
        activation_trace.append(sa.activation_snapshot())

    trace_df = DataFrame.from_records(activation_trace)
    trace_df.to_csv("/Users/caiwingfield/Desktop/trace.csv")


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


def get_word_list(freq_dist, top_n):
    return [word for word, _ in freq_dist.most_common(top_n)]


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
