"""
===========================
Save a number of commonly-used graphs built from the most-frequent words in the corpora.
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

import json
import logging
import sys
from os import path

from ldm.corpus.indexing import FreqDist, TokenIndex
from ldm.model.base import DistributionalSemanticModel
from ldm.model.ngram import NgramModel, LogNgramModel
from ldm.utils.logging import log_message, date_format
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.graph import save_edgelist_from_similarity_matrix
from model.utils.indexing import list_index_dictionaries
from preferences import Preferences

logger = logging.getLogger(__name__)


def main():

    length_factor = 100

    corpus = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    token_index = TokenIndex.from_freqdist_ranks(freq_dist)
    distributional_model: NgramModel = LogNgramModel(corpus, window_radius=5, freq_dist=freq_dist)

    for word_count in Preferences.graph_sizes:
        logger.info(f"{word_count:,} words:")

        filtered_words = freq_dist.most_common_tokens(word_count)
        filtered_ldm_ids = sorted([token_index.token2id[w] for w in filtered_words])

        # These dictionaries translate between matrix-row/column indices (after filtering) and token indices within the LDM.
        _, matrix_to_ldm = list_index_dictionaries(filtered_ldm_ids)

        # Where to save edgelist
        if distributional_model.model_type.metatype is DistributionalSemanticModel.MetaType.ngram:
            edgelist_filename = f"{distributional_model.name} {word_count} words length {length_factor}.edgelist"
        else:
            raise NotImplementedError()
        edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)

        if path.isfile(edgelist_path):
            logger.info(f"{edgelist_filename} already computed.")
            logger.info("Skipping")
        else:
            logger.info("Computing between-word distances")
            distributional_model.train(memory_map=True)

            # Count, predict and n-gram models will be treated differently when building the graph
            if distributional_model.model_type.metatype is DistributionalSemanticModel.MetaType.ngram:

                logger.info("Saving edgelist")

                # Convert to csr for slicing rows and to csc for slicing columns
                similarity_matrix = distributional_model.underlying_count_model.matrix
                save_edgelist_from_similarity_matrix(
                    file_path=edgelist_path,
                    similarity_matrix=similarity_matrix,
                    filtered_node_ids=filtered_ldm_ids,
                    length_factor=length_factor)

            else:
                raise NotImplementedError()

        # Node label dictionaries

        # A dictionary whose keys are nodes (i.e. row-ids for the distance matrix) and whose values are labels for those
        # nodes (i.e. the word for the LDM-id corresponding to that row-id).
        node_label_dict = {node_id: token_index.id2token[ldm_id]
                           for (node_id, ldm_id) in matrix_to_ldm.items()}

        # Save node label dictionary
        node_label_filename = f"{corpus.name} {word_count} words.nodelabels"
        node_label_filename = path.join(Preferences.graphs_dir, node_label_filename)
        if path.isfile(node_label_filename):
            logger.info(f"{path.basename(node_label_filename)} exists, skipping")
        else:
            logger.info(f"Saving node label dictionary to {path.basename(node_label_filename)}")
            with open(node_label_filename, mode="w", encoding="utf-8") as node_label_file:
                json.dump(node_label_dict, node_label_file)


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
