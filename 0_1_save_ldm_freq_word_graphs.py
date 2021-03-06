#!/Users/cai/Applications/miniconda3/bin/python
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
import argparse
import json
import sys
from os import path

from sklearn.metrics.pairwise import pairwise_distances

from framework.cli.lookups import get_corpus_from_name, get_model_from_params
from framework.cognitive_model.ldm.corpus.indexing import FreqDist, TokenIndex
from framework.cognitive_model.ldm.model.base import DistributionalSemanticModel
from framework.cognitive_model.ldm.model.count import CountVectorModel
from framework.cognitive_model.ldm.utils.maths import DistanceType
from framework.cognitive_model.utils.logging import logger
from framework.cognitive_model.graph import save_edgelist_from_distance_matrix
from framework.cognitive_model.utils.indexing import list_index_dictionaries
from framework.cognitive_model.preferences.preferences import Preferences


def main(length_factor: int, corpus_name: str, distance_type_name: str, model_name: str, radius: int):

    corpus = get_corpus_from_name(corpus_name)
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    token_index = TokenIndex.from_freqdist_ranks(freq_dist)
    distance_type = DistanceType.from_name(distance_type_name)
    distributional_model: CountVectorModel = get_model_from_params(corpus, freq_dist, model_name, radius)

    for word_count in Preferences.graph_sizes:
        logger.info(f"{word_count:,} words:")

        filtered_words = freq_dist.most_common_tokens(word_count)
        filtered_ldm_ids = sorted([token_index.token2id[w] for w in filtered_words])

        # These dictionaries translate between matrix-row/column indices (after filtering) and token indices within the LDM.
        _, matrix_to_ldm = list_index_dictionaries(filtered_ldm_ids)

        # Where to save edgelist
        if distributional_model.model_type.metatype is DistributionalSemanticModel.MetaType.count:
            edgelist_filename = f"{distributional_model.name} {distance_type.name} {word_count} words length {length_factor}.edgelist"
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
            if distributional_model.model_type.metatype is DistributionalSemanticModel.MetaType.count:
                # Convert to csr for slicing, then slice out the rows (i.e. target words) corresponding to the filtered
                # ids, but leave columns (i.e. context words: data) in-tact for computation of pairwise distances.
                embedding_matrix = distributional_model.matrix.tocsr()[filtered_ldm_ids, :]
                distance_matrix = pairwise_distances(embedding_matrix, metric=distance_type.name, n_jobs=-1)
                # free ram
                del embedding_matrix

                logger.info(f"Saving graph edgelist to {edgelist_filename}")
                save_edgelist_from_distance_matrix(
                    file_path=edgelist_path,
                    distance_matrix=distance_matrix,
                    length_factor=length_factor)
                # free ram
                del distance_matrix

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

    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Save a number of commonly-used graphs built from the most frequent words in the corpora.")
    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-c", "--corpus_name", required=True, type=str)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-m", "--model_name", required=True, type=str)
    parser.add_argument("-r", "--radius", required=True, type=int)
    args = parser.parse_args()

    main(length_factor=args.length_factor,
         corpus_name=args.corpus_name,
         distance_type_name=args.distance_type,
         model_name=args.model_name,
         radius=args.radius)

    logger.info("Done!")
