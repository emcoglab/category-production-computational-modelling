"""
===========================
Proof of concept for linguistic spreading activation model.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2017
---------------------------
"""

import sys
import logging
from typing import List, Set, Dict

import networkx as nx

from ldm.core.utils.exceptions import WordNotFoundError
from ldm.core.utils.maths import DistanceType
from ldm.core.corpus.indexing import TokenIndexDictionary, FreqDist
from ldm.core.model.count import PPMIModel
from ldm.preferences.preferences import Preferences as CorpusPreferences

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main():
    corpus_meta = CorpusPreferences.source_corpus_metas[0]
    distributional_model = PPMIModel(
        corpus_meta,
        window_radius=1,
        token_indices=TokenIndexDictionary.load(corpus_meta.index_path),
        freq_dist=FreqDist.load(corpus_meta.freq_dist_path))

    distributional_model.train()

    word_list = get_vocabulary()

    # Initialise grpah

    # Build unactivated graph
    graph = nx.Graph()
    for word_1 in word_list:
        for word_2 in word_list:
            try:
                # cosine similarity is 1 - cosine distance
                similarity = 1 - distributional_model.distance_between(word_1, word_2, DistanceType.cosine)
            # Skip words not in the model
            except WordNotFoundError():
                continue
            graph.add_nodes_from([word_1, word_2],
                                 activation=0.0,
                                 will_fire=False,
                                 has_fired=False)
            graph.add_edge(word_1, word_2, weight=similarity)

    logger.info(f"Graph built! {len(graph.nodes)} node, {len(graph.edges})} edges.")

    # Activate initial source
    source_word = "fruit"
    logger.info(f"Activating source word {source_word}")
    graph.nodes[source_word]["activation"] = 1

    THRESHOLD = 0.5
    DECAY = 0.5

    # Start spreading activation
    n_ticks = 10
    logger.info(f"Starting spreading activation, {n_ticks} ticks")
    for tick in range(n_ticks):
        logger.info(f"Tick {tick}")

        # Node loop

        for node, neighbours in graph.adjacency():
            if node["has_fired"]:
                continue

            # Fire unfired nodes
            if node["will_fire"]:
                node["has_fired"] = True

            source_activation = node["activation"]
            if source_activation <= THRESHOLD:
                continue

            # neighbour loop
            for neighbour, edge_attributes in neighbours.items():
                # Update activations
                weight = edge_attributes["weight"]
                neighbour["activation"] += source_activation * weight * DECAY
                # Cap activations at 0 and 1
                if neighbour["activation"] > 1:
                    neighbour["activation"] = 1
                if neighbour["activation"] < 0:
                    neighbour["activation"] = 0

        # Mark for firing
        for node in graph.nodes:
            if node["has_fired"]:
                continue
            if node["activation"] > THRESHOLD:
                node["will_fire"] = True


def get_vocabulary() -> Set[str]:
    pass


# class WordActivation(object):
#
#     def __init__(self, word: str):
#         self.word: str = word
#         self.activation: float = 0
#         self.will_fire: bool = False
#         self.has_fired: bool = False
#
#     def set_to_fire_if_exceeding_threshold(self):
#         if not self.has_fired:
#             if self.activation > WordActivation.threshold:
#                 self.will_fire = True


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, evel=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
