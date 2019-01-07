"""
===========================
Search parameter space.
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
import logging
import sys
from collections import defaultdict
from os import path, mkdir

from pandas import DataFrame

from category_production.category_production import CategoryProduction
from ldm.core.corpus.indexing import FreqDist, TokenIndex
from ldm.core.model.count import LogCoOccurrenceCountModel
from ldm.core.utils.maths import DistanceType
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.graph import Graph, iter_edges_from_edgelist
from model.temporal_spreading_activation import TemporalSpreadingActivation
from model.utils.email import Emailer
from model.utils.file import comment_line_from_str
from model.utils.indexing import list_index_dictionaries
from model.utils.math import decay_function_exponential_with_decay_factor, decay_function_gaussian_with_sd
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


# Results DataFrame column names
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "Activation"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"


def main(n_words: int, prune_importance: int, firing_threshold: float, conscious_access_threshold: float):

    n_ticks = 3_000
    length_factor = 1_000
    impulse_pruning_threshold = 0.05
    node_decay_factor = 0.99
    edge_decay_sd_frac = 0.4

    # Bail if too many words get activated
    bailout = 5_000

    corpus = CorpusPreferences.source_corpus_metas.bbc
    distance_type = DistanceType.cosine
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    token_index = TokenIndex.from_freqdist_ranks(freq_dist)
    distributional_model = LogCoOccurrenceCountModel(corpus, window_radius=5, freq_dist=freq_dist)

    filtered_words = set(freq_dist.most_common_tokens(n_words))
    filtered_ldm_ids = sorted([token_index.token2id[w] for w in filtered_words])

    # These dictionaries translate between matrix-row/column indices (after filtering) and token indices within the LDM.
    _, matrix_to_ldm = list_index_dictionaries(filtered_ldm_ids)

    # Load distance matrix
    graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"

    # Build edge distributions
    logger.info("Collating edge length distributions.")
    edge_lengths_from_node = defaultdict(list)
    for edge, length in iter_edges_from_edgelist(path.join(Preferences.graphs_dir, graph_file_name)):
        for node in edge:
            edge_lengths_from_node[node].append(length)
    edge_lengths_from_node.default_factory = None  # Make into a dict, to catch KeyErrors

    # Get pruning length
    logger.info(f"Loading graph from {graph_file_name}, pruning importance {prune_importance}")

    # Load graph
    graph = Graph.load_from_edgelist_with_importance_pruning(
        file_path=path.join(Preferences.graphs_dir, graph_file_name),
        ignore_edges_with_importance_greater_than=prune_importance,
        keep_at_least_n_edges=Preferences.min_edges_per_node
    )
    n_edges = len(graph.edges)

    # Topology
    orphans = graph.has_orphaned_nodes()
    connected = graph.is_connected()
    if connected:
        logger.info("Graph is connected")
    else:
        logger.warning("Graph is disconnected")
        if orphans:
            logger.warning("Graph has orphans")
        else:
            logger.info("Graph has no orphans")

    # Load node relabelling dictionary
    logger.info(f"Loading node labels")
    with open(path.join(Preferences.graphs_dir, f"{corpus.name} {n_words} words.nodelabels"), mode="r", encoding="utf-8") as nrd_file:
        node_relabelling_dictionary_json = json.load(nrd_file)
    # TODO: this isn't a great way to do this
    node_relabelling_dictionary = dict()
    for k, v in node_relabelling_dictionary_json.items():
        node_relabelling_dictionary[int(k)] = v

    cp = CategoryProduction()

    for category_label in cp.category_labels:

        # Skip the check if the category won't be in the network
        if category_label not in filtered_words:
            continue

        # Output file path
        response_dir = path.join(Preferences.output_dir,
                                 f"Category production traces ({n_words:,} words; "
                                 f"firing {firing_threshold}; "
                                 f"access {conscious_access_threshold}; "
                                 f"edge importance threshold {prune_importance})")
        if not path.isdir(response_dir):
            logger.warning(f"{response_dir} directory does not exist; making it.")
            mkdir(response_dir)
        model_responses_path = path.join(response_dir, f"responses_{category_label}_{n_words:,}.csv")

        # Only run the TSA if we've not already done it
        if path.exists(model_responses_path):
            logger.info(f"{model_responses_path} exists, skipping.")
            continue

        csv_comments = []

        logger.info(f"Running spreading activation for category {category_label}")

        csv_comments.append(f"Running spreading activation using parameters:")
        csv_comments.append(f"\t      words = {n_words:_}")
        csv_comments.append(f"\t      edges = {n_edges:_}")
        csv_comments.append(f"\tgranularity = {length_factor:_}")
        if prune_importance is not None:
            csv_comments.append(f"\t    pruning = {prune_importance}")
        csv_comments.append(f"\t   firing θ = {firing_threshold}")
        csv_comments.append(f"\tconc.acc. θ = {conscious_access_threshold}")
        csv_comments.append(f"\t          δ = {node_decay_factor}")
        csv_comments.append(f"\t    sd_frac = {edge_decay_sd_frac}")
        csv_comments.append(f"\t  connected = {'yes' if connected else 'no'}")
        if not connected:
            csv_comments.append(f"\t    orphans = {'yes' if orphans else 'no'}")

        # Do the spreading activation

        tsa: TemporalSpreadingActivation = TemporalSpreadingActivation(
            graph=graph,
            item_labelling_dictionary=node_relabelling_dictionary,
            firing_threshold=firing_threshold,
            conscious_access_threshold=conscious_access_threshold,
            impulse_pruning_threshold=impulse_pruning_threshold,
            node_decay_function=decay_function_exponential_with_decay_factor(
                decay_factor=node_decay_factor),
            edge_decay_function=decay_function_gaussian_with_sd(
                sd=edge_decay_sd_frac*length_factor))

        tsa.activate_item_with_label(category_label, 1)

        model_response_entries = []
        for tick in range(1, n_ticks):

            logger.info(f"Clock = {tick}")
            node_activations = tsa.tick()

            for na in node_activations:
                model_response_entries.append((
                    na.label,
                    tsa.label2idx[na.label],
                    na.activation,
                    na.time_activated
                ))

            # Break early if we've got a probable explosion
            if tsa.n_suprathreshold_nodes() > bailout:
                csv_comments.append(f"")
                csv_comments.append(f"Spreading activation ended with a bailout after {tick} ticks "
                                    f"with {tsa.n_suprathreshold_nodes()} nodes activated.")
                break

        model_responses_df = DataFrame(model_response_entries, columns=[
            RESPONSE,
            NODE_ID,
            ACTIVATION,
            TICK_ON_WHICH_ACTIVATED
        ])

        # Output results

        with open(model_responses_path, mode="w", encoding="utf-8") as output_file:
            # Write comments
            for comment in csv_comments:
                output_file.write(comment_line_from_str(comment))
            # Write data
            model_responses_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")
    parser.add_argument("n_words", type=int, help="The number of words to use from the corpus. (Top n words.)")
    parser.add_argument("prune_importance", type=int, help="Edge importance above which to prune.")
    parser.add_argument("firing_threshold", type=float, help="The firing threshold.")
    parser.add_argument("conscious_access_threshold", type=float, help="The conscious access threshold.")
    args = parser.parse_args()

    main(n_words=args.n_words, prune_importance=args.prune_importance,
         firing_threshold=args.firing_threshold, conscious_access_threshold=args.conscious_access_threshold)
    logger.info("Done!")

    emailer = Emailer(Preferences.email_connection_details_path)
    emailer.send_email(f"Done running {path.basename(__file__)} with {args.n_words} words"
                       f" and words at least {args.prune_importance} importance,"
                       f" using a firing threshold of {args.firing_threshold}"
                       f" and a conscious-access threshold of {args.conscious_access_threshold}.",
                       Preferences.target_email_address)
