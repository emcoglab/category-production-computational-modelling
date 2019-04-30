"""
===========================
Spreading activation on graphs pruned by importance rather than length.
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
import logging
import sys
from collections import defaultdict
from os import path, makedirs

from pandas import DataFrame

from category_production.category_production import CategoryProduction
from cli.lookups import get_corpus_from_name, get_model_from_params
from ldm.corpus.indexing import FreqDist, TokenIndex
from ldm.model.base import DistributionalSemanticModel
from ldm.utils.maths import DistanceType
from model.graph import Graph, iter_edges_from_edgelist, log_graph_topology
from model.temporal_spreading_activation import TemporalSpreadingActivation, load_labels_from_corpus
from model.utils.email import Emailer
from model.utils.file import comment_line_from_str
from model.utils.indexing import list_index_dictionaries
from model.utils.maths import decay_function_exponential_with_decay_factor, decay_function_gaussian_with_sd
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


# Results DataFrame column names
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "Activation"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"


def main(n_words: int,
         prune_importance: int,
         corpus_name: str,
         model_name: str,
         radius: int,
         distance_type_name: str,
         length_factor: int,
         firing_threshold: float,
         node_decay_factor: float,
         edge_decay_sd_factor: float,
         impulse_pruning_threshold: float,
         run_for_ticks: int,
         bailout: int,
         ):

    corpus = get_corpus_from_name(corpus_name)
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    token_index = TokenIndex.from_freqdist_ranks(freq_dist)
    distance_type = DistanceType.from_name(distance_type_name)
    distributional_model: DistributionalSemanticModel = get_model_from_params(corpus, freq_dist, model_name, radius)

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
    if prune_importance is not None:
        logger.info(f"Loading graph from {graph_file_name}, pruning importance {prune_importance}")
    else:
        logger.info(f"Loading graph from {graph_file_name}")

    # Load graph
    graph = Graph.load_from_edgelist_with_importance_pruning(
        file_path=path.join(Preferences.graphs_dir, graph_file_name),
        ignore_edges_with_importance_greater_than=prune_importance,
        keep_at_least_n_edges=Preferences.min_edges_per_node
    )
    n_edges = len(graph.edges)

    connected, orphans = log_graph_topology(graph)

    # Load node relabelling dictionary
    logger.info(f"Loading node labels")
    node_labelling_dictionary = load_labels_from_corpus(corpus, n_words)

    cp = CategoryProduction()

    for category_label in cp.category_labels:

        # Skip the check if the category won't be in the network
        if category_label not in filtered_words:
            continue

        # Output file path
        if prune_importance is not None:
            response_dir = path.join(Preferences.output_dir,
                                     f"Category production traces ({n_words:,} words; "
                                     f"firing {firing_threshold}; "
                                     f"edge importance threshold {prune_importance})")
        else:
            response_dir = path.join(Preferences.output_dir,
                                     f"Category production traces ({n_words:,} words; "
                                     f"firing {firing_threshold}; ")
        if not path.isdir(response_dir):
            logger.warning(f"{response_dir} directory does not exist; making it.")
            makedirs(response_dir)
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
        csv_comments.append(f"\t          δ = {node_decay_factor}")
        csv_comments.append(f"\t  sd_factor = {edge_decay_sd_factor}")
        csv_comments.append(f"\t  connected = {'yes' if connected else 'no'}")
        if not connected:
            csv_comments.append(f"\t    orphans = {'yes' if orphans else 'no'}")

        # Do the spreading activation

        tsa: TemporalSpreadingActivation = TemporalSpreadingActivation(
            graph=graph,
            item_labelling_dictionary=node_labelling_dictionary,
            firing_threshold=firing_threshold,
            impulse_pruning_threshold=impulse_pruning_threshold,
            node_decay_function=decay_function_exponential_with_decay_factor(
                decay_factor=node_decay_factor),
            edge_decay_function=decay_function_gaussian_with_sd(
                sd=edge_decay_sd_factor*length_factor))

        tsa.activate_item_with_label(category_label, 1)

        model_response_entries = []
        for tick in range(1, run_for_ticks):

            logger.info(f"Clock = {tick}")
            node_activations = tsa.tick()

            for na in node_activations:
                model_response_entries.append((
                    na.label,
                    tsa.label2idx[na.label],
                    na.activation,
                    na.time_activated,
                ))

            # Break early if we've got a probable explosion
            if len(tsa.suprathreshold_items()) > bailout:
                csv_comments.append(f"")
                csv_comments.append(f"Spreading activation ended with a bailout after {tick} ticks "
                                    f"with {len(tsa.suprathreshold_items())} nodes activated.")
                break

        model_responses_df = DataFrame(model_response_entries, columns=[
            RESPONSE,
            NODE_ID,
            ACTIVATION,
            TICK_ON_WHICH_ACTIVATED,
        ]).sort_values([TICK_ON_WHICH_ACTIVATED, NODE_ID])

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

    parser.add_argument("-b", "--bailout", required=True, type=int)
    parser.add_argument("-c", "--corpus_name", required=True, type=str)
    parser.add_argument("-f", "--firing_threshold", required=True, type=float)
    parser.add_argument("-i", "--impulse_pruning_threshold", required=True, type=float)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-m", "--model_name", required=True, type=str)
    parser.add_argument("-n", "--node_decay_factor", required=True, type=float)
    parser.add_argument("-p", "--prune_importance", required=False, type=int, help="The importance level from which to prune from the graph.", default=None)
    parser.add_argument("-r", "--radius", required=True, type=int)
    parser.add_argument("-s", "--edge_decay_sd_factor", required=True, type=float)
    parser.add_argument("-t", "--run_for_ticks", required=True, type=int)
    parser.add_argument("-w", "--words", type=int, required=True, help="The number of words to use from the corpus. (Top n words.)")

    args = parser.parse_args()

    main(n_words=args.words,
         prune_importance=args.prune_importance,
         corpus_name=args.corpus_name,
         model_name=args.model_name,
         radius=args.radius,
         distance_type_name=args.distance_type_name,
         length_factor=args.length_factor,
         firing_threshold=args.firing_threshold,
         node_decay_factor=args.node_decay_factor,
         edge_decay_sd_factor=args.edge_decay_sd_factor,
         impulse_pruning_threshold=args.impulse_pruning_threshold,
         run_for_ticks=args.run_for_ticks,
         bailout=args.bailout)
    logger.info("Done!")

    emailer = Emailer(Preferences.email_connection_details_path)
    if args.prune_importance is not None:
        emailer.send_email(f"Done running {path.basename(__file__)} with {args.words} words"
                           f" and words at least {args.prune_importance} importance.",
                           Preferences.target_email_address)
    else:
        emailer.send_email(f"Done running {path.basename(__file__)} with {args.words} words.",
                           Preferences.target_email_address)
