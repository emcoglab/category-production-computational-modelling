"""
===========================
Model responses to Briony's category production categories.
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
from os import path, mkdir

from pandas import DataFrame

from category_production.category_production import CategoryProduction
from ldm.core.corpus.indexing import FreqDist, TokenIndex
from ldm.core.model.count import LogCoOccurrenceCountModel
from ldm.core.utils.maths import DistanceType
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.graph import Graph
from model.component import load_labels
from model.temporal_spreading_activation import TemporalSpreadingActivation
from model.utils.math import decay_function_exponential_with_decay_factor, decay_function_gaussian_with_sd
from model.utils.email import Emailer
from model.utils.file import comment_line_from_str
from model.utils.indexing import list_index_dictionaries
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


# Results DataFrame column names
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "ActivationValue"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"
EXCEEDED_CAT = "Exceeded conc.acc. θ"


def main(n_words: int, prune_percent: int):

    if prune_percent == 0:
        prune_percent = None

    propagation_speed = 1/1_000
    run_for_ticks = 1_000
    impulse_pruning_threshold = 0.05
    firing_threshold = 0.8
    conscious_access_threshold = 0.9
    node_decay_factor = 0.99
    edge_decay_sd = 0.4 / propagation_speed  # t=x/v

    # Bail if too many words get activated
    bailout = 2_000

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
    graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words.edgelist"
    quantile_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words edge length quantiles.csv"

    quantile_data = DataFrame.from_csv(path.join(Preferences.graphs_dir, quantile_file_name), header=0, index_col=None)

    # Get pruning length
    if prune_percent is not None:
        pruning_length = quantile_data[
            # Use 1 - so that smallest top quantiles get converted to longest edges
            quantile_data["Top quantile"] == 1 - (prune_percent / 100)
            ]["Pruning length"].iloc[0]
        logger.info(f"Loading graph from {graph_file_name}, pruning longest {prune_percent}% of edges (anything over {pruning_length})")
    else:
        pruning_length = None
        logger.info(f"Loading graph from {graph_file_name}")

    # Load graph
    graph = Graph.load_from_edgelist(file_path=path.join(Preferences.graphs_dir, graph_file_name),
                                     ignore_edges_longer_than=pruning_length,
                                     keep_at_least_n_edges=Preferences.min_edges_per_node)

    n_edges = len(graph.edges)

    # Topology
    orphans = graph.has_orphaned_nodes()
    connected = graph.is_connected()
    if connected:
        logger.info("Graph is connected")
    else:
        logger.info("Graph is disconnected")
        if orphans:
            logger.info("Graph has orphans")
        else:
            logger.info("Graph has no orphans")

    # Load node relabelling dictionary
    logger.info(f"Loading node labels")
    node_labelling_dictionary = load_labels(corpus, n_words)

    cp = CategoryProduction()

    for category_label in cp.category_labels:

        # Skip the check if the category won't be in the network
        if category_label not in filtered_words:
            continue

        # Output file path
        if prune_percent is not None:
            response_dir = path.join(Preferences.output_dir,
                                     f"Category production traces ({n_words:,} words; "
                                     f"firing {firing_threshold}; "
                                     f"access {conscious_access_threshold}; "
                                     f"longest {prune_percent}% edges removed)")
        else:
            response_dir = path.join(Preferences.output_dir,
                                     f"Category production traces ({n_words:,} words; "
                                     f"firing {firing_threshold}; "
                                     f"access {conscious_access_threshold})")
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
        if prune_percent is not None:
            csv_comments.append(f"\t      pruning = {prune_percent:.2f}% ({pruning_length})")
        csv_comments.append(f"\timpulse speed = {propagation_speed}")
        csv_comments.append(f"\t     firing θ = {firing_threshold}")
        csv_comments.append(f"\t  conc.acc. θ = {conscious_access_threshold}")
        csv_comments.append(f"\t node decay δ = {node_decay_factor}")
        csv_comments.append(f"\tedge decay sd = {edge_decay_sd}")
        csv_comments.append(f"\t    connected = {'yes' if connected else 'no'}")
        if not connected:
            csv_comments.append(f"\t      orphans = {'yes' if orphans else 'no'}")

        # Do the spreading activation

        tsa = TemporalSpreadingActivation(
            graph=graph,
            item_labelling_dictionary=node_labelling_dictionary,
            firing_threshold=firing_threshold,
            impulse_pruning_threshold=impulse_pruning_threshold,
            impulse_propagation_speed=propagation_speed,
            node_decay_function=decay_function_exponential_with_decay_factor(
                decay_factor=node_decay_factor),
            edge_decay_function=decay_function_gaussian_with_sd(
                sd=edge_decay_sd))

        tsa.activate_item_with_label(category_label, 1)

        model_response_entries = []
        for tick in range(1, run_for_ticks):

            logger.info(f"Clock = {tick}")
            node_activations = tsa.tick()

            for na in node_activations:
                model_response_entries.append((
                    na.label,
                    tsa.label2idx[na.node],
                    na.activation,
                    na.time_activated,
                    "Exceeded conc.acc. θ" if na.activation >= conscious_access_threshold else ""
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
            TICK_ON_WHICH_ACTIVATED,
            EXCEEDED_CAT
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
    parser.add_argument("n_words", type=int, help="The number of words to use from the corpus. (Top n words.)")
    parser.add_argument("prune_percent", type=int, nargs="?", help="The percentage of longest edges to prune from the graph.", default=None)
    args = parser.parse_args()

    main(n_words=args.n_words, prune_percent=args.prune_percent)
    logger.info("Done!")

    emailer = Emailer(Preferences.email_connection_details_path)
    if args.prune_percent is not None:
        emailer.send_email(f"Done running {path.basename(__file__)} with {args.n_words} words and {args.prune_percent:.2f}% pruning.",
                           Preferences.target_email_address)
    else:
        emailer.send_email(f"Done running {path.basename(__file__)} with {args.n_words} words.",
                           Preferences.target_email_address)
