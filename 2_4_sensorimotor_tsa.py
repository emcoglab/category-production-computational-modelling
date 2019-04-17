"""
===========================
Model responses to Briony's category production categories using the sensorimotor model.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""
import argparse
import logging
import sys
from os import path, makedirs

from pandas import DataFrame

from category_production.category_production import CategoryProduction
from ldm.utils.maths import DistanceType
from model.graph import Graph, log_graph_topology
from model.temporal_spreading_activation import TemporalSpreadingActivation, load_labels_from_sensorimotor
from model.utils.email import Emailer
from model.utils.file import comment_line_from_str
from model.utils.maths import decay_function_exponential_with_decay_factor, decay_function_constant
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"

# Results DataFrame column names
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "ActivationValue"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"


def main(distance_type_name: str,
         length_factor: int,
         pruning_length: int,
         node_decay_factor: float,
         impulse_pruning_threshold: float,
         run_for_ticks: int,
         bailout: int,
         ):
    distance_type = DistanceType.from_name(distance_type_name)

    edgelist_filename = f"sensorimotor {distance_type.name} distance length {length_factor}.edgelist"
    edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)

    # Load graph
    sensorimotor_graph = Graph.load_from_edgelist(file_path=edgelist_path,
                                                  ignore_edges_longer_than=pruning_length)

    n_edges = len(sensorimotor_graph.edges)

    # Topology
    connected, orphans = log_graph_topology(sensorimotor_graph)

    # Load node relabelling dictionary
    logger.info(f"Loading node labels")
    node_labelling_dictionary = load_labels_from_sensorimotor()
    sm_words = set(w for i, w in node_labelling_dictionary.items())

    cp = CategoryProduction()

    for category_label in cp.category_labels:

        # Get the SM-compatible version if there is one
        category_label = cp.apply_sensorimotor_substitution(category_label)

        # Skip the check if the category won't be in the network
        if category_label not in sm_words:
            continue

        # Output file path
        response_dir = path.join(Preferences.output_dir,
                                 f"Category production traces [sensorimotor {distance_type.name}]",
                                 f"length {length_factor}, pruning at {pruning_length}"
                                 f"df {node_decay_factor}; pt {impulse_pruning_threshold}; "
                                 f"rft {run_for_ticks}; bailout {bailout}")
        if not path.isdir(response_dir):
            logger.warning(f"{response_dir} directory does not exist; making it.")
            makedirs(response_dir)
        model_responses_path = path.join(response_dir, f"responses_{category_label}.csv")

        # Only run the TSA if we've not already done it
        if path.exists(model_responses_path):
            logger.info(f"{model_responses_path} exists, skipping.")
            continue

        csv_comments = []

        logger.info(f"Running spreading activation for category {category_label}")

        csv_comments.append(f"Running sensorimotor spreading activation using parameters:")
        csv_comments.append(f"\t      edges = {n_edges:_}")
        csv_comments.append(f"\tgranularity = {length_factor:_}")
        csv_comments.append(f"\t    pruning = {pruning_length}")
        csv_comments.append(f"\t          δ = {node_decay_factor}")
        csv_comments.append(f"\t  connected = {'yes' if connected else 'no'}")
        if not connected:
            csv_comments.append(f"\t    orphans = {'yes' if orphans else 'no'}")

        # Do the spreading activation

        tsa = TemporalSpreadingActivation(
            graph=sensorimotor_graph,
            item_labelling_dictionary=node_labelling_dictionary,
            # Points can't reactivate as long as they are still in the buffer
            # (for now this just means that they have a non-zero activation)
            firing_threshold=0,
            impulse_pruning_threshold=impulse_pruning_threshold,
            node_decay_function=decay_function_exponential_with_decay_factor(
                decay_factor=node_decay_factor),
            # Once pruning has been done, we don't need to decay, as target nodes receive the full activations of
            # incident impulses. The maximum sphere radius is achieved by the initial graph pruning.
            edge_decay_function=decay_function_constant())

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
                    na.time_activated))

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
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-i", "--impulse_pruning_threshold", required=True, type=float)
    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-n", "--node_decay_factor", required=True, type=float)
    parser.add_argument("-p", "--pruning_length", required=False, type=int,
                        help="The percentage of longest edges to prune from the graph.", default=None)
    parser.add_argument("-t", "--run_for_ticks", required=True, type=int)

    args = parser.parse_args()

    main(pruning_length=args.pruning_length,
         distance_type_name=args.distance_type,
         length_factor=args.length_factor,
         node_decay_factor=args.node_decay_factor,
         impulse_pruning_threshold=args.impulse_pruning_threshold,
         run_for_ticks=args.run_for_ticks,
         bailout=args.bailout)
    logger.info("Done!")

    emailer = Emailer(Preferences.email_connection_details_path)
    if args.prune_percent is not None:
        emailer.send_email(
            f"Done running {path.basename(__file__)} with {args.words} words and {args.prune_percent:.2f}% pruning.",
            Preferences.target_email_address)
    else:
        emailer.send_email(f"Done running {path.basename(__file__)} with {args.words} words.",
                           Preferences.target_email_address)
