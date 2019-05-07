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
from evaluation.model_specs import save_model_spec_sensorimotor
from ldm.utils.maths import DistanceType
from model.graph import Graph, log_graph_topology, Node
from model.temporal_spatial_propagation import TemporalSpatialPropagation
from model.temporal_spreading_activation import load_labels_from_sensorimotor, ActivationValue
from model.utils.email import Emailer
from model.utils.file import comment_line_from_str
from model.utils.maths import make_decay_function_lognormal
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"

# Results DataFrame column names
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "Activation"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"

FULL_ACTIVATION = ActivationValue(1.0)


def main(distance_type_name: str,
         length_factor: int,
         pruning_length: int,
         impulse_pruning_threshold: float,
         run_for_ticks: int,
         sigma: float,
         bailout: int = None,
         use_prepruned: bool = False):

    distance_type = DistanceType.from_name(distance_type_name)

    # Load node relabelling dictionary
    logger.info(f"Loading node labels")
    node_labelling_dictionary = load_labels_from_sensorimotor()
    sm_words = set(w for i, w in node_labelling_dictionary.items())

    if use_prepruned:
        logger.warning("Using pre-pruned graph. THIS SHOULD BE USED FOR TESTING PURPOSES ONLY!")

        edgelist_filename = f"sensorimotor for testing only {distance_type.name} distance length {length_factor} pruned {pruning_length}.edgelist"
        edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)

        logger.info(f"Loading sensorimotor graph ({edgelist_filename})")
        sensorimotor_graph = Graph.load_from_edgelist(file_path=edgelist_path, with_feedback=True)

        # nodes which got removed from the edgelist because all their edges got pruned
        for i, w in node_labelling_dictionary.items():
            sensorimotor_graph.add_node(Node(i))

    else:

        edgelist_filename = f"sensorimotor {distance_type.name} distance length {length_factor}.edgelist"
        edgelist_path = path.join(Preferences.graphs_dir, edgelist_filename)

        logger.info(f"Loading sensorimotor graph ({edgelist_filename})")
        sensorimotor_graph = Graph.load_from_edgelist(file_path=edgelist_path,
                                                      ignore_edges_longer_than=pruning_length)

    n_edges = len(sensorimotor_graph.edges)

    # Topology
    connected, orphans = log_graph_topology(sensorimotor_graph)

    # Output file path
    response_dir = path.join(Preferences.output_dir,
                             f"Category production traces [sensorimotor {distance_type.name}] "
                             f"length {length_factor}, pruning at {pruning_length} "
                             f"sigma {sigma}; pt {impulse_pruning_threshold}; "
                             f"rft {run_for_ticks}; bailout {bailout}")
    if not path.isdir(response_dir):
        logger.warning(f"{response_dir} directory does not exist; making it.")
        makedirs(response_dir)

    save_model_spec_sensorimotor(length_factor, pruning_length, run_for_ticks, bailout, sigma, response_dir)

    cp = CategoryProduction()

    for category_label in cp.category_labels:

        # Get the SM-compatible version if there is one
        category_label = cp.apply_sensorimotor_substitution(category_label)

        # Skip the check if the category won't be in the network
        if category_label not in sm_words:
            continue

        model_responses_path = path.join(response_dir, f"responses_{category_label}.csv")

        # Only run the TSA if we've not already done it
        if path.exists(model_responses_path):
            logger.info(f"{model_responses_path} exists, skipping.")
            continue

        csv_comments = []

        logger.info(f"Running spreading activation for category {category_label}")

        csv_comments.append(f"Running sensorimotor spreading activation using parameters:")
        csv_comments.append(f"\t        edges = {n_edges:_}")
        csv_comments.append(f"\tlength_factor = {length_factor:_}")
        csv_comments.append(f"\t      pruning = {pruning_length}")
        csv_comments.append(f"\t            σ = {sigma} (σ * lf = {sigma * length_factor})")
        csv_comments.append(f"\t    connected = {'yes' if connected else 'no'}")
        if not connected:
            csv_comments.append(f"\t      orphans = {'yes' if orphans else 'no'}")

        # Do the spreading activation

        tsp = TemporalSpatialPropagation(
            underlying_graph=sensorimotor_graph,
            point_labelling_dictionary=node_labelling_dictionary,
            buffer_pruning_threshold=impulse_pruning_threshold,
            # Once a node is fully activated, that's enough.
            activation_cap=FULL_ACTIVATION,
            # Sigma for the log-normal decay gets multiplied by the length factor, so that if we change the length
            # factor, sigma doesn't also  have to change for the behaviour of the model to be approximately equivalent.
            node_decay_function=make_decay_function_lognormal(sigma=sigma * length_factor)
        )

        tsp.activate_item_with_label(category_label, FULL_ACTIVATION)

        model_response_entries = []
        for tick in range(1, run_for_ticks):

            logger.info(f"Clock = {tick}")
            node_activations = tsp.tick()

            for na in node_activations:
                model_response_entries.append((
                    na.label,
                    tsp.label2idx[na.label],
                    na.activation,
                    na.time_activated))

            # Break early if we've got a probable explosion
            if bailout is not None and len(tsp.items_in_buffer()) > bailout:
                csv_comments.append(f"")
                csv_comments.append(f"Spreading activation ended with a bailout after {tick} ticks "
                                    f"with {len(tsp.items_in_buffer())} nodes activated.")
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

    parser.add_argument("-b", "--bailout", required=False, type=int, default=None)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-i", "--impulse_pruning_threshold", required=True, type=float)
    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-p", "--pruning_length", required=True, type=int)
    parser.add_argument("-s", "--node_decay_sigma", required=True, type=float)
    parser.add_argument("-t", "--run_for_ticks", required=True, type=int)
    parser.add_argument("-U", "--use_prepruned", action="store_true")

    args = parser.parse_args()

    main(pruning_length=args.pruning_length,
         distance_type_name=args.distance_type,
         length_factor=args.length_factor,
         impulse_pruning_threshold=args.impulse_pruning_threshold,
         run_for_ticks=args.run_for_ticks,
         sigma=args.node_decay_sigma,
         bailout=args.bailout,
         use_prepruned=args.use_prepruned)
    logger.info("Done!")

    Emailer(Preferences.email_connection_details_path).send_email(
        f"Done running {path.basename(__file__)} with {args.pruning_length} pruning.",
        Preferences.target_email_address)
