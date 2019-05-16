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
from model.graph import Length
from model.sensorimotor_component import SensorimotorComponent, save_model_spec_sensorimotor
from model.common import ActivationValue
from model.utils.email import Emailer
from model.utils.file import comment_line_from_str
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
         max_sphere_radius: int,
         buffer_size_limit: int,
         buffer_entry_threshold: ActivationValue,
         buffer_pruning_threshold: float,
         run_for_ticks: int,
         sigma: float,
         use_prepruned: bool,
         bailout: int = None,
         ):

    distance_type = DistanceType.from_name(distance_type_name)

    # Output file path
    response_dir = path.join(Preferences.output_dir,
                             f"Category production traces [sensorimotor {distance_type.name}] "
                             f"length {length_factor}, max r {max_sphere_radius} "
                             f"sigma {sigma}; bpt {buffer_pruning_threshold}; "
                             f"rft {run_for_ticks}; bailout {bailout}")
    if not path.isdir(response_dir):
        logger.warning(f"{response_dir} directory does not exist; making it.")
        makedirs(response_dir)

    cp = CategoryProduction()
    sc = SensorimotorComponent(
        distance_type=distance_type,
        length_factor=length_factor,
        max_sphere_radius=max_sphere_radius,
        lognormal_sigma=sigma,
        buffer_size_limit=buffer_size_limit,
        buffer_entry_threshold=buffer_entry_threshold,
        buffer_pruning_threshold=buffer_pruning_threshold,
        # Once a node is fully activated, that's enough.
        activation_cap=FULL_ACTIVATION,
        use_prepruned=use_prepruned,
    )

    save_model_spec_sensorimotor(length_factor, max_sphere_radius, sigma, response_dir)

    for category_label in cp.category_labels_sensorimotor:

        model_responses_path = path.join(response_dir, f"responses_{category_label}.csv")
        concurrent_activations_path = path.join(response_dir, f"concurrent_activations_{category_label}.csv")

        csv_comments = []

        # Skip the check if the category won't be in the network
        if category_label not in sc.concept_labels:
            continue

        # Only run the TSA if we've not already done it
        if path.exists(model_responses_path):
            logger.info(f"{model_responses_path} exists, skipping.")
            continue

        logger.info(f"Running spreading activation for category {category_label}")

        sc.reset()

        # Record topology
        csv_comments.append(f"Running sensorimotor spreading activation using parameters:")
        csv_comments.append(f"\tlength_factor = {length_factor:_}")
        csv_comments.append(f"\t      pruning = {max_sphere_radius}")
        csv_comments.append(f"\t            σ = {sigma} (σ * lf = {sigma * length_factor})")
        if sc.graph.is_connected():
            csv_comments.append(f"\t    connected = yes")
        else:
            csv_comments.append(f"\t    connected = no")
            csv_comments.append(f"\t      orphans = {'yes' if sc.graph.has_orphaned_nodes() else 'no'}")

        # Do the spreading activation

        sc.activate_item_with_label(category_label, FULL_ACTIVATION)

        n_concurrent_activations = []
        model_response_entries = []
        for tick in range(1, run_for_ticks):

            logger.info(f"Clock = {tick}")
            node_activations = sc.tick()

            n_concurrent_activations.append((tick, len(node_activations), len(sc.accessible_set())))

            for na in node_activations:
                model_response_entries.append((
                    na.label,
                    sc.label2idx[na.label],
                    na.activation,
                    na.time_activated))

            # Break early if we've got a probable explosion
            if bailout is not None and len(sc.accessible_set()) > bailout:
                csv_comments.append(f"")
                csv_comments.append(f"Spreading activation ended with a bailout after {tick} ticks "
                                    f"with {len(sc.accessible_set())} nodes activated.")
                break

        model_responses_df = DataFrame(model_response_entries, columns=[
            RESPONSE,
            NODE_ID,
            ACTIVATION,
            TICK_ON_WHICH_ACTIVATED,
        ]).sort_values([TICK_ON_WHICH_ACTIVATED, NODE_ID])

        # Output results

        # Model ouput
        with open(model_responses_path, mode="w", encoding="utf-8") as output_file:
            # Write comments
            for comment in csv_comments:
                output_file.write(comment_line_from_str(comment))
            # Write data
            model_responses_df.to_csv(output_file, index=False)

        # Concurrent activations
        with open(concurrent_activations_path, mode="w", encoding="utf-8") as concurrent_activations_file:
            DataFrame.from_records(n_concurrent_activations,
                                   columns=["Tick", "New activations", "Concurrent activations"])\
                     .to_csv(concurrent_activations_file, index=False)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")

    parser.add_argument("-b", "--bailout", required=False, type=int, default=None)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-e", "--buffer_entry_threshold", required=True, type=ActivationValue)
    parser.add_argument("-f", "--buffer_pruning_threshold", required=True, type=ActivationValue)
    parser.add_argument("-l", "--length_factor", required=True, type=Length)
    parser.add_argument("-r", "--max_sphere_radius", required=True, type=Length)
    parser.add_argument("-s", "--node_decay_sigma", required=True, type=float)
    parser.add_argument("-t", "--run_for_ticks", required=True, type=int)
    parser.add_argument("-w", "--buffer_size_limit", required=True, type=int)
    parser.add_argument("-U", "--use_prepruned", action="store_true")

    args = parser.parse_args()

    main(max_sphere_radius=args.max_sphere_radius,
         distance_type_name=args.distance_type,
         length_factor=args.length_factor,
         buffer_size_limit=args.buffer_size_limit,
         buffer_entry_threshold=args.buffer_entry_threshold,
         buffer_pruning_threshold=args.buffer_pruning_threshold,
         run_for_ticks=args.run_for_ticks,
         sigma=args.node_decay_sigma,
         bailout=args.bailout,
         use_prepruned=args.use_prepruned)
    logger.info("Done!")

    Emailer(Preferences.email_connection_details_path).send_email(
        f"Done running {path.basename(__file__)} with {args.pruning_length} pruning.",
        Preferences.target_email_address)
