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

from numpy import nan
from pandas import DataFrame

from category_production.category_production import CategoryProduction
from ldm.corpus.tokenising import modified_word_tokenize
from ldm.utils.maths import DistanceType
from model.basic_types import ActivationValue, Length
from model.events import ItemEnteredBufferEvent, ItemActivatedEvent, BufferFloodEvent
from model.sensorimotor_component import SensorimotorComponent, NormAttenuationStatistic
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
ENTERED_BUFFER = "Item entered WM buffer"

FULL_ACTIVATION = ActivationValue(1.0)


def main(distance_type_name: str,
         length_factor: int,
         max_sphere_radius: int,
         buffer_size_limit: int,
         buffer_threshold: ActivationValue,
         activation_threshold: ActivationValue,
         run_for_ticks: int,
         median: float,
         sigma: float,
         use_prepruned: bool,
         bailout: int = None,
         ):

    distance_type = DistanceType.from_name(distance_type_name)
    norm_attenuation_statistic = NormAttenuationStatistic.Prevalence
    # Once a node is fully activated, that's enough.
    activation_cap = FULL_ACTIVATION

    # Output file path
    response_dir = path.join(Preferences.output_dir,
                             f"Category production traces [sensorimotor {distance_type.name}] "
                             f"r {max_sphere_radius} "
                             f"m {median}; "
                             f"s {sigma}; "
                             f"a {activation_threshold}; "
                             f"b {buffer_threshold}; "
                             f"attenuate {norm_attenuation_statistic.name}; "
                             f"rft {run_for_ticks}; "
                             f"bailout {bailout}")
    if not path.isdir(response_dir):
        logger.warning(f"{response_dir} directory does not exist; making it.")
        makedirs(response_dir)

    # If we're using the prepruned version, we can risk using the cache too
    cp = CategoryProduction(use_cache=use_prepruned)
    sc = SensorimotorComponent(
        distance_type=distance_type,
        length_factor=length_factor,
        max_sphere_radius=max_sphere_radius,
        lognormal_median=median,
        lognormal_sigma=sigma,
        buffer_size_limit=buffer_size_limit,
        buffer_threshold=buffer_threshold,
        activation_cap=activation_cap,
        activation_threshold=activation_threshold,
        norm_attenuation_statistic=norm_attenuation_statistic,
        use_prepruned=use_prepruned,
    )

    SensorimotorComponent.save_model_spec({
        "Distance type": distance_type.name,
        "Length factor": length_factor,
        "Max sphere radius": max_sphere_radius,
        "Log-normal median": median,
        "Log-normal sigma": sigma,
        "Buffer size limit": buffer_size_limit,
        "Buffer threshold": buffer_threshold,
        "Norm attenuation statistic": norm_attenuation_statistic.name,
        "Activation cap": activation_cap,
        "Activation threshold": activation_threshold,
        "Run for ticks": run_for_ticks,
        "Bailout": bailout
    }, response_dir)

    for category_label in cp.category_labels_sensorimotor:

        model_responses_path = path.join(response_dir, f"responses_{category_label}.csv")
        accessible_set_path = path.join(response_dir, f"accessible_set_{category_label}.csv")

        csv_comments = []

        # Only run the TSA if we've not already done it
        if path.exists(model_responses_path):
            logger.info(f"{model_responses_path} exists, skipping.")
            continue

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

        # If the category has a single norm, activate it
        if category_label in sc.concept_labels:
            logger.info(f"Running spreading activation for category {category_label}")
            sc.activate_item_with_label(category_label, FULL_ACTIVATION)

        # If the category has no single norm, activate all constituent words
        else:
            category_words = [word for word in modified_word_tokenize(category_label) if word not in cp.ignored_words]
            logger.info(f"Running spreading activation for category {category_label}"
                        f" (activating individual words: {', '.join(category_words)})")
            sc.activate_items_with_labels(category_words, FULL_ACTIVATION)

        model_response_entries = []
        # Initialise list of concurrent activations which will be nan-populated if the run ends early
        accessible_set_this_category = [nan] * run_for_ticks
        for tick in range(0, run_for_ticks):

            logger.info(f"Clock = {tick}")
            tick_events = sc.tick()

            activation_events = [e for e in tick_events if isinstance(e, ItemActivatedEvent)]

            accessible_set_size = len(sc.accessible_set())

            accessible_set_this_category[tick] = accessible_set_size

            flood_event = [e for e in tick_events if isinstance(e, BufferFloodEvent)][0] if len([e for e in tick_events if isinstance(e, BufferFloodEvent)]) > 0 else None
            if flood_event:
                logger.warning(f"Buffer flood occurred at t={flood_event.time}")

            for activation_event in activation_events:
                model_response_entries.append((
                    sc.idx2label[activation_event.item],                   # RESPONSE
                    activation_event.item,                                 # NODE_ID
                    activation_event.activation,                           # ACTIVATION
                    activation_event.time,                                 # TICK_ON_WHICH_ACTIVATED
                    isinstance(activation_event, ItemEnteredBufferEvent),  # ENTERED_BUFFER
                ))

            # Break early if we've got a probable explosion
            if bailout is not None and accessible_set_size > bailout:
                csv_comments.append(f"")
                csv_comments.append(f"Spreading activation ended with a bailout after {tick} ticks "
                                    f"with {accessible_set_size} nodes activated.")
                logger.warning(f"Spreading activation ended with a bailout after {tick} ticks "
                               f"with {accessible_set_size} nodes activated.")
                break

        model_responses_df = DataFrame.from_records(
            model_response_entries,
            columns=[
                RESPONSE,
                NODE_ID,
                ACTIVATION,
                TICK_ON_WHICH_ACTIVATED,
                ENTERED_BUFFER,
            ]).sort_values([TICK_ON_WHICH_ACTIVATED, NODE_ID])

        # Output results

        # Record accessible set size
        with open(accessible_set_path, mode="w", encoding="utf-8") as accessible_set_file:
            DataFrame.from_records([[category_label] + accessible_set_this_category])\
                     .to_csv(accessible_set_file, index=False, header=False)

        # Model ouput
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

    parser.add_argument("-a", "--activation_threshold", required=True, type=ActivationValue)
    parser.add_argument("-b", "--bailout", required=False, type=int, default=None)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-e", "--buffer_threshold", required=True, type=ActivationValue)
    parser.add_argument("-l", "--length_factor", required=True, type=Length)
    parser.add_argument("-m", "--node_decay_median", required=True, type=float)
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
         activation_threshold=args.activation_threshold,
         buffer_threshold=args.buffer_threshold,
         run_for_ticks=args.run_for_ticks,
         median=args.node_decay_median,
         sigma=args.node_decay_sigma,
         bailout=args.bailout,
         use_prepruned=args.use_prepruned)
    logger.info("Done!")

    Emailer(Preferences.email_connection_details_path).send_email(
        f"Done running {path.basename(__file__)} with radius {args.max_sphere_radius}.",
        Preferences.target_email_address)
