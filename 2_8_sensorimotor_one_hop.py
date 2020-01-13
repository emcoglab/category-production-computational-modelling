#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Model responses to Briony's category production categories using a one-hop sensorimotor model.
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
from itertools import count
from os import path, makedirs

from pandas import DataFrame

from category_production.category_production import CategoryProduction
from ldm.corpus.tokenising import modified_word_tokenize
from ldm.utils.maths import DistanceType
from model.basic_types import ActivationValue, Length
from model.events import ItemEnteredBufferEvent, ItemActivatedEvent
from model.naïve_sensorimotor import SensorimotorOneHopComponent
from model.sensorimotor_component import NormAttenuationStatistic
from model.utils.email import Emailer
from model.utils.file import comment_line_from_str
from model.version import VERSION
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s:l.%(lineno)d | %(message)s'
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
         buffer_capacity: int,
         accessible_set_capacity: int,
         buffer_threshold: ActivationValue,
         accessible_set_threshold: ActivationValue,
         node_decay_median: float,
         node_decay_sigma: float,
         use_prepruned: bool = False,
         ):

    distance_type = DistanceType.from_name(distance_type_name)
    norm_attenuation_statistic = NormAttenuationStatistic.Prevalence
    # Once a node is fully activated, that's enough.
    activation_cap = FULL_ACTIVATION

    response_dir = path.join(Preferences.output_dir,
                             "Category production",
                             f"Sensorimotor one-hop {VERSION}",
                             f"{distance_type.name} length {length_factor} attenuate {norm_attenuation_statistic.name}",
                             f"max-r {max_sphere_radius};"
                                f" n-decay-median {node_decay_median};"
                                f" n-decay-sigma {node_decay_sigma};"
                                f" as-θ {accessible_set_threshold};"
                                f" as-cap {accessible_set_capacity:,};"
                                f" buff-θ {buffer_threshold};"
                                f" buff-cap {buffer_capacity}")
    if not path.isdir(response_dir):
        logger.warning(f"{response_dir} directory does not exist; making it.")
        makedirs(response_dir)

    # If we're using the prepruned version, we can risk using the cache too
    cp = CategoryProduction()
    sc = SensorimotorOneHopComponent(
        distance_type=distance_type,
        length_factor=length_factor,
        max_sphere_radius=max_sphere_radius,
        node_decay_lognormal_median=node_decay_median,
        node_decay_lognormal_sigma=node_decay_sigma,
        buffer_capacity=buffer_capacity,
        buffer_threshold=buffer_threshold,
        activation_cap=activation_cap,
        accessible_set_threshold=accessible_set_threshold,
        norm_attenuation_statistic=norm_attenuation_statistic,
        use_prepruned=use_prepruned,
        accessible_set_capacity=accessible_set_capacity,
    )

    sc.save_model_spec(response_dir)

    for category_label in cp.category_labels_sensorimotor:

        model_responses_path = path.join(response_dir, f"responses_{category_label}.csv")

        csv_comments = []

        # Only run the TSA if we've not already done it
        if path.exists(model_responses_path):
            logger.info(f"{model_responses_path} exists, skipping.")
            continue

        sc.reset()

        # Record topology
        csv_comments.append(f"Running sensorimotor spreading activation (v{VERSION}) using parameters:")
        csv_comments.append(f"\tlength_factor = {length_factor:_}")
        csv_comments.append(f"\t      pruning = {max_sphere_radius}")
        csv_comments.append(f"\t  node decay σ = {node_decay_sigma} (σ * lf = {node_decay_sigma * length_factor})")

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
        for tick in count(start=0):

            logger.info(f"Clock = {tick}")
            tick_events = sc.tick()

            activation_events = [e for e in tick_events if isinstance(e, ItemActivatedEvent)]

            for activation_event in activation_events:
                model_response_entries.append((
                    sc.idx2label[activation_event.item],                   # RESPONSE
                    activation_event.item,                                 # NODE_ID
                    activation_event.activation,                           # ACTIVATION
                    activation_event.time,                                 # TICK_ON_WHICH_ACTIVATED
                    isinstance(activation_event, ItemEnteredBufferEvent),  # ENTERED_BUFFER
                ))

            # Break when there are no impulses remaining on-route
            if sc.scheduled_activation_count() == 0:
                csv_comments.append(f"No further scheduled activations after {tick} ticks")
                logger.info(f"No further scheduled activations after {tick} ticks")
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
        with open(model_responses_path, mode="w", encoding="utf-8") as output_file:
            # Write comments
            for comment in csv_comments:
                output_file.write(comment_line_from_str(comment))
            # Write data
            model_responses_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--accessible_set_threshold", required=True, type=ActivationValue)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-e", "--buffer_threshold", required=True, type=ActivationValue)
    parser.add_argument("-l", "--length_factor", required=True, type=Length)
    parser.add_argument("-m", "--node_decay_median", required=True, type=float)
    parser.add_argument("-r", "--max_sphere_radius", required=True, type=Length)
    parser.add_argument("-s", "--node_decay_sigma", required=True, type=float)
    parser.add_argument("-w", "--buffer_capacity", required=True, type=int)
    parser.add_argument("-c", "--accessible_set_capacity", required=True, type=int)
    parser.add_argument("-U", "--use_prepruned", action="store_true")

    args = parser.parse_args()

    main(max_sphere_radius=args.max_sphere_radius,
         distance_type_name=args.distance_type,
         length_factor=args.length_factor,
         buffer_capacity=args.buffer_capacity,
         accessible_set_capacity=args.accessible_set_capacity,
         accessible_set_threshold=args.accessible_set_threshold,
         buffer_threshold=args.buffer_threshold,
         node_decay_median=args.node_decay_median,
         node_decay_sigma=args.node_decay_sigma,
         use_prepruned=args.use_prepruned)
    logger.info("Done!")

    Emailer(Preferences.email_connection_details_path).send_email(
        f"Done running {path.basename(__file__)} with radius {args.max_sphere_radius}.",
        Preferences.target_email_address)
