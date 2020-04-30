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
import sys
from itertools import count
from pathlib import Path

from pandas import DataFrame

from category_production.category_production import CategoryProduction
from ldm.corpus.tokenising import modified_word_tokenize
from ldm.utils.maths import DistanceType
from model.basic_types import ActivationValue, Length
from model.events import ItemEnteredBufferEvent, ItemActivatedEvent
from model.sensorimotor_components import NormAttenuationStatistic, BufferedSensorimotorComponent
from model.components import FULL_ACTIVATION
from model.sensorimotor_propagator import SensorimotorOneHopPropagator
from model.utils.file import comment_line_from_str
from model.utils.job import SensorimotorOneHopJobSpec
from model.version import VERSION
from model.utils.logging import logger
from preferences import Preferences

# Results DataFrame column names
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "Activation"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"
ENTERED_BUFFER = "Item entered WM buffer"


def main(distance_type_name: str,
         length_factor: int,
         max_sphere_radius: int,
         buffer_capacity: int,
         accessible_set_capacity: int,
         buffer_threshold: ActivationValue,
         accessible_set_threshold: ActivationValue,
         node_decay_median: float,
         node_decay_sigma: float,
         attenuation: NormAttenuationStatistic,
         use_prepruned: bool = False,
         ):

    distance_type = DistanceType.from_name(distance_type_name)
    # Once a node is fully activated, that's enough.
    activation_cap = FULL_ACTIVATION

    job_spec = SensorimotorOneHopJobSpec(
      distance_type=DistanceType.Minkowski3, length_factor=length_factor,
      max_radius=max_sphere_radius,
      buffer_threshold=buffer_threshold, buffer_capacity=buffer_capacity,
      accessible_set_threshold=accessible_set_threshold,
      accessible_set_capacity=accessible_set_capacity,
      node_decay_sigma=node_decay_sigma, node_decay_median=node_decay_median,
      attenuation_statistic=attenuation,
      run_for_ticks=None, bailout=None,
    )

    response_dir: Path = Path(Preferences.output_dir,
                              "Category production",
                              job_spec.output_location_relative())

    if not response_dir.is_dir():
        logger.warning(f"{response_dir} directory does not exist; making it.")
        response_dir.mkdir(parents=True)

    job_spec.save(in_location=response_dir)

    # If we're using the prepruned version, we can risk using the cache too
    cp = CategoryProduction()
    sc = BufferedSensorimotorComponent(
        propagator=SensorimotorOneHopPropagator(
            distance_type=distance_type,
            length_factor=length_factor,
            max_sphere_radius=max_sphere_radius,
            node_decay_lognormal_median=node_decay_median,
            node_decay_lognormal_sigma=node_decay_sigma,
            use_prepruned=use_prepruned,
        ),
        buffer_capacity=buffer_capacity,
        buffer_threshold=buffer_threshold,
        activation_cap=activation_cap,
        accessible_set_threshold=accessible_set_threshold,
        norm_attenuation_statistic=attenuation,
        accessible_set_capacity=accessible_set_capacity,
    )

    for category_label in cp.category_labels_sensorimotor:

        model_responses_path = Path(response_dir, f"responses_{category_label}.csv")

        csv_comments = []

        # Only run the TSA if we've not already done it
        if model_responses_path.exists():
            logger.info(f"{model_responses_path} exists, skipping.")
            continue

        sc.reset()

        # Record topology
        csv_comments.append(f"Running sensorimotor spreading activation (v{VERSION}) using parameters:")
        csv_comments.append(f"\t length_factor = {length_factor:_}")
        csv_comments.append(f"\t distance_type = {distance_type.name}")
        csv_comments.append(f"\t   attenuation = {attenuation.name}")
        csv_comments.append(f"\t       pruning = {max_sphere_radius}")
        csv_comments.append(f"\t  WMB capacity = {buffer_capacity}")
        csv_comments.append(f"\t   AS capacity = {accessible_set_capacity}")
        csv_comments.append(f"\t WMB threshold = {buffer_threshold}")
        csv_comments.append(f"\t  AS threshold = {accessible_set_threshold}")
        csv_comments.append(f"\t  node decay m = {node_decay_median}")
        csv_comments.append(f"\t  node decay σ = {node_decay_sigma} (σ * lf = {node_decay_sigma * length_factor})")
        csv_comments.append(f"\tactivation cap = {activation_cap}")

        # Do the spreading activation

        # If the category has a single norm, activate it
        if category_label in sc.available_labels:
            logger.info(f"Running spreading activation for category {category_label}")
            sc.propagator.activate_item_with_label(category_label, FULL_ACTIVATION)

        # If the category has no single norm, activate all constituent words
        else:
            category_words = [word
                              for word in modified_word_tokenize(category_label)
                              if word not in cp.ignored_words
                              # Ignore words which aren't available: activate all words we can
                              and word in sc.available_labels]
            logger.info(f"Running spreading activation for category {category_label}"
                        f" (activating individual words: {', '.join(category_words)})")
            sc.propagator.activate_items_with_labels(category_words, FULL_ACTIVATION)

        model_response_entries = []
        for tick in count(start=0):

            logger.info(f"Clock = {tick}")
            tick_events = sc.tick()

            activation_events = [e for e in tick_events if isinstance(e, ItemActivatedEvent)]

            for activation_event in activation_events:
                model_response_entries.append((
                    sc.propagator.idx2label[activation_event.item],                   # RESPONSE
                    activation_event.item,                                 # NODE_ID
                    activation_event.activation,                           # ACTIVATION
                    activation_event.time,                                 # TICK_ON_WHICH_ACTIVATED
                    isinstance(activation_event, ItemEnteredBufferEvent),  # ENTERED_BUFFER
                ))

            # Break when there are no impulses remaining on-route
            assert isinstance(sc.propagator, SensorimotorOneHopPropagator)
            if sc.propagator.scheduled_activation_count() == 0:
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
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--accessible_set_threshold", required=True, type=ActivationValue)
    parser.add_argument("--distance_type", required=True, type=str)
    parser.add_argument("--buffer_threshold", required=True, type=ActivationValue)
    parser.add_argument("--length_factor", required=True, type=Length)
    parser.add_argument("--node_decay_median", required=True, type=float)
    parser.add_argument("--max_sphere_radius", required=True, type=Length)
    parser.add_argument("--node_decay_sigma", required=True, type=float)
    parser.add_argument("--buffer_capacity", required=True, type=int)
    parser.add_argument("--accessible_set_capacity", required=True, type=int)
    parser.add_argument("--use_prepruned", action="store_true")
    parser.add_argument("--attenuation", required=True, type=str,
                        choices=[n.name for n in NormAttenuationStatistic])

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
         use_prepruned=args.use_prepruned,
         attenuation=NormAttenuationStatistic.from_slug(args.attenuation))
    logger.info("Done!")
