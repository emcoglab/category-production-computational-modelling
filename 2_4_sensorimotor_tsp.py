#!/Users/cai/Applications/miniconda3/bin/python
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
import sys
from pathlib import Path

from numpy import nan
from pandas import DataFrame

from category_production.category_production import CategoryProduction
from ldm.corpus.tokenising import modified_word_tokenize
from ldm.utils.maths import DistanceType

from model.sensorimotor_components import BufferedSensorimotorComponent
from model.attenuation_statistic import AttenuationStatistic
from model.components import FULL_ACTIVATION
from model.utils.job import BufferedSensorimotorPropagationJobSpec
from model.version import VERSION
from model.basic_types import ActivationValue, Length
from model.events import ItemEnteredBufferEvent, ItemActivatedEvent, BufferFloodEvent
from model.utils.file import comment_line_from_str
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
         buffer_threshold: ActivationValue,
         buffer_capacity: int,
         accessible_set_threshold: ActivationValue,
         accessible_set_capacity: int,
         run_for_ticks: int,
         node_decay_median: float,
         node_decay_sigma: float,
         attenuation: AttenuationStatistic,
         bailout: int = None,
         use_prepruned: bool = False,
         ):

    distance_type = DistanceType.from_name(distance_type_name)
    # Once a node is fully activated, that's enough.
    activation_cap = FULL_ACTIVATION

    job_spec = BufferedSensorimotorPropagationJobSpec(
        distance_type=distance_type, length_factor=length_factor, max_radius=max_sphere_radius,
        buffer_threshold=buffer_threshold, buffer_capacity=buffer_capacity,
        accessible_set_threshold=accessible_set_threshold, accessible_set_capacity=accessible_set_capacity,
        attenuation_statistic=attenuation,
        node_decay_sigma=node_decay_sigma, node_decay_median=node_decay_median,
        run_for_ticks=run_for_ticks, bailout=bailout,
    )

    response_dir: Path = Path(Preferences.output_dir,
                              "Category production",
                              job_spec.output_location_relative())

    if not response_dir.is_dir():
        logger.warning(f"{response_dir} directory does not exist; making it.")
        response_dir.mkdir(parents=True)

    job_spec.save(in_location=response_dir)

    cp = CategoryProduction()
    sc = BufferedSensorimotorComponent.from_spec(job_spec, use_prepruned=use_prepruned)

    for category_label in cp.category_labels_sensorimotor:

        accessible_set_path  = Path(response_dir, f"accessible_set_{category_label}.csv")
        buffer_floods_path   = Path(response_dir, f"buffer_floods_{category_label}.csv")
        model_responses_path = Path(response_dir, f"responses_{category_label}.csv")

        # Only run the TSA if we've not already done it
        if model_responses_path.exists():
            logger.info(f"{model_responses_path} exists, skipping.")
            continue

        sc.reset()

        csv_comments = [f"Running sensorimotor spreading activation (v{VERSION}) using parameters:"]
        csv_comments.extend(job_spec.csv_comments())

        # Do the spreading activation
        initial_activation: ActivationValue = FULL_ACTIVATION

        # If the category has a single norm, activate it
        if category_label in sc.available_labels:
            logger.info(f"Running spreading activation for category {category_label}")
            sc.propagator.activate_item_with_label(category_label, initial_activation)

        # If the category has no single norm, activate all constituent words
        else:
            category_words = [word
                              for word in modified_word_tokenize(category_label)
                              if word not in cp.ignored_words
                              # Ignore words which aren't available: activate all words we can
                              and word in sc.available_labels]
            logger.info(f"Running spreading activation for category {category_label}"
                        f" (activating individual words: {', '.join(category_words)})")
            if category_words:
                # Divide activation among multi-word categories
                logger.info(f"Dividing activation of multi-word category {len(category_words)} ways")
                csv_comments.append(f"Dividing activation of multi-word category {len(category_words)} ways")
                initial_activation /= len(category_words)
                sc.propagator.activate_items_with_labels(category_words, initial_activation)

        model_response_entries = []
        # Initialise list of concurrent activations which will be nan-populated if the run ends early
        accessible_set_this_category = [nan] * run_for_ticks
        # Buffer-floods
        buffer_floods = []
        for tick in range(0, run_for_ticks):

            logger.info(f"Clock = {tick}")
            tick_events = sc.tick()

            activation_events = [e for e in tick_events if isinstance(e, ItemActivatedEvent)]

            accessible_set_size = len(sc.accessible_set)

            accessible_set_this_category[tick] = accessible_set_size

            if any(isinstance(e, BufferFloodEvent) for e in tick_events):
                logger.warning(f"Buffer flood occurred at t={sc.propagator.clock}")
                buffer_floods.append(sc.propagator.clock)

            for activation_event in activation_events:
                model_response_entries.append((
                    sc.propagator.idx2label[activation_event.item.idx],    # RESPONSE
                    activation_event.item.idx,                             # NODE_ID
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

        # Record buffer floods
        with open(buffer_floods_path, mode="w", encoding="utf-8") as buffer_flood_file:
            flood_str = f"{len(buffer_floods)} floods"
            first_str = f"first at {min(buffer_floods)}" if len(buffer_floods) > 0 else ""
            buffer_flood_file.write(f"{flood_str},{first_str}")

        # Model ouput
        with open(model_responses_path, mode="w", encoding="utf-8") as output_file:
            # Write comments
            for comment in csv_comments:
                output_file.write(comment_line_from_str(comment))
            # Write data
            model_responses_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")

    parser.add_argument("--accessible_set_threshold", required=True, type=ActivationValue)
    parser.add_argument("--accessible_set_capacity", required=True, type=int)
    parser.add_argument("--bailout", required=False, type=int, default=None)
    parser.add_argument("--distance_type", required=True, type=str)
    parser.add_argument("--buffer_threshold", required=True, type=ActivationValue)
    parser.add_argument("--length_factor", required=True, type=Length)
    parser.add_argument("--node_decay_median", required=True, type=float)
    parser.add_argument("--node_decay_sigma", required=True, type=float)
    parser.add_argument("--max_sphere_radius", required=True, type=float)
    parser.add_argument("--run_for_ticks", required=True, type=int)
    parser.add_argument("--buffer_capacity", required=True, type=int)
    parser.add_argument("--use_prepruned", action="store_true")
    parser.add_argument("--attenuation", required=True, type=str, choices=[n.name for n in AttenuationStatistic])

    args = parser.parse_args()

    main(max_sphere_radius=args.max_sphere_radius,
         distance_type_name=args.distance_type,
         length_factor=args.length_factor,
         buffer_capacity=args.buffer_capacity,
         accessible_set_capacity=args.accessible_set_capacity,
         accessible_set_threshold=args.accessible_set_threshold,
         buffer_threshold=args.buffer_threshold,
         run_for_ticks=args.run_for_ticks,
         node_decay_median=args.node_decay_median,
         node_decay_sigma=args.node_decay_sigma,
         bailout=args.bailout,
         use_prepruned=args.use_prepruned,
         attenuation=AttenuationStatistic.from_slug(args.attenuation),
         )
    logger.info("Done!")
