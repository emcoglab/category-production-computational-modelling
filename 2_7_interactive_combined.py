#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Interactive combined model script.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2020
---------------------------
"""

from sys import argv
from argparse import ArgumentParser
from pathlib import Path

from numpy import nan
from pandas import DataFrame

from os import path
from framework.cognitive_model.preferences.config import Config as ModelConfig
ModelConfig(use_config_overrides_from_file=path.join(path.dirname(__file__), "wayland_config_override.yaml"))
from framework.cognitive_model.ldm.preferences.config import Config as LDMConfig
LDMConfig(use_config_overrides_from_file=path.join(path.dirname(__file__), "wayland_config_override.yaml"))
from framework.cognitive_model.sensorimotor_norms.config.config import Config as SMConfig
SMConfig(use_config_overrides_from_file=path.join(path.dirname(__file__), "wayland_config_override.yaml"))

from framework.category_production.category_production import CategoryProduction
from framework.cognitive_model.ldm.corpus.tokenising import modified_word_tokenize
from framework.cognitive_model.ldm.utils.maths import DistanceType
from framework.cognitive_model.basic_types import ActivationValue, Component, Length
from framework.cognitive_model.combined_cognitive_model import InteractiveCombinedCognitiveModel
from framework.cognitive_model.components import FULL_ACTIVATION
from framework.cognitive_model.events import ItemActivatedEvent, ItemEnteredBufferEvent
from framework.cognitive_model.linguistic_components import LinguisticComponent
from framework.cognitive_model.sensorimotor_components import SensorimotorComponent
from framework.cognitive_model.attenuation_statistic import AttenuationStatistic
from framework.cognitive_model.utils.logging import logger
from framework.cognitive_model.version import VERSION
from framework.cognitive_model.preferences.preferences import Preferences
from framework.cli.job import InteractiveCombinedJobSpec, LinguisticPropagationJobSpec, SensorimotorPropagationJobSpec

# Results DataFrame column names
RESPONSE = "Response"
ITEM_ID = "Item ID"
COMPONENT = "Component"
ACTIVATION = "Activation"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"
ENTERED_BUFFER = "Item entered WM buffer"


def main(job_spec: InteractiveCombinedJobSpec, use_prepruned: bool):

    response_dir: Path = Path(Preferences.output_dir,
                              "Category production",
                              job_spec.output_location_relative())

    if not response_dir.is_dir():
        logger.warning(f"{response_dir} directory does not exist; making it.")
        response_dir.mkdir(parents=True)

    job_spec.save(in_location=response_dir)

    model = InteractiveCombinedCognitiveModel(
        sensorimotor_component=(job_spec.sensorimotor_spec.to_component_prepruned(SensorimotorComponent)
                                if use_prepruned
                                else job_spec.sensorimotor_spec.to_component(SensorimotorComponent)),
        linguistic_component=job_spec.linguistic_spec.to_component(LinguisticComponent),
        lc_to_smc_delay=job_spec.lc_to_smc_delay,
        smc_to_lc_delay=job_spec.smc_to_lc_delay,
        inter_component_attenuation=job_spec.inter_component_attenuation,
        buffer_threshold=job_spec.buffer_threshold,
        buffer_capacity_linguistic_items=job_spec.buffer_capacity_linguistic_items,
        buffer_capacity_sensorimotor_items=job_spec.buffer_capacity_sensorimotor_items,
    )

    model.mapping.save_to(directory=response_dir)

    cp = CategoryProduction()

    for category_label in cp.category_labels:

        model_responses_path = Path(response_dir, f"responses_{category_label}.csv")
        accessible_set_path  = Path(response_dir, f"accessible_set_{category_label}.csv")

        # Only run the TSA if we've not already done it
        if model_responses_path.exists():
            logger.info(f"{model_responses_path} exists, skipping.")
            continue

        logger.info(f"Running spreading activation for category {category_label}")

        model.reset()

        csv_comments = [f"Running sensorimotor spreading activation (v{VERSION}) using parameters:"]
        csv_comments.extend(job_spec.csv_comments())

        # Activate linguistic item(s) ONLY, since stimuli were presented as words
        if category_label in model.linguistic_component.available_labels:
            logger.info(f"Activating {category_label} in linguistic component")
            model.linguistic_component.propagator.activate_item_with_label(category_label, FULL_ACTIVATION)
        else:
            category_words = [word
                              for word in modified_word_tokenize(category_label)
                              if word not in cp.ignored_words
                              # Ignore words which aren't available: activate all words we can
                              and word in model.linguistic_component.available_labels]
            if category_words:
                logger.info(f"Activating individual words {category_words} in linguistic component")
                model.linguistic_component.propagator.activate_items_with_labels(
                    category_words,
                    # Divide activation among multi-word categories
                    FULL_ACTIVATION / len(category_words))

        model_response_entries = []
        # Initialise list of concurrent activations which will be nan-populated if the run ends early
        accessible_set_this_category_linguistic = [nan] * job_spec.run_for_ticks
        accessible_set_this_category_sensorimotor = [nan] * job_spec.run_for_ticks

        for tick in range(0, job_spec.run_for_ticks):

            logger.info(f"Clock = {model.clock}")

            tick_events = model.tick()

            activation_events = [e for e in tick_events if isinstance(e, ItemActivatedEvent)]

            accessible_set_this_category_linguistic[tick] = len(model.linguistic_component.accessible_set)
            accessible_set_this_category_sensorimotor[tick] = len(model.sensorimotor_component.accessible_set)

            for activation_event in activation_events:
                label = (model.sensorimotor_component
                         if activation_event.item.component == Component.sensorimotor
                         else model.linguistic_component
                         ).propagator.idx2label[activation_event.item.idx]
                model_response_entries.append((
                    label,                                                 # RESPONSE
                    activation_event.item.idx,                             # ITEM_ID
                    activation_event.item.component.name,                  # COMPONENT
                    activation_event.activation,                           # ACTIVATION
                    activation_event.time,                                 # TICK_ON_WHICH_ACTIVATED
                    isinstance(activation_event, ItemEnteredBufferEvent),  # ENTERED_BUFFER
                ))

            if job_spec.bailout is not None and (
                    len(model.linguistic_component.accessible_set) > job_spec.bailout
                    or len(model.sensorimotor_component.accessible_set) > job_spec.bailout):
                bailout_message = (f"Spreading activation ended with a bailout after {tick} ticks"
                                   f" with {len(model.linguistic_component.accessible_set)} items"
                                   f" activated in the linguistic component and"
                                   f" {len(model.sensorimotor_component.accessible_set)} items"
                                   f" activated in the sensorimotor component.")

                csv_comments.append(f"")
                csv_comments.append(bailout_message)
                logger.warning(bailout_message)
                break

        model_response_df = DataFrame(model_response_entries, columns=[
            RESPONSE,
            ITEM_ID,
            COMPONENT,
            ACTIVATION,
            TICK_ON_WHICH_ACTIVATED,
            ENTERED_BUFFER,
        ]).sort_values([TICK_ON_WHICH_ACTIVATED, COMPONENT, ITEM_ID])

        # Record model output
        with open(model_responses_path, mode="w", encoding="utf-8") as output_file:
            model_response_df.to_csv(output_file, index=False)

        # Record accessible set size
        with open(accessible_set_path, mode="w", encoding="utf-8") as accessible_set_file:
            DataFrame.from_records([[category_label + " (linguistic)"] + accessible_set_this_category_linguistic,
                                    [category_label + " (sensorimotor)"] + accessible_set_this_category_sensorimotor])\
                     .to_csv(accessible_set_file, index=False, header=False)


if __name__ == '__main__':
    logger.info("Running %s" % " ".join(argv))

    parser = ArgumentParser(description="Run interactive combined model.")

    parser.add_argument("--linguistic_accessible_set_threshold", required=True, type=ActivationValue)
    parser.add_argument("--linguistic_accessible_set_capacity", required=False, type=int)
    parser.add_argument("--linguistic_corpus_name", required=True, type=str)
    parser.add_argument("--linguistic_firing_threshold", required=True, type=ActivationValue)
    parser.add_argument("--linguistic_impulse_pruning_threshold", required=True, type=ActivationValue)
    parser.add_argument("--linguistic_length_factor", required=True, type=int)
    parser.add_argument("--linguistic_model_name", required=True, type=str)
    parser.add_argument("--linguistic_node_decay_factor", required=True, type=float)
    parser.add_argument("--linguistic_radius", required=True, type=int)
    parser.add_argument("--linguistic_edge_decay_sd_factor", required=True, type=float)
    parser.add_argument("--linguistic_words", type=int, required=True)

    parser.add_argument("--sensorimotor_accessible_set_threshold", required=True, type=ActivationValue)
    parser.add_argument("--sensorimotor_accessible_set_capacity", required=False, type=int)
    parser.add_argument("--sensorimotor_distance_type", required=True, type=str)
    parser.add_argument("--sensorimotor_length_factor", required=True, type=Length)
    parser.add_argument("--sensorimotor_node_decay_median", required=True, type=float)
    parser.add_argument("--sensorimotor_node_decay_sigma", required=True, type=float)
    parser.add_argument("--sensorimotor_max_sphere_radius", required=True, type=float)
    parser.add_argument("--sensorimotor_use_prepruned", action="store_true")
    parser.add_argument("--sensorimotor_attenuation", required=True, type=str, choices=[n.name for n in AttenuationStatistic])
    # We have to add this argument to make the interface compatible, but we always use the BrEng translation
    parser.add_argument("--sensorimotor_use_breng_translation", action="store_true")

    parser.add_argument("--buffer_threshold", required=True, type=ActivationValue)
    parser.add_argument("--buffer_capacity_linguistic_items", required=True, type=int)
    parser.add_argument("--buffer_capacity_sensorimotor_items", required=True, type=int)
    parser.add_argument("--lc_to_smc_delay", required=True, type=int)
    parser.add_argument("--smc_to_lc_delay", required=True, type=int)
    parser.add_argument("--inter_component_attenuation", required=True, type=float)
    parser.add_argument("--bailout", required=False, default=0, type=int)
    parser.add_argument("--run_for_ticks", required=True, type=int)

    args = parser.parse_args()

    if not args.sensorimotor_use_breng_translation:
        logger.warning("BrEng translation will always be used in the interactive model.")

    main(
        job_spec=InteractiveCombinedJobSpec(
            linguistic_spec=LinguisticPropagationJobSpec(
                accessible_set_threshold=args.linguistic_accessible_set_threshold,
                accessible_set_capacity=args.linguistic_accessible_set_capacity,
                corpus_name=args.linguistic_corpus_name,
                firing_threshold=args.linguistic_firing_threshold,
                impulse_pruning_threshold=args.linguistic_impulse_pruning_threshold,
                length_factor=args.linguistic_length_factor,
                model_name=args.linguistic_model_name,
                node_decay_factor=args.linguistic_node_decay_factor,
                model_radius=args.linguistic_radius,
                edge_decay_sd=args.linguistic_edge_decay_sd_factor,
                n_words=args.linguistic_words,
                pruning=None,
                pruning_type=None,
                bailout=args.bailout,
                run_for_ticks=args.run_for_ticks
            ),
            sensorimotor_spec=SensorimotorPropagationJobSpec(
                accessible_set_threshold=args.sensorimotor_accessible_set_threshold,
                accessible_set_capacity=args.sensorimotor_accessible_set_capacity,
                distance_type=DistanceType.from_name(args.sensorimotor_distance_type),
                length_factor=args.sensorimotor_length_factor,
                node_decay_median=args.sensorimotor_node_decay_median,
                node_decay_sigma=args.sensorimotor_node_decay_sigma,
                attenuation_statistic=AttenuationStatistic.from_slug(args.sensorimotor_attenuation),
                max_radius=args.sensorimotor_max_sphere_radius,
                use_breng_translation=True,
                bailout=args.bailout,
                run_for_ticks=args.run_for_ticks,
            ),
            buffer_threshold=args.buffer_threshold,
            buffer_capacity_linguistic_items=args.buffer_capacity_linguistic_items,
            buffer_capacity_sensorimotor_items=args.buffer_capacity_sensorimotor_items,
            lc_to_smc_delay=args.lc_to_smc_delay,
            smc_to_lc_delay=args.smc_to_lc_delay,
            inter_component_attenuation=args.inter_component_attenuation,
            run_for_ticks=args.run_for_ticks,
            bailout=args.bailout,
        ),
        use_prepruned=args.sensorimotor_use_prepruned)

    logger.info("Done!")
