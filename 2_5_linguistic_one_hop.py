#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Spreading activation on ngram graphs.
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

from framework.category_production.category_production import CategoryProduction
from framework.cli.lookups import get_corpus_from_name, get_model_from_params
from framework.cognitive_model.ldm.corpus.indexing import FreqDist
from framework.cognitive_model.ldm.corpus.tokenising import modified_word_tokenize
from framework.cognitive_model.ldm.model.base import DistributionalSemanticModel
from framework.cognitive_model.components import FULL_ACTIVATION
from framework.cognitive_model.linguistic_components import LinguisticComponent
from framework.cognitive_model.linguistic_propagator import LinguisticOneHopPropagator
from framework.cognitive_model.version import VERSION
from framework.cognitive_model.basic_types import ActivationValue
from framework.cognitive_model.events import ItemActivatedEvent
from framework.cognitive_model.utils.file import comment_line_from_str
from framework.cognitive_model.utils.logging import logger
from framework.cognitive_model.preferences.preferences import Preferences
from framework.cli.job import LinguisticOneHopJobSpec
from framework.evaluation.column_names import ITEM_ID, RESPONSE, ACTIVATION, TICK_ON_WHICH_ACTIVATED


def main(n_words: int,
         corpus_name: str,
         model_name: str,
         radius: int,
         length_factor: int,
         firing_threshold: ActivationValue,
         node_decay_factor: float,
         edge_decay_sd: float,
         accessible_set_threshold: ActivationValue,
         accessible_set_capacity: int,
         impulse_pruning_threshold: ActivationValue,
         ):

    corpus = get_corpus_from_name(corpus_name)
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distributional_model: DistributionalSemanticModel = get_model_from_params(corpus, freq_dist, model_name, radius)

    job_spec = LinguisticOneHopJobSpec(
        model_name=distributional_model.name, model_radius=radius,
        corpus_name=distributional_model.corpus_meta.name,
        distance_type=None, n_words=n_words,
        firing_threshold=firing_threshold, length_factor=length_factor,
        pruning_type=None, pruning=None,
        node_decay_factor=node_decay_factor, edge_decay_sd=edge_decay_sd,
        accessible_set_threshold=accessible_set_threshold, accessible_set_capacity=accessible_set_capacity,
        use_activation_cap=False,
        impulse_pruning_threshold=impulse_pruning_threshold,
        run_for_ticks=None, bailout=None,
    )

    response_dir: Path = Path(Preferences.output_dir,
                              "Category production",
                              job_spec.output_location_relative())

    if not response_dir.is_dir():
        logger.warning(f"{response_dir} directory does not exist; making it.")
        response_dir.mkdir(parents=True)

    job_spec.save(in_location=response_dir)

    cp = CategoryProduction()
    lc = LinguisticComponent(
        propagator=LinguisticOneHopPropagator(
            length_factor=length_factor,
            n_words=n_words,
            distributional_model=distributional_model,
            distance_type=None,
            node_decay_factor=node_decay_factor,
            edge_decay_sd=edge_decay_sd,
            edge_pruning=None,
            edge_pruning_type=None,
        ),
        firing_threshold=firing_threshold,
        activation_cap=None,
        accessible_set_threshold=accessible_set_threshold,
        accessible_set_capacity=accessible_set_capacity,
    )

    for category_label in cp.category_labels:

        model_responses_path = Path(response_dir, f"responses_{category_label}_{n_words:,}.csv")

        csv_comments = []

        # Only run the TSA if we've not already done it
        if model_responses_path.exists():
            logger.info(f"{model_responses_path} exists, skipping.")
            continue

        logger.info(f"Running spreading activation for category {category_label}")

        lc.reset()

        csv_comments.append(f"Running spreading activation (v{VERSION}) using parameters:")
        csv_comments.extend(job_spec.csv_comments())
        if lc.propagator.graph.is_connected():
            csv_comments.append(f"\t      connected = yes")
        else:
            csv_comments.append(f"\t      connected = no")
            csv_comments.append(f"\t        orphans = {'yes' if lc.propagator.graph.has_orphaned_nodes() else 'no'}")

        # Do the spreading activation
        initial_activation: ActivationValue = FULL_ACTIVATION

        # If the category has a single label, activate it
        if category_label in lc.available_labels:
            logger.info(f"Running spreading activation for category {category_label}")
            lc.propagator.activate_item_with_label(category_label, initial_activation)

        # If the category has no single label, activate all constituent words
        else:
            category_words = [word
                              for word in modified_word_tokenize(category_label)
                              if word not in cp.ignored_words
                              # Ignore words which aren't available: activate all words we can
                              and word in lc.available_labels]
            logger.info(f"Running spreading activation for category {category_label}"
                        f" (activating individual words: {', '.join(category_words)})")
            if category_words:
                # Divide activation among multi-word categories
                logger.info(f"Dividing activation of multi-word category {len(category_words)} ways")
                csv_comments.append(f"Dividing activation of multi-word category {len(category_words)} ways")
                initial_activation /= len(category_words)
                lc.propagator.activate_items_with_labels(category_words, initial_activation)

        model_response_entries = []
        for tick in count(start=0):

            logger.info(f"Clock = {tick}")
            events = lc.tick()
            firing_events = (e for e in events if isinstance(e, ItemActivatedEvent) and e.fired)

            for event in firing_events:
                model_response_entries.append((
                    lc.propagator.idx2label[event.item.idx],
                    event.item.idx,
                    event.activation,
                    event.time))

            # Break when there are no impulses remaining on-route
            if lc.propagator.scheduled_activation_count() == 0:
                csv_comments.append(f"No further schedule activations after {tick} ticks")
                logger.info(f"No further scheduled activations after {tick} ticks")
                break

        model_responses_df = DataFrame(model_response_entries, columns=[
            RESPONSE,
            ITEM_ID,
            ACTIVATION,
            TICK_ON_WHICH_ACTIVATED,
        ]).sort_values([TICK_ON_WHICH_ACTIVATED, ITEM_ID])

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
    parser.add_argument("--accessible_set_capacity", required=True)
    parser.add_argument("--corpus_name", required=True, type=str)
    parser.add_argument("--firing_threshold", required=True, type=ActivationValue)
    parser.add_argument("--impulse_pruning_threshold", required=True, type=ActivationValue)
    parser.add_argument("--length_factor", required=True, type=int)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--node_decay_factor", required=True, type=float)
    parser.add_argument("--radius", required=True, type=int)
    parser.add_argument("--edge_decay_sd", required=True, type=float)
    parser.add_argument("--words", type=int, required=True)
    # Unused, just here for interface matching with 2_3
    parser.add_argument("--bailout", required=False, default=0, type=int)
    parser.add_argument("--run_for_ticks", required=False, default=1000, type=int)

    args = parser.parse_args()

    main(n_words=args.words,
         corpus_name=args.corpus_name,
         model_name=args.model_name,
         radius=args.radius,
         length_factor=args.length_factor,
         firing_threshold=args.firing_threshold,
         node_decay_factor=args.node_decay_factor,
         edge_decay_sd=args.edge_decay_sd,
         accessible_set_capacity=int(args.accessible_set_capacity) if args.accessible_set_capacity != 'None' else None,
         accessible_set_threshold=args.accessible_set_threshold,
         impulse_pruning_threshold=args.impulse_pruning_threshold,
         )
    logger.info("Done!")
