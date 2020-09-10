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
from pathlib import Path

from numpy import nan
from pandas import DataFrame

from category_production.category_production import CategoryProduction
from ldm.corpus.tokenising import modified_word_tokenize
from model.basic_types import ActivationValue
from model.components import FULL_ACTIVATION
from model.events import ItemActivatedEvent
from model.linguistic_components import LinguisticComponent
from model.utils.file import comment_line_from_str
from model.utils.job import LinguisticPropagationJobSpec
from model.utils.logging import logger
from model.version import VERSION
from preferences import Preferences

# Results DataFrame column names
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "Activation"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"


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
         run_for_ticks: int,
         bailout: int,
         divide_initial_activation_for_multiword_categories: bool,
         ):

    job_spec = LinguisticPropagationJobSpec(
        model_name=model_name, model_radius=radius,
        corpus_name=corpus_name,
        distance_type=None, n_words=n_words,
        firing_threshold=firing_threshold, length_factor=length_factor,
        pruning_type=None, pruning=None,
        node_decay_factor=node_decay_factor, edge_decay_sd=edge_decay_sd,
        accessible_set_threshold=accessible_set_threshold, accessible_set_capacity=accessible_set_capacity,
        impulse_pruning_threshold=impulse_pruning_threshold,
        run_for_ticks=run_for_ticks, bailout=bailout,
    )

    response_dir: Path = Path(Preferences.output_dir,
                              "Category production",
                              job_spec.output_location_relative())
    if divide_initial_activation_for_multiword_categories:
        response_dir = Path(*response_dir.parts[:-1], response_dir.parts[-1] + " divided")

    if not response_dir.is_dir():
        logger.warning(f"{response_dir} directory does not exist; making it.")
        response_dir.mkdir(parents=True)

    job_spec.save(in_location=response_dir)

    lc = LinguisticComponent.from_spec(job_spec)

    cp = CategoryProduction()
    for category_label in cp.category_labels:

        suprathreshold_path  = Path(response_dir, f"supra_threshold_{category_label}.csv")
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

        # If the category has a single label, activate it
        if category_label in lc.available_labels:
            logger.info(f"Running spreading activation for category {category_label}")
            lc.propagator.activate_item_with_label(category_label, FULL_ACTIVATION)

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
                initial_activation = FULL_ACTIVATION
                if divide_initial_activation_for_multiword_categories:
                    # Divide activation among multi-word categories
                    logger.info(f"Dividing activation of multi-word category {len(category_words)} ways")
                    csv_comments.extend(f"Dividing activation of multi-word category {len(category_words)} ways")
                    initial_activation /= len(category_words)
                lc.propagator.activate_items_with_labels(category_words, initial_activation)

        model_response_entries = []
        # Initialise list of concurrent activations which will be nan-populated if the run ends early
        suprathreshold_this_category = [nan] * run_for_ticks
        for tick in range(0, run_for_ticks):

            logger.info(f"Clock = {tick}")
            events = lc.tick()
            firing_events = (e for e in events if isinstance(e, ItemActivatedEvent) and e.fired)

            for event in firing_events:
                model_response_entries.append((
                    lc.propagator.idx2label[event.item.idx],
                    event.item.idx,
                    event.activation,
                    event.time))

            suprathreshold_this_category[tick] = len(lc.accessible_set)

            # Break early if we've got a probable explosion
            if len(lc.accessible_set) > bailout > 0:
                csv_comments.append(f"")
                csv_comments.append(f"Spreading activation ended with a bailout after {tick} ticks "
                                    f"with {lc.accessible_set.items} nodes activated.")
                break

        with open(suprathreshold_path, mode="w", encoding="utf-8") as supratheshold_file:
            DataFrame.from_records([[category_label] + suprathreshold_this_category])\
                .to_csv(supratheshold_file, index=False, header=False)

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
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")

    parser.add_argument("--accessible_set_threshold", required=True, type=ActivationValue)
    parser.add_argument("--accessible_set_capacity", required=True)
    parser.add_argument("--bailout", required=False, default=0, type=int)
    parser.add_argument("--corpus_name", required=True, type=str)
    parser.add_argument("--firing_threshold", required=True, type=ActivationValue)
    parser.add_argument("--impulse_pruning_threshold", required=True, type=ActivationValue)
    parser.add_argument("--length_factor", required=True, type=int)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--node_decay_factor", required=True, type=float)
    parser.add_argument("--radius", required=True, type=int)
    parser.add_argument("--edge_decay_sd", required=True, type=float)
    parser.add_argument("--run_for_ticks", required=True, type=int)
    parser.add_argument("--words", type=int, required=True)
    parser.add_argument("--multiword_divide", action="store_true")

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
         run_for_ticks=args.run_for_ticks,
         bailout=args.bailout,
         divide_initial_activation_for_multiword_categories=args.multiword_divide,
         )
    logger.info("Done!")
