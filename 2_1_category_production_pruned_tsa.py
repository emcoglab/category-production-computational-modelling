#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Model responses to Briony's category production categories.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""
import argparse
import sys
from pathlib import Path

from pandas import DataFrame

from category_production.category_production import CategoryProduction
from cli.lookups import get_corpus_from_name, get_model_from_params
from ldm.corpus.indexing import FreqDist
from ldm.corpus.tokenising import modified_word_tokenize
from ldm.model.base import DistributionalSemanticModel
from ldm.utils.maths import DistanceType
from model.components import FULL_ACTIVATION
from model.linguistic_propagator import LinguisticPropagator
from model.utils.job import LinguisticPropagationJobSpec
from model.version import VERSION
from model.basic_types import ActivationValue
from model.events import ItemActivatedEvent
from model.linguistic_components import LinguisticComponent
from model.graph import EdgePruningType
from model.utils.email import Emailer
from model.utils.file import comment_line_from_str
from model.utils.logging import logger
from preferences import Preferences


# Results DataFrame column names
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "ActivationValue"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"


def main(n_words: int,
         prune_percent: int,
         corpus_name: str,
         model_name: str,
         radius: int,
         distance_type_name: str,
         length_factor: int,
         firing_threshold: float,
         node_decay_factor: float,
         edge_decay_sd_factor: float,
         impulse_pruning_threshold: float,
         run_for_ticks: int,
         bailout: int,
         ):

    if prune_percent == 0:
        prune_percent = None

    corpus = get_corpus_from_name(corpus_name)
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distance_type = DistanceType.from_name(distance_type_name)
    distributional_model: DistributionalSemanticModel = get_model_from_params(corpus, freq_dist, model_name, radius)

    job_spec = LinguisticPropagationJobSpec(
        model_name=distributional_model.name, model_radius=radius, corpus_name=distributional_model.corpus_meta.name,
        distance_type=distance_type, n_words=n_words,
        firing_threshold=firing_threshold, length_factor=length_factor,
        pruning_type=EdgePruningType.Percent, pruning=prune_percent,
        node_decay_factor=node_decay_factor, edge_decay_sd=edge_decay_sd_factor,
        impulse_pruning_threshold=impulse_pruning_threshold,
        run_for_ticks=run_for_ticks, bailout=bailout,
    )

    response_dir: Path = Path(Preferences.output_dir,
                              "Category production",
                              job_spec.output_location_relative())

    if not response_dir.is_dir():
        logger.warning(f"{response_dir} directory does not exist; making it.")
        response_dir.mkdir(parents=True)

    job_spec.save(in_location=response_dir)

    lc = LinguisticComponent(
        propagator=LinguisticPropagator(
            length_factor=length_factor,
            n_words=n_words,
            distributional_model=distributional_model,
            distance_type=distance_type,
            node_decay_factor=node_decay_factor,
            edge_decay_sd_factor=edge_decay_sd_factor,
            edge_pruning=prune_percent,
            edge_pruning_type=EdgePruningType.Percent
        ),
        firing_threshold=firing_threshold,
    )

    cp = CategoryProduction()
    for category_label in cp.category_labels:

        model_responses_path = Path(response_dir,
                                    f"responses_{category_label}_{n_words:,}.csv")

        csv_comments = []

        # Only run the TSA if we've not already done it
        if model_responses_path.exists():
            logger.info(f"{model_responses_path} exists, skipping.")
            continue

        logger.info(f"Running spreading activation for category {category_label}")

        lc.reset()

        csv_comments.append(f"Running spreading activation (v{VERSION}) using parameters:")
        csv_comments.append(f"\t          words = {n_words:_}")
        csv_comments.append(f"\t    granularity = {length_factor:_}")
        if prune_percent is not None:
            csv_comments.append(f"\t        pruning = {prune_percent:.2f}%")
        csv_comments.append(f"\t       firing θ = {firing_threshold}")
        csv_comments.append(f"\t              δ = {node_decay_factor}")
        csv_comments.append(f"\t      sd_factor = {edge_decay_sd_factor}")
        csv_comments.append(f"\timpulse pruning = {impulse_pruning_threshold}")
        if lc.propagator.graph.is_connected():
            csv_comments.append(f"\t        connected = yes")
        else:
            csv_comments.append(f"\t        connected = no")
            csv_comments.append(f"\t          orphans = {'yes' if lc.propagator.graph.has_orphaned_nodes() else 'no'}")

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
            lc.propagator.activate_items_with_labels(category_words, FULL_ACTIVATION)

        model_response_entries = []
        for tick in range(0, run_for_ticks):

            logger.info(f"Clock = {tick}")
            events = lc.tick()
            firing_events = (e for e in events if isinstance(e, ItemActivatedEvent) and e.fired)

            for event in firing_events:
                model_response_entries.append((
                    lc.propagator.idx2label[event.item],
                    event.item,
                    event.activation,
                    event.time))

            # Break early if we've got a probable explosion
            if len(lc.suprathreshold_items()) > bailout:
                csv_comments.append(f"")
                csv_comments.append(f"Spreading activation ended with a bailout after {tick} ticks "
                                    f"with {len(lc.suprathreshold_items())} nodes activated.")
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
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")

    parser.add_argument("-b", "--bailout", required=True, type=int)
    parser.add_argument("-c", "--corpus_name", required=True, type=str)
    parser.add_argument("-f", "--firing_threshold", required=True, type=ActivationValue)
    parser.add_argument("-i", "--impulse_pruning_threshold", required=True, type=ActivationValue)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-m", "--model_name", required=True, type=str)
    parser.add_argument("-n", "--node_decay_factor", required=True, type=float)
    parser.add_argument("-p", "--prune_percent", required=False, type=int,
                        help="The percentage of longest edges to prune from the graph.", default=None)
    parser.add_argument("-r", "--radius", required=True, type=int)
    parser.add_argument("-s", "--edge_decay_sd_factor", required=True, type=float)
    parser.add_argument("-t", "--run_for_ticks", required=True, type=int)
    parser.add_argument("-w", "--words", type=int, required=True,
                        help="The number of words to use from the corpus. (Top n words.)")

    args = parser.parse_args()

    main(n_words=args.words,
         prune_percent=args.prune_percent,
         corpus_name=args.corpus_name,
         model_name=args.model_name,
         radius=args.radius,
         distance_type_name=args.distance_type_name,
         length_factor=args.length_factor,
         firing_threshold=args.firing_threshold,
         node_decay_factor=args.node_decay_factor,
         edge_decay_sd_factor=args.edge_decay_sd_factor,
         impulse_pruning_threshold=args.impulse_pruning_threshold,
         run_for_ticks=args.run_for_ticks,
         bailout=args.bailout)
    logger.info("Done!")

    emailer = Emailer(Preferences.email_connection_details_path)
    if args.prune_percent is not None:
        emailer.send_email(f"Done running {Path(__file__).name} with {args.words} words and {args.prune_percent:.2f}% pruning.",
                           Preferences.target_email_address)
    else:
        emailer.send_email(f"Done running {Path(__file__).name} with {args.words} words.",
                           Preferences.target_email_address)
