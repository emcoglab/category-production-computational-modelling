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
import logging
import sys
from os import path, makedirs

from pandas import DataFrame

from category_production.category_production import CategoryProduction
from cli.lookups import get_corpus_from_name, get_model_from_params
from ldm.corpus.indexing import FreqDist
from ldm.corpus.tokenising import modified_word_tokenize
from ldm.model.base import DistributionalSemanticModel
from ldm.utils.maths import DistanceType
from model.version import VERSION
from model.basic_types import ActivationValue
from model.events import ItemActivatedEvent
from model.linguistic_component import EdgePruningType, LinguisticComponent
from model.utils.email import Emailer
from model.utils.file import comment_line_from_str
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


# Results DataFrame column names
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "ActivationValue"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"

FULL_ACTIVATION = ActivationValue(1.0)


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

    filtered_words = set(freq_dist.most_common_tokens(n_words))

    pruning_suffix = f", longest {prune_percent}% edges removed" if prune_percent is not None else ""
    response_dir = path.join(Preferences.output_dir,
                             "Category production",
                             f"Linguistic {VERSION}",
                             f"{distributional_model.name} {distance_type.name}"
                                f" {n_words:,} words, length {length_factor}{pruning_suffix}",
                             f"firing-θ {firing_threshold};"
                                f" n-decay-f {node_decay_factor};"
                                f" e-decay-sd {edge_decay_sd_factor};"
                                f" imp-prune-θ {impulse_pruning_threshold};"
                                f" run-for {run_for_ticks};"
                                f" bail {bailout}")
    if not path.isdir(response_dir):
        logger.warning(f"{response_dir} directory does not exist; making it.")
        makedirs(response_dir)

    cp = CategoryProduction()
    lc = LinguisticComponent(
        n_words=n_words,
        distributional_model=distributional_model,
        length_factor=length_factor,
        impulse_pruning_threshold=impulse_pruning_threshold,
        edge_decay_sd_factor=edge_decay_sd_factor,
        node_decay_factor=node_decay_factor,
        firing_threshold=firing_threshold,
        edge_pruning=prune_percent,
        distance_type=distance_type,
        edge_pruning_type=EdgePruningType.Percent,
    )

    lc.save_model_spec(response_dir, {
        "Run for ticks": run_for_ticks,
        "Bailout": bailout
    })

    for category_label in cp.category_labels:

        model_responses_path = path.join(response_dir, f"responses_{category_label}_{n_words:,}.csv")

        csv_comments = []

        # Only run the TSA if we've not already done it
        if path.exists(model_responses_path):
            logger.info(f"{model_responses_path} exists, skipping.")
            continue

        logger.info(f"Running spreading activation for category {category_label}")

        lc.reset()

        csv_comments.append(f"Running spreading activation (v{VERSION}) using parameters:")
        csv_comments.append(f"\t      words = {n_words:_}")
        csv_comments.append(f"\tgranularity = {length_factor:_}")
        if prune_percent is not None:
            csv_comments.append(f"\t    pruning = {prune_percent:.2f}%")
        csv_comments.append(f"\t   firing θ = {firing_threshold}")
        csv_comments.append(f"\t          δ = {node_decay_factor}")
        csv_comments.append(f"\t  sd_factor = {edge_decay_sd_factor}")
        if lc.graph.is_connected():
            csv_comments.append(f"\t    connected = yes")
        else:
            csv_comments.append(f"\t    connected = no")
            csv_comments.append(f"\t      orphans = {'yes' if lc.graph.has_orphaned_nodes() else 'no'}")

        # If the category has a single norm, activate it
        if category_label in filtered_words:
            logger.info(f"Running spreading activation for category {category_label}")
            lc.activate_item_with_label(category_label, FULL_ACTIVATION)

        # If the category has no single norm, activate all constituent words
        else:
            category_words = [word for word in modified_word_tokenize(category_label) if word not in cp.ignored_words]
            logger.info(f"Running spreading activation for category {category_label}"
                        f" (activating individual words: {', '.join(category_words)})")
            lc.activate_items_with_labels(category_words, FULL_ACTIVATION)

        model_response_entries = []
        for tick in range(0, run_for_ticks):

            logger.info(f"Clock = {tick}")
            events = lc.tick()
            firing_events = (e for e in events if isinstance(e, ItemActivatedEvent) and e.fired)

            for event in firing_events:
                model_response_entries.append((
                    lc.idx2label[event.item],
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
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
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
        emailer.send_email(f"Done running {path.basename(__file__)} with {args.words} words and {args.prune_percent:.2f}% pruning.",
                           Preferences.target_email_address)
    else:
        emailer.send_email(f"Done running {path.basename(__file__)} with {args.words} words.",
                           Preferences.target_email_address)
