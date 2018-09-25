"""
===========================
Compare model to Briony's category production actual responses.

Pass the parent location to a bunch of results.
TODO: This is just a temporary way to use this script. Should be more flexible/automatable.
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
import glob
import logging
import re
import sys
from collections import defaultdict
from os import path
from typing import Dict, DefaultDict

from numpy import nan
from pandas import read_csv, DataFrame
from scipy.stats import spearmanr

from category_production.category_production import CategoryProduction
from category_production.category_production import ColNames as CPColNames
from ldm.core.corpus.indexing import FreqDist
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.component import ItemActivatedEvent
from model.utils.exceptions import ParseError
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


# Results DataFrame column names
# TODO: repeated values 😕
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "Activation"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"
TTFA = "TTFA"

MIN_FIRST_RANK_FREQ = 4


def main_in_path(results_dir: str):
    n_words = interpret_path(results_dir)

    # Load main data

    corpus = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    cp = CategoryProduction()
    linguistic_model_vocab = set(freq_dist.most_common_tokens(n_words))

    # Vectors for statistics of interest, to be computed over all categories

    # Corresponding time-to-activation of member nodes from category seed
    first_rank_tta = []

    # Of responses which were produced by both model and in the data, what are the ordinals of the (first occurrences of
    #  the) responses?
    mean_response_ranks_data = []
    time_to_activation_model = []

    category_comparisons = []

    # per-category stats

    corrs_mean_rank_vs_tta = []

    # Main dataframe holds category production data and model response data
    main_dataframe: DataFrame = cp.data.copy()

    # Drop precomputed distance measures
    main_dataframe.drop(['LgSUBTLWF', 'Sensorimotor', 'Linguistic'], axis=1, inplace=True)

    # Add model TTFA column
    model_ttfas: Dict[str, DefaultDict[str, int]] = dict()   # category -> response -> TTFA
    for category in cp.category_labels:
        model_ttfas[category] = get_model_ttfas_for_category(category, results_dir, n_words)
    main_dataframe[TTFA] = main_dataframe.apply(lambda row: model_ttfas[row[CPColNames.Category]][row[CPColNames.Response]], axis=1)

    # Drop rows corresponding to responses which weren't produced by the model
    main_dataframe = main_dataframe[main_dataframe[TTFA].notnull()]

    # Compute overall stats

    # The Pearson's correlation over all categories between average first-response RT (for responses with
    # first-rank frequency ≥4) and the time to the first activation (TTFA) within the model.

    # First rank frequency cutoff
    first_rank_frequent_data = main_dataframe[main_dataframe[CPColNames.FirstRankFrequency] >= MIN_FIRST_RANK_FREQ]
    n_first_rank_frequent = first_rank_frequent_data.shape[0]

    first_rank_frequent_corr_rt_vs_ttfa = first_rank_frequent_data[CPColNames.MeanZRT].corr(first_rank_frequent_data[TTFA], method='pearson')

    # The average (over categories) Spearman's correlation (over responses) between the mean rank of the response and
    # the time to first activation in the model.

    correlations_per_category = (
        main_dataframe
        .groupby(CPColNames.Category)
        # Correlate MeanRank and TTFA
        [[CPColNames.MeanRank, TTFA]].corr(method='spearman')
        # .corr gives the 2x2 correlation matrix for these two variables, but we just want the off-diagonal entries
        .iloc[0::2, -1])

    average_corr_meanrank_vs_ttfa = correlations_per_category.mean()
    sem_corr_meanrank_vs_ttfa = correlations_per_category.sem()

    for category_label in cp.category_labels:

        # Skip the check if the category won't be in the network
        if category_label not in linguistic_model_vocab:
            continue

        # Load MODEL responses
        try:
            model_responses_path = path.join(
                results_dir,
                f"responses_{category_label}_{n_words:,}.csv")
            with open(model_responses_path, mode="r", encoding="utf-8") as model_responses_file:
                model_responses_df = read_csv(model_responses_file, header=0, comment="#", index_col=False)

        # Skip any we don't have yet
        except FileNotFoundError as e:
            logger.warning(f"File not found: {e.filename}")
            continue

        # Collect ACTUAL responses

        # Dictionary of differently-ordered lists of words
        actual_response_words = [response
                                 for response in cp.responses_for_category(category_label, single_word_only=True)
                                 if response in linguistic_model_vocab]
        corpus_coverage_this_category = len(actual_response_words) / len(cp.responses_for_category(category_label, single_word_only=True))

        model_response_entries = [
            ItemActivatedEvent(label=row[RESPONSE], activation=row[ACTIVATION], time_activated=row[TICK_ON_WHICH_ACTIVATED])
            for row_i, row in model_responses_df.sort_values(by=TICK_ON_WHICH_ACTIVATED).iterrows()
        ]

        # Get overlap
        model_response_overlap_entries = []
        for model_response in model_response_entries:
            # Only interested in overlap
            if model_response.label not in actual_response_words:
                continue
            # Only interested in first among repeated entries
            if model_response.label in [existing_mr.label for existing_mr in model_response_overlap_entries]:
                continue
            model_response_overlap_entries.append(model_response)

        # Model–data comparisons vectors for individual categories

        # model response vector will contain ticks on which the entry was (first) activated
        times_to_first_activation_in_model = [
            common_entry.time_activated
            for common_entry in model_response_overlap_entries
        ]
        # production frequency vector will contain the production frequency
        production_frequencies_in_data = [
            cp.data_for_category_response_pair(category_label, common_entry.label, CPColNames.ProductionFrequency)
            for common_entry in model_response_overlap_entries
        ]
        # mean rank vector will contain mean ranks
        mean_ranks = [
            cp.data_for_category_response_pair(category_label, common_entry.label, CPColNames.MeanRank)
            for common_entry in model_response_overlap_entries
        ]

        # Comparisons over all categories

        # TODO: this stuff would be better done by building one large dataframe and then just doing the stats on that

        mean_response_ranks_data.extend(mean_ranks)
        time_to_activation_model.extend(times_to_first_activation_in_model)
        first_rank_tta.extend([
            response.tick_activated
            for response in model_response_overlap_entries
            # but only those above threshold
            if cp.data_for_category_response_pair(category_label, response.label, CPColNames.FirstRankFrequency) >= MIN_FIRST_RANK_FREQ
        ])

        # Compute statistics for this category

        # noinspection PyTypeChecker
        corr_mean_rank_vs_time_to_activation, _ = spearmanr(times_to_first_activation_in_model, mean_ranks)
        corrs_mean_rank_vs_tta.append(corr_mean_rank_vs_time_to_activation)
        # noinspection PyTypeChecker
        corr_production_freq_vs_time_to_activation, _ = spearmanr(times_to_first_activation_in_model, production_frequencies_in_data)

        category_comparisons.append((
            n_words,
            category_label,
            len(actual_response_words),
            100 * corpus_coverage_this_category,
            len(model_response_overlap_entries),
            100 * len(model_response_overlap_entries) / len(actual_response_words) if len(actual_response_words) > 0 else nan,
            str([e.label for e in model_response_overlap_entries]),
            str(times_to_first_activation_in_model),
            str(mean_ranks),
            corr_mean_rank_vs_time_to_activation,
            str(production_frequencies_in_data),
            corr_production_freq_vs_time_to_activation,
        ))

    # Paths
    per_category_stats_output_path = path.join(Preferences.results_dir, "Category production fit", f"model_effectiveness_per_category ({path.basename(results_dir)}).csv")
    overall_stats_output_path      = path.join(Preferences.results_dir, "Category production fit", f"model_effectiveness_overall ({path.basename(results_dir)}).txt")

    # Save per-category stats
    category_comparisons_df = DataFrame(category_comparisons, columns=[
        f"Number of words",
        f"Category",
        f"Actual responses in corpus",
        f"Corpus coverage of actual responses (%)",
        f"Model response overlap size",
        f"Model response overlap %",
        f"Model response overlap words",
        f"Model: time to first activation",
        f"Mean ranks",
        f"Mean rank correlation (Spearman's; positive is better fit)",
        f"Production frequencies",
        f"Production frequency correlation (Spearman's; negative is better fit)",
    ])
    with open(per_category_stats_output_path, mode="w", encoding="utf-8") as output_file:
        category_comparisons_df.to_csv(output_file, index=False)

    # Save overall stats
    with open(overall_stats_output_path, mode="w", encoding="utf-8") as output_file:
        # Correlation of first response RT with time-to-activation
        output_file.write(path.basename(results_dir) + "\n\n")
        output_file.write(f"First response RT vs TTA correlation ("
                          f"Pearson's; positive is better fit; "
                          f"FRF≥{MIN_FIRST_RANK_FREQ}; "
                          f"N = {n_first_rank_frequent}) "
                          f"= {first_rank_frequent_corr_rt_vs_ttfa}\n")
        output_file.write(f"Average mean_rank vs time-to-activation correlation ("
                          f"Spearman's; positive is better fit) "
                          f"= {average_corr_meanrank_vs_ttfa} (SEM = {sem_corr_meanrank_vs_ttfa})\n")


def interpret_path(results_dir_path: str) -> int:
    """

    :param results_dir_path:
    :return:
        n_words: int,

    """

    dir_name = path.basename(results_dir_path)

    unpruned_graph_match = re.match(re.compile(
        r"^"
        r"Category production traces \("
        r"(?P<n_words>[0-9,]+) words; "
        r"firing (?P<firing_threshold>[0-9.]+); "
        r"access (?P<access_threshold>[0-9.]+)\)$"), dir_name)
    length_pruned_graph_match = re.match(re.compile(
        r"^"
        r"Category production traces \("
        r"(?P<n_words>[0-9,]+) words; "
        r"firing (?P<firing_threshold>[0-9.]+); "
        r"access (?P<access_threshold>[0-9.]+); "
        r"longest (?P<length_pruning_percent>[0-9]+)% edges removed\)$"), dir_name)
    importance_pruned_graph_match = re.match(re.compile(
        r"^"
        r"Category production traces \("
        r"(?P<n_words>[0-9,]+) words; "
        r"firing (?P<firing_threshold>[0-9.]+); "
        r"access (?P<access_threshold>[0-9.]+); "
        r"edge importance threshold (?P<importance_pruning>[0-9]+)\)$"), dir_name)

    if unpruned_graph_match:
        n_words = int(unpruned_graph_match.group('n_words').replace(',', ''))
    elif length_pruned_graph_match:
        n_words = int(length_pruned_graph_match.group('n_words').replace(',', ''))
    elif importance_pruned_graph_match:
        n_words = int(importance_pruned_graph_match.group('n_words').replace(',', ''))
    else:
        raise ParseError(f"Could not parse \"{dir_name}\"")

    return n_words


def get_model_ttfas_for_category(category: str, results_dir: str, n_words: int) -> DefaultDict[str, int]:
    """
    Dictionary of
        response -> time to first activation
    for the specified category.

    DefaultDict gives nans where response not found

    :param category:
    :param results_dir:
    :param n_words:
    :return:
    """

    # Try to load model response
    try:
        model_responses_path = path.join(
            results_dir,
            f"responses_{category}_{n_words:,}.csv")
        with open(model_responses_path, mode="r", encoding="utf-8") as model_responses_file:
            model_responses_df: DataFrame = read_csv(model_responses_file, header=0, comment="#", index_col=False)

        ttfas = defaultdict(lambda: nan)
        for row_i, row in model_responses_df.sort_values(by=TICK_ON_WHICH_ACTIVATED).iterrows():
            activation_event = ItemActivatedEvent(label=row[RESPONSE], activation=row[ACTIVATION], time_activated=row[TICK_ON_WHICH_ACTIVATED])
            # We've sorted by activation time, so we only need to consider the first entry for each item
            if activation_event.label not in ttfas.keys():
                ttfas[activation_event.label] = activation_event.time_activated
        return ttfas

    # If the category wasn't found, there are no TTFAs
    except FileNotFoundError:
        return defaultdict(lambda: nan)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    # TODO: Permit execution in just one folder...
    # TODO: ...then batch that

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("path", type=str, help="The path in which to find the results.")
    args = parser.parse_args()

    dirs = glob.glob(path.join(args.path, "*"))

    for d in dirs:
        if path.isdir(d):
            logger.info(path.basename(d))
            main_in_path(d)

    logger.info("Done!")
