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
from typing import Dict, DefaultDict, Set, Tuple

from numpy import nan
from pandas import read_csv, DataFrame

from category_production.category_production import CategoryProduction
from category_production.category_production import ColNames as CPColNames
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

# Analysis settings
MIN_FIRST_RANK_FREQ = 4


def main_in_path(results_dir: str, category_production: CategoryProduction, available_items) -> Set[Tuple[str, str]]:
    n_words = interpret_path(results_dir)

    restricting_items = (available_items is not None)

    # region Build main dataframe

    # Main dataframe holds category production data and model response data
    main_dataframe: DataFrame = category_production.data.copy()
    # Drop precomputed distance measures
    main_dataframe.drop(['LgSUBTLWF', 'Sensorimotor', 'Linguistic'], axis=1, inplace=True)

    # Add model TTFA column
    model_ttfas: Dict[str, DefaultDict[str, int]] = dict()  # category -> response -> TTFA
    for category in category_production.category_labels:
        model_ttfas[category] = get_model_ttfas_for_category(category, results_dir, n_words)
    main_dataframe[TTFA] = main_dataframe.apply(
        lambda row: model_ttfas[row[CPColNames.Category]][row[CPColNames.Response]], axis=1)

    # Drop rows corresponding to responses which weren't produced by the model
    main_dataframe = main_dataframe[main_dataframe[TTFA].notnull()]

    # Now we can convert TTFAs to ints as there won't be null values
    main_dataframe[TTFA] = main_dataframe[TTFA].astype(int)

    # If we specified to restrict to a set of available items, do the restriction now,
    # otherwise collect what items we have for this run
    if restricting_items:
        per_category_stats_output_path = path.join(Preferences.results_dir, "Category production fit",
                                                   f"item-level data (restricted) ({path.basename(results_dir)}).csv")
        # Restrict
        main_dataframe = main_dataframe[
            # Filter on categories
            main_dataframe[CPColNames.Category].isin([c for c, r in available_items])
            # Filter on responses
            & main_dataframe[CPColNames.Response].isin([r for c, r in available_items])]
    else:
        per_category_stats_output_path = path.join(Preferences.results_dir, "Category production fit",
                                                   f"item-level data ({path.basename(results_dir)}).csv")
        # Collect
        available_items = set(main_dataframe[[CPColNames.Category, CPColNames.Response]].groupby([CPColNames.Category, CPColNames.Response]).groups.keys())

    # Save main dataframe
    main_dataframe.to_csv(per_category_stats_output_path, index=False)

    # endregion

    # region Compute overall stats

    # The Pearson's correlation over all categories between average first-response RT (for responses with
    # first-rank frequency ≥4) and the time to the first activation (TTFA) within the model.
    first_rank_frequent_data = main_dataframe[main_dataframe[CPColNames.FirstRankFrequency] >= MIN_FIRST_RANK_FREQ]
    n_first_rank_frequent = first_rank_frequent_data.shape[0]
    first_rank_frequent_corr_rt_vs_ttfa = first_rank_frequent_data[CPColNames.MeanZRT].corr(first_rank_frequent_data[TTFA], method='pearson')

    # Pearson's correlation between production frequency and ttfa
    corr_prodfreq_vs_ttfa = main_dataframe[CPColNames.ProductionFrequency].corr(main_dataframe[TTFA], method='pearson')

    # endregion

    # region Average over category stats

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

    # endregion

    # region Save stats

    # If we're restricting or not we save results in a different place
    # TODO: this is a bad way to record this logic,
    # TODO: and in general I'm not happy with the control logic of this script
    if restricting_items:
        overall_stats_output_path = path.join(Preferences.results_dir, "Category production fit",
                                              f"model_effectiveness_overall (restricted) ({path.basename(results_dir)}).txt")
    else:
        overall_stats_output_path = path.join(Preferences.results_dir, "Category production fit",
                                              f"model_effectiveness_overall ({path.basename(results_dir)}).txt")
    with open(overall_stats_output_path, mode="w", encoding="utf-8") as output_file:
        # Correlation of first response RT with time-to-activation
        output_file.write(path.basename(results_dir) + "\n\n")
        output_file.write(f"First response RT vs TTFA correlation ("
                          f"Pearson's; positive is better fit; "
                          f"FRF≥{MIN_FIRST_RANK_FREQ}; "
                          f"N = {n_first_rank_frequent}) "
                          f"= {first_rank_frequent_corr_rt_vs_ttfa}\n")
        output_file.write(f"Production frequency vs TTFA correlation ("
                          f"Pearson's; negative is better fit; "
                          f"N = {len(available_items)}) "
                          f"= {corr_prodfreq_vs_ttfa}\n")
        output_file.write(f"Average mean_rank vs TTFA correlation ("
                          f"Spearman's; positive is better fit) "
                          f"= {average_corr_meanrank_vs_ttfa} (SEM = {sem_corr_meanrank_vs_ttfa})\n")

    # endregion

    return available_items


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
            activation_event = ItemActivatedEvent(label=row[RESPONSE],
                                                  activation=row[ACTIVATION],
                                                  time_activated=row[TICK_ON_WHICH_ACTIVATED])
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

    # TODO: The manner of invocation of this feels rather convoluted and fragile.
    # TODO: THere must be a better way!

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("path", type=str, help="The path in which to find the results.")
    parser.add_argument("-n", "--n_words", type=int, nargs="+", required=False, help="Only consider this many words.")
    args = parser.parse_args()

    n_wordss = args.n_words

    if n_wordss is not None:
        str_list = " or ".join(f"{n:,}" for n in n_wordss)
        logger.info(f"Only looking at outputs from models with {str_list} words")
    else:
        logger.info(f"Looking at all available model outputs")

    dirs = glob.glob(path.join(args.path, "*"))

    cp = CategoryProduction()

    available_items = set()

    for d in dirs:
        if path.isdir(d):
            # If we're restricting by words
            if n_wordss is not None and interpret_path(d) not in n_wordss:
                continue
            logger.info(path.basename(d))
            if not available_items:
                available_items = main_in_path(d, cp, None)
            else:
                available_items = set.intersection(available_items, main_in_path(d, cp, None))

    logger.info(f"Looking at restricted set of {len(available_items)} items")

    for d in dirs:
        if path.isdir(d):
            # If we're restricting by words
            if n_wordss is not None and interpret_path(d) not in n_wordss:
                continue
            logger.info(path.basename(d))
            main_in_path(d, cp, available_items)

    logger.info("Done!")
