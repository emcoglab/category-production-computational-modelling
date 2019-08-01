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
import logging
import sys
from os import path
from typing import Dict, DefaultDict, Set, Tuple

from pandas import DataFrame

from category_production.category_production import CategoryProduction
from category_production.category_production import ColNames as CPColNames
from evaluation.category_production import interpret_path_linguistic, get_model_ttfas_for_category_linguistic, \
    save_stats_linguistic
from evaluation.column_names import TTFA
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main_restricted(results_dir: str, category_production: CategoryProduction,
                    available_items: Set[Tuple[str, str]],
                    conscious_access_threshold: float,
                    min_first_rank_freq: int):

    n_words = interpret_path_linguistic(results_dir)

    per_category_stats_output_path = path.join(Preferences.results_dir,
                                               "Category production fit",
                                               f"item-level data (restricted) ({path.basename(results_dir)}).csv")

    # region Build main dataframe

    # Main dataframe holds category production data and model response data
    main_dataframe: DataFrame = category_production.data.copy()

    # Drop precomputed distance measures
    main_dataframe.drop(['LgSUBTLWF', 'Sensorimotor.proximity', 'Linguistic.proximity'], axis=1, inplace=True)

    # Add model TTFA column
    model_ttfas: Dict[str, DefaultDict[str, int]] = dict()  # category -> response -> TTFA
    for category in category_production.category_labels:
        model_ttfas[category] = get_model_ttfas_for_category_linguistic(category, results_dir, n_words, conscious_access_threshold)
    main_dataframe[TTFA] = main_dataframe.apply(
        lambda row: model_ttfas[row[CPColNames.Category]][row[CPColNames.Response]], axis=1)

    # Drop rows corresponding to responses which weren't produced by the model
    main_dataframe = main_dataframe[main_dataframe[TTFA].notnull()]

    # Now we can convert TTFAs to ints as there won't be null values
    main_dataframe[TTFA] = main_dataframe[TTFA].astype(int)

    # Restrict on available items
    main_dataframe = main_dataframe[
        # Filter on categories
        main_dataframe[CPColNames.Category].isin([c for c, r in available_items])
        # Filter on responses
        & main_dataframe[CPColNames.Response].isin([r for c, r in available_items])]

    # Save main dataframe
    main_dataframe.to_csv(per_category_stats_output_path, index=False)

    # endregion

    # region Compute overall stats

    # frf vs ttfa
    corr_frf_vs_ttfa = main_dataframe[CPColNames.FirstRankFrequency].corr(main_dataframe[TTFA], method='pearson')

    # The Pearson's correlation over all categories between average first-response RT (for responses with
    # first-rank frequency ≥4) and the time to the first activation (TTFA) within the model.
    first_rank_frequent_data = main_dataframe[main_dataframe[CPColNames.FirstRankFrequency] >= min_first_rank_freq]
    n_first_rank_frequent = first_rank_frequent_data.shape[0]
    first_rank_frequent_corr_rt_vs_ttfa = first_rank_frequent_data[CPColNames.MeanZRT].corr(first_rank_frequent_data[TTFA], method='pearson')

    # Pearson's correlation between production frequency and ttfa
    corr_prodfreq_vs_ttfa = main_dataframe[CPColNames.ProductionFrequency].corr(main_dataframe[TTFA], method='pearson')

    corr_meanrank_vs_ttfa = main_dataframe[CPColNames.MeanRank].corr(main_dataframe[TTFA], method='pearson')

    # endregion

    save_stats_linguistic(available_items, corr_frf_vs_ttfa, corr_meanrank_vs_ttfa, corr_prodfreq_vs_ttfa,
                          first_rank_frequent_corr_rt_vs_ttfa, n_first_rank_frequent, results_dir, True, min_first_rank_freq, conscious_access_threshold)


def get_available_items_from_path(p: str, cp: CategoryProduction, conscious_access_threshold) -> Set[Tuple[str, str]]:
    n_words = interpret_path_linguistic(p)

    # Main dataframe holds category production data and model response data
    main_dataframe: DataFrame = cp.data.copy()

    # Add model TTFA column
    model_ttfas: Dict[str, DefaultDict[str, int]] = dict()  # category -> response -> TTFA
    for category in cp.category_labels:
        model_ttfas[category] = get_model_ttfas_for_category_linguistic(category, p, n_words, conscious_access_threshold)
    main_dataframe[TTFA] = main_dataframe.apply(
        lambda row: model_ttfas[row[CPColNames.Category]][row[CPColNames.Response]], axis=1)

    # Drop rows corresponding to responses which weren't produced by the model
    main_dataframe = main_dataframe[main_dataframe[TTFA].notnull()]

    # Now we can convert TTFAs to ints as there won't be null values
    main_dataframe[TTFA] = main_dataframe[TTFA].astype(int)

    # If we specified to restrict to a set of available items, do the restriction now,
    # otherwise collect what items we have for this run
    available_items: Set = set(main_dataframe[[CPColNames.Category, CPColNames.Response]]
                               .groupby([CPColNames.Category, CPColNames.Response])
                               .groups.keys())

    return available_items


def main(paths, conscious_access_threshold: float, min_first_rank_freq: int = None):

    # Set defaults
    if min_first_rank_freq is None:
        min_first_rank_freq = 1

    logger.info(f'Only looking at outputs shared between models with {" and ".join(f"{n:,}" for n in [interpret_path_linguistic(p) for p in paths])} words')

    cp = CategoryProduction()

    # Get intersection of available items
    available_items: Set = None
    for p in paths:
        if not available_items:
            available_items = get_available_items_from_path(p, cp, conscious_access_threshold)
        else:
            new_available_items = get_available_items_from_path(p, cp, conscious_access_threshold)
            available_items = set.intersection(available_items, new_available_items)

    logger.info(f"Looking at restricted set of {len(available_items)} items")
    first_path = paths[0]
    logger.info(path.basename(first_path))
    main_restricted(first_path, cp, available_items, conscious_access_threshold, min_first_rank_freq)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("-p", "--path", required=True, type=str, help="Path in which to find the results.")
    parser.add_argument("-c", "--cat", required=True, type=float, help="Conscious-access threshold.")
    parser.add_argument("-m", "--min_frf", type=int, required=False, default=None, help="The minimum FRF required for zRT and FRF correlations.")
    args = parser.parse_args()

    main(args.paths, args.cat, args.min_frf)

    logger.info("Done!")
