"""
===========================
Compare model to Briony's category production actual responses.

Pass the parent location to a bunch of results.
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
from math import floor
from os import path
from typing import Dict, DefaultDict

from pandas import DataFrame, isna

from category_production.category_production import CategoryProduction
from category_production.category_production import ColNames as CPColNames

from preferences import Preferences
from evaluation.column_names import TTFA, MODEL_HIT, CATEGORY_AVAILABLE, PRODUCTION_PROPORTION, ROUNDED_MEAN_RANK, \
    RANK_FREQUENCY_OF_PRODUCTION
from evaluation.category_production import interpret_path_linguistic, get_model_ttfas_for_category_linguistic, \
    save_stats, N_PARTICIPANTS, available_categories
from evaluation.comparison import hitrate_within_sd_of_mean_frac, get_summary_table

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"

category_production = CategoryProduction()


def main(input_results_dir: str, conscious_access_threshold: float, min_first_rank_freq: int = None):

    # Set defaults
    if min_first_rank_freq is None:
        min_first_rank_freq = 1

    n_words = interpret_path_linguistic(input_results_dir)

    logger.info(f"Looking at output from model with {n_words:,} words.")

    per_category_stats_output_path = path.join(Preferences.results_dir,
                                               "Category production fit",
                                               f"item-level data ({path.basename(input_results_dir)}) "
                                               f"CAT={conscious_access_threshold}.csv")

    # region Build main dataframe

    # Main dataframe holds category production data and model response data
    main_dataframe: DataFrame = category_production.data.copy()

    # Add model TTFA column to main_dataframe
    model_ttfas: Dict[str, DefaultDict[str, int]] = dict()  # category -> response -> TTFA
    for category in category_production.category_labels:
        model_ttfas[category] = get_model_ttfas_for_category_linguistic(category, input_results_dir, n_words, conscious_access_threshold)
    main_dataframe[TTFA] = main_dataframe.apply(
        lambda row: model_ttfas[row[CPColNames.Category]][row[CPColNames.Response]], axis=1)

    # Derived column for whether the model produced the response at all (bool)
    main_dataframe[MODEL_HIT] = main_dataframe.apply(lambda row: not isna(row[TTFA]), axis=1)

    # Whether the category was available to the model
    # Category available iff there is an output file for it
    main_dataframe[CATEGORY_AVAILABLE] = main_dataframe.apply(lambda row: row[CPColNames.Category] in available_categories(input_results_dir), axis=1)

    # Production proportion is the number of times a response was given,
    # divided by the number of participants who gave it
    main_dataframe[PRODUCTION_PROPORTION] = main_dataframe.apply(
        lambda row: row[CPColNames.ProductionFrequency] / N_PARTICIPANTS, axis=1)

    main_dataframe[RANK_FREQUENCY_OF_PRODUCTION] = (main_dataframe
                                                    # Within each category
                                                    .groupby(CPColNames.CategorySensorimotor)
                                                    # Rank the responses according to production frequency
                                                    [CPColNames.ProductionFrequency]
                                                    .rank(ascending=False,
                                                          # For ties, order alphabetically (i.e. pseudorandomly (?))
                                                          method='first'))

    # Exclude idiosyncratic responses
    main_dataframe = main_dataframe[main_dataframe[CPColNames.ProductionFrequency] > 1]

    # Production proportion per rank frequency of production
    main_dataframe[ROUNDED_MEAN_RANK] = main_dataframe.apply(lambda row: floor(row[CPColNames.MeanRank]), axis=1)

    # endregion

    # region summary tables

    production_proportion_per_rfop = get_summary_table(main_dataframe, RANK_FREQUENCY_OF_PRODUCTION)
    production_proportion_per_rfop_restricted = get_summary_table(main_dataframe[main_dataframe[CATEGORY_AVAILABLE]],
                                                                  RANK_FREQUENCY_OF_PRODUCTION)

    # Production proportion per rounded mean rank
    production_proportion_per_rmr = get_summary_table(main_dataframe, ROUNDED_MEAN_RANK)
    production_proportion_per_rmr_restricted = get_summary_table(main_dataframe[main_dataframe[CATEGORY_AVAILABLE]],
                                                                 ROUNDED_MEAN_RANK)

    # Compute hitrate fits

    hitrate_fit_rfop = hitrate_within_sd_of_mean_frac(production_proportion_per_rfop)
    hitrate_fit_rfop_restricted = hitrate_within_sd_of_mean_frac(production_proportion_per_rfop_restricted)
    hitrate_fit_rmr = hitrate_within_sd_of_mean_frac(production_proportion_per_rmr)
    hitrate_fit_rmr_restricted = hitrate_within_sd_of_mean_frac(production_proportion_per_rmr_restricted)

    # region Compute correlations with DVs

    # Drop rows not produced by model or in norms
    main_dataframe.dropna(inplace=True, how='any', subset=[TTFA])

    # Now we can convert TTFAs to ints and distances to floats as there won't be null values
    main_dataframe[TTFA] = main_dataframe[TTFA].astype(int)

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
    first_rank_frequent_corr_rt_vs_ttfa = first_rank_frequent_data[CPColNames.MeanZRT].corr(
        first_rank_frequent_data[TTFA], method='pearson')

    # Pearson's correlation between production frequency and ttfa
    corr_prodfreq_vs_ttfa = main_dataframe[CPColNames.ProductionFrequency].corr(main_dataframe[TTFA], method='pearson')

    corr_meanrank_vs_ttfa = main_dataframe[CPColNames.MeanRank].corr(main_dataframe[TTFA], method='pearson')

    # endregion

    # region Save tables

    # Save item-level data

    per_category_stats_output_path = path.join(Preferences.results_dir,
                                               "Category production fit",
                                               f"item-level data ({path.basename(input_results_dir)}).csv")
    main_dataframe.to_csv(per_category_stats_output_path, index=False)

    # Save summary tables

    production_proportion_per_rfop.to_csv(path.join(Preferences.results_dir, "Category production fit",
                                                    f"Production proportion per rank frequency of production"
                                                    f" ({path.basename(input_results_dir)}).csv"),
                                          index=False)
    production_proportion_per_rmr.to_csv(path.join(Preferences.results_dir, "Category production fit",
                                                   f"Production proportion per rounded mean rank"
                                                   f" ({path.basename(input_results_dir)}).csv"),
                                         index=False)

    production_proportion_per_rfop_restricted.to_csv(path.join(Preferences.results_dir, "Category production fit",
                                                               f"Production proportion per rank frequency of production"
                                                               f" ({path.basename(input_results_dir)}) restricted.csv"),
                                                     index=False)
    production_proportion_per_rmr_restricted.to_csv(path.join(Preferences.results_dir, "Category production fit",
                                                              f"Production proportion per rounded mean rank"
                                                              f" ({path.basename(input_results_dir)}) restricted.csv"),
                                                    index=False)

    # Collect
    available_items = set(main_dataframe[[CPColNames.Category, CPColNames.Response]]
                          .groupby([CPColNames.Category, CPColNames.Response])
                          .groups.keys())

    save_stats(
        sensorimotor=False,
        available_items=available_items,
        conscious_access_threshold=conscious_access_threshold,
        corr_frf_vs_ttfa=corr_frf_vs_ttfa,
        corr_meanrank_vs_ttfa=corr_meanrank_vs_ttfa,
        corr_prodfreq_vs_ttfa=corr_prodfreq_vs_ttfa,
        first_rank_frequent_corr_rt_vs_ttfa=first_rank_frequent_corr_rt_vs_ttfa,
        n_first_rank_frequent=n_first_rank_frequent,
        results_dir=input_results_dir,
        min_first_rank_freq=min_first_rank_freq,
        hitrate_fit_rfop=hitrate_fit_rfop,
        hitrate_fit_rfop_available_cats_only=hitrate_fit_rfop_restricted,
        hitrate_fit_rmr=hitrate_fit_rmr,
        hitrate_fit_rmr_available_cats_only=hitrate_fit_rmr_restricted,
    )

    # endregion


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("path", type=str, help="The path in which to find the results.")
    parser.add_argument("cat", type=float, help="The conscious-access threshold.")
    parser.add_argument("min_frf", type=int, nargs="?", default=None, help="The minimum FRF required for zRT and FRF correlations.")
    args = parser.parse_args()

    main(args.path, args.cat, args.min_frf)

    logger.info("Done!")
