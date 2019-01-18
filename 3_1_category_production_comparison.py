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
from os import path
from typing import Dict, DefaultDict

from pandas import DataFrame

from category_production.category_production import CategoryProduction
from category_production.category_production import ColNames as CPColNames
from evaluation.category_production import TTFA, interpret_path, get_model_ttfas_for_category
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


# Analysis settings
MIN_FIRST_RANK_FREQ = 4


def main(results_dir):

    # TODO: this isn't a great way to do this
    n_words = interpret_path(results_dir)

    logger.info(f"Looking at output from model with {n_words} words.")

    category_production = CategoryProduction()

    # n_words = interpret_path(results_dir)

    per_category_stats_output_path = path.join(Preferences.results_dir,
                                               "Category production fit",
                                               f"item-level data ({path.basename(results_dir)}).csv")

    # region Build main dataframe

    # Main dataframe holds category production data and model response data
    main_dataframe: DataFrame = category_production.data.copy()

    # Drop precomputed distance measures
    main_dataframe.drop(['LgSUBTLWF', 'Sensorimotor', 'Linguistic'], axis=1, inplace=True)

    # Add model TTFA column to main_dataframe
    model_ttfas: Dict[str, DefaultDict[str, int]] = dict()  # category -> response -> TTFA
    for category in category_production.category_labels:
        model_ttfas[category] = get_model_ttfas_for_category(category, results_dir, n_words)
    main_dataframe[TTFA] = main_dataframe.apply(
        lambda row: model_ttfas[row[CPColNames.Category]][row[CPColNames.Response]],
        axis=1)

    # Drop rows corresponding to responses which weren't produced by the model
    main_dataframe = main_dataframe[main_dataframe[TTFA].notnull()]

    # Now we can convert TTFAs to ints as there won't be null values
    main_dataframe[TTFA] = main_dataframe[TTFA].astype(int)

    # Collect
    available_items = set(main_dataframe[[CPColNames.Category, CPColNames.Response]].groupby(
        [CPColNames.Category, CPColNames.Response]).groups.keys())

    # Save main dataframe
    main_dataframe.to_csv(per_category_stats_output_path, index=False)

    # endregion

    # region Compute overall stats

    # frf vs ttfa
    corr_frf_vs_ttfa = main_dataframe[CPColNames.FirstRankFrequency].corr(main_dataframe[TTFA], method='pearson')

    # The Pearson's correlation over all categories between average first-response RT (for responses with
    # first-rank frequency ≥4) and the time to the first activation (TTFA) within the model.
    first_rank_frequent_data = main_dataframe[main_dataframe[CPColNames.FirstRankFrequency] >= MIN_FIRST_RANK_FREQ]
    n_first_rank_frequent = first_rank_frequent_data.shape[0]
    first_rank_frequent_corr_rt_vs_ttfa = first_rank_frequent_data[CPColNames.MeanZRT].corr(
        first_rank_frequent_data[TTFA], method='pearson')

    # Pearson's correlation between production frequency and ttfa
    corr_prodfreq_vs_ttfa = main_dataframe[CPColNames.ProductionFrequency].corr(main_dataframe[TTFA], method='pearson')

    corr_meanrank_vs_ttfa = main_dataframe[CPColNames.MeanRank].corr(main_dataframe[TTFA], method='pearson')

    # endregion

    save_stats(available_items, corr_frf_vs_ttfa, corr_meanrank_vs_ttfa, corr_prodfreq_vs_ttfa,
               first_rank_frequent_corr_rt_vs_ttfa, n_first_rank_frequent, results_dir)

    return available_items


def save_stats(available_items, corr_frf_vs_ttfa, corr_meanrank_vs_ttfa, corr_prodfreq_vs_ttfa,
               first_rank_frequent_corr_rt_vs_ttfa, n_first_rank_frequent, results_dir):
    overall_stats_output_path = path.join(Preferences.results_dir,
                                          "Category production fit",
                                          f"model_effectiveness_overall ({path.basename(results_dir)}).txt")
    with open(overall_stats_output_path, mode="w", encoding="utf-8") as output_file:
        # Correlation of first response RT with time-to-activation
        output_file.write(path.basename(results_dir) + "\n\n")
        output_file.write(f"First rank frequency vs TTFA correlation ("
                          f"Pearson's; negative is better fit; "
                          f"N = {n_first_rank_frequent}) "
                          f"= {corr_frf_vs_ttfa}\n")
        output_file.write(f"First response RT vs TTFA correlation ("
                          f"Pearson's; positive is better fit; "
                          f"FRF≥{MIN_FIRST_RANK_FREQ}; "
                          f"N = {n_first_rank_frequent}) "
                          f"= {first_rank_frequent_corr_rt_vs_ttfa}\n")
        output_file.write(f"Production frequency vs TTFA correlation ("
                          f"Pearson's; negative is better fit; "
                          f"N = {len(available_items)}) "
                          f"= {corr_prodfreq_vs_ttfa}\n")
        output_file.write(f"Mean rank vs TTFA correlation ("
                          f"Pearson's; positive is better fit "
                          f"N = {len(available_items)}) "
                          f"= {corr_meanrank_vs_ttfa}\n")


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("path", type=str, help="The path in which to find the results.")
    args = parser.parse_args()

    main(args.path)

    logger.info("Done!")
