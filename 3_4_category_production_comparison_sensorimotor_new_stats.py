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
2019
---------------------------
"""
import argparse
import logging
import sys
from math import sqrt, floor
from os import path
from typing import Dict, DefaultDict

from pandas import DataFrame, isna
from scipy.stats import t as studentt

from category_production.category_production import CategoryProduction
from category_production.category_production import ColNames as CPColNames
from evaluation.category_production import get_model_ttfas_for_category_sensorimotor, TTFA
from preferences import Preferences
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


RANK_FREQUENCY_OF_PRODUCTION = "RankFreqOfProduction"
ROUNDED_MEAN_RANK = "RoundedMeanRank"
PRODUCTION_PROPORTION = "ProductionProportion"
CATEGORY_AVAILABLE = "CategoryAvailable"
MODEL_HIT = "ModelHit"


N_PARTICIPANTS = 20

SN = SensorimotorNorms()


def main(input_results_dir: str):

    per_category_stats_output_path = path.join(Preferences.results_dir,
                                               "Category production fit",
                                               f"item-level data ({path.basename(input_results_dir)}) newstats.csv")

    # We need to include all data (including idiosyncratic responses) so that we can rank properly.
    # Then we will exclude them later.
    logger.info("Loading category production data")
    category_production = CategoryProduction(minimum_production_frequency=1)
    # Main dataframe holds category production data and model response data.
    main_dataframe: DataFrame = category_production.data.copy()

    # Drop precomputed distance measures
    main_dataframe.drop(['Sensorimotor.distance.cosine.additive', 'Linguistic.PPMI'], axis=1, inplace=True)

    # region Add new columns for model

    # category -> response -> TTFA
    model_ttfas: Dict[str, DefaultDict[str, int]] = dict()
    for category in category_production.category_labels_sensorimotor:
        model_ttfas[category] = get_model_ttfas_for_category_sensorimotor(category, input_results_dir)
    # Column for TTFA (int)
    main_dataframe[TTFA] = main_dataframe.apply(
        lambda row: model_ttfas[row[CPColNames.CategorySensorimotor]][row[CPColNames.ResponseSensorimotor]],
        axis=1)
    # Derived column for whether the model produced the response at all (bool)
    main_dataframe[MODEL_HIT] = main_dataframe.apply(
        lambda row: not isna(row[TTFA]),
        axis=1)
    # Whether the category was available to the model
    main_dataframe[CATEGORY_AVAILABLE] = main_dataframe.apply(
        lambda row: SN.has_word(row[CPColNames.CategorySensorimotor]),
        axis=1)

    # endregion

    # region Add new columns for participants

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

    # Produciton proportion per rank frequency of producton
    main_dataframe[ROUNDED_MEAN_RANK] = main_dataframe.apply(lambda row: floor(row[CPColNames.MeanRank]), axis=1)

    # endregion

    # region summary tables

    production_proportion_per_rfop = get_summary_table(main_dataframe, RANK_FREQUENCY_OF_PRODUCTION)
    production_proportion_per_rfop_restricted = get_summary_table(main_dataframe[main_dataframe[CATEGORY_AVAILABLE]], RANK_FREQUENCY_OF_PRODUCTION)

    # Produciton proportion per rounded mean rank
    production_proportion_per_rmr = get_summary_table(main_dataframe, ROUNDED_MEAN_RANK)
    production_proportion_per_rmr_restricted = get_summary_table(main_dataframe[main_dataframe[CATEGORY_AVAILABLE]], ROUNDED_MEAN_RANK)

    # endregion

    # region save tables

    main_dataframe.to_csv(per_category_stats_output_path, index=False)

    production_proportion_per_rfop.to_csv(path.join(Preferences.results_dir, "Category production fit",
                                                    f"Production proportion per rank frequency of production"
                                                    f" ({path.basename(input_results_dir)}).csv"))

    production_proportion_per_rmr.to_csv(path.join(Preferences.results_dir, "Category production fit",
                                                   f"Production proportion per rounded mean rank"
                                                   f" ({path.basename(input_results_dir)}).csv"))

    production_proportion_per_rfop_restricted.to_csv(path.join(Preferences.results_dir, "Category production fit",
                                                               f"Production proportion per rank frequency of production"
                                                               f" ({path.basename(input_results_dir)}) restricted.csv"))

    production_proportion_per_rmr_restricted.to_csv(path.join(Preferences.results_dir, "Category production fit",
                                                              f"Production proportion per rounded mean rank"
                                                              f" ({path.basename(input_results_dir)}) restricted.csv"))

    # endregion


def get_summary_table(main_dataframe, groupby_column):
    df = DataFrame()
    # Participant columns
    df['Mean'] = (
        main_dataframe.groupby(groupby_column).mean()[PRODUCTION_PROPORTION])
    df['SD'] = (
        main_dataframe.groupby(groupby_column).std()[PRODUCTION_PROPORTION])
    df['Count'] = (
        main_dataframe.groupby(groupby_column).count()[PRODUCTION_PROPORTION])
    df['CI95'] = df.apply(lambda row: t_ci(row['SD'], row['Count'], 0.95), axis=1)
    # Model columns
    df['Model hitrate'] = (
        main_dataframe.groupby(groupby_column).mean()[MODEL_HIT])
    return df


def t_ci(sd, n, alpha):
    """
    Confidence interval for t distribution.
    Roughly equivalent to Excell's confidence.t()
    :param sd:
    :param n:
    :param alpha:
    :return:
    """
    sem = sd/sqrt(n)
    return sem * studentt.ppf((1 + alpha) / 2, df=n-1)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("path", type=str, help="The path in which to find the results.")
    args = parser.parse_args()

    main(args.path)

    logger.info("Done!")
