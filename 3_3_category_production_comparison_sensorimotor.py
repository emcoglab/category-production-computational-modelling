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
from math import floor
from os import path
from typing import Dict

from matplotlib import pyplot
from numpy import nan, array
from pandas import DataFrame, isna, Series

from category_production.category_production import CategoryProduction
from category_production.category_production import ColNames as CPColNames
from evaluation.category_production import TTFA, get_model_ttfas_for_category_sensorimotor
from ldm.corpus.tokenising import modified_word_tokenize
from ldm.utils.maths import DistanceType, distance
from model.graph_propagation import GraphPropagation
from model.utils.maths import t_confidence_interval
from preferences import Preferences
from sensorimotor_norms.exceptions import WordNotInNormsError
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


RANK_FREQUENCY_OF_PRODUCTION = "RankFreqOfProduction"
ROUNDED_MEAN_RANK = "RoundedMeanRank"
PRODUCTION_PROPORTION = "ProductionProportion"
CATEGORY_AVAILABLE = "CategoryAvailable"
MODEL_HIT = "ModelHit"
MODEL_HITRATE = "Model hitrate"


N_PARTICIPANTS = 20


category_production = CategoryProduction(use_cache=True)
sensorimotor_norms = SensorimotorNorms()

distance_column = f"{DistanceType.Minkowski3.name} distance"


def main(input_results_dir: str, min_first_rank_freq: int = None):

    # region Set defaults
    min_first_rank_freq = 1 if min_first_rank_freq is None else min_first_rank_freq
    # endregion

    # Main dataframe holds category production data and model response data
    main_dataframe: DataFrame = category_production.data.copy()
    main_dataframe.drop(['Sensorimotor.distance.cosine.additive', 'Linguistic.PPMI'], axis=1, inplace=True)

    main_dataframe = exclude_idiosyncratic_responses(main_dataframe)

    # region Add predictor columns

    # Add predictor columns for model
    main_dataframe = add_predictor_column_distance(main_dataframe)
    main_dataframe = add_predictor_column_ttfa(main_dataframe, input_results_dir)
    main_dataframe = add_predictor_column_model_hit(main_dataframe)
    main_dataframe = add_predictor_column_category_available(main_dataframe)

    main_dataframe = add_predictor_column_production_proportion(main_dataframe)
    main_dataframe = add_rfop_column(main_dataframe)
    main_dataframe = add_rmr_column(main_dataframe)

    # endregion

    # region Save data files

    save_item_level_data(input_results_dir, main_dataframe)
    hitrate_stats = save_hitrate_summary_tables(input_results_dir, main_dataframe)
    save_model_performance_stats_sensorimotor(
        main_dataframe,
        results_dir=input_results_dir,
        min_first_rank_freq=min_first_rank_freq,
        **hitrate_stats
    )

    # endregion


def save_model_performance_stats_sensorimotor(main_dataframe,
                                              results_dir,
                                              min_first_rank_freq,
                                              hitrate_fit_rfop,
                                              hitrate_fit_rfop_restricted,
                                              hitrate_fit_rmr,
                                              hitrate_fit_rmr_restricted):

    overall_stats_output_path = path.join(Preferences.results_dir,
                                          "Category production fit",
                                          f"model_effectiveness_overall "
                                          f"({path.basename(results_dir)}).csv")
    model_spec = GraphPropagation.load_model_spec(results_dir)
    stats = {
        **get_correlation_stats(main_dataframe, min_first_rank_freq),
        # hitrate stats
        "Hitrate within SD of mean (RFoP)": hitrate_fit_rfop,
        "Hitrate within SD of mean (RFoP; available categories only)": hitrate_fit_rfop_restricted,
        "Hitrate within SD of mean (RMR)": hitrate_fit_rmr,
        "Hitrate within SD of mean (RMR; available categories only)": hitrate_fit_rmr_restricted,
    }
    data: DataFrame = DataFrame.from_records([{
        **model_spec,
        **stats,
    }])

    with open(overall_stats_output_path, mode="w", encoding="utf-8") as data_file:
        data.to_csv(data_file, index=False,
                    # Make sure columns are in consistent order for stacking,
                    # and make sure the model spec columns come first.
                    columns=sorted(model_spec.keys()) + sorted(stats.keys()))


def save_item_level_data(input_results_dir, main_dataframe):
    per_category_stats_output_path = path.join(Preferences.results_dir,
                                               "Category production fit",
                                               f"item-level data ({path.basename(input_results_dir)}).csv")
    main_dataframe.to_csv(per_category_stats_output_path, index=False)


def get_correlation_stats(main_dataframe, min_first_rank_freq):
    # Drop rows not produced by model or in norms
    correlation_dataframe = main_dataframe[main_dataframe[TTFA].notnull()]
    correlation_dataframe = correlation_dataframe[correlation_dataframe[distance_column].notnull()]
    # Now we can convert TTFAs to ints and distances to floats as there won't be null values
    correlation_dataframe[TTFA] = correlation_dataframe[TTFA].astype(int)
    correlation_dataframe[distance_column] = correlation_dataframe[distance_column].astype(float)
    # frf vs ttfa
    corr_frf_vs_ttfa = correlation_dataframe[CPColNames.FirstRankFrequency].corr(correlation_dataframe[TTFA],
                                                                                 method='pearson')
    # The Pearson's correlation over all categories between average first-response RT (for responses with
    # first-rank frequency ≥4) and the time to the first activation (TTFA) within the model.
    first_rank_frequent_data = correlation_dataframe[
        correlation_dataframe[CPColNames.FirstRankFrequency] >= min_first_rank_freq]
    n_first_rank_frequent = first_rank_frequent_data.shape[0]
    first_rank_frequent_corr_rt_vs_ttfa = first_rank_frequent_data[CPColNames.MeanZRT].corr(
        first_rank_frequent_data[TTFA], method='pearson')
    # Pearson's correlation between production frequency and ttfa
    corr_prodfreq_vs_ttfa = correlation_dataframe[CPColNames.ProductionFrequency].corr(correlation_dataframe[TTFA],
                                                                                       method='pearson')
    corr_meanrank_vs_ttfa = correlation_dataframe[CPColNames.MeanRank].corr(correlation_dataframe[TTFA],
                                                                            method='pearson')
    # Save correlation and hitrate stats
    available_pairs = set(correlation_dataframe[[CPColNames.CategorySensorimotor, CPColNames.ResponseSensorimotor]]
                          .groupby([CPColNames.CategorySensorimotor, CPColNames.ResponseSensorimotor])
                          .groups.keys())
    correlation_stats = {
        "available_items": available_pairs,
        "corr_frf_vs_ttfa": corr_frf_vs_ttfa,
        "corr_meanrank_vs_ttfa": corr_meanrank_vs_ttfa,
        "corr_prodfreq_vs_ttfa": corr_prodfreq_vs_ttfa,
        "first_rank_frequent_corr_rt_vs_ttfa": first_rank_frequent_corr_rt_vs_ttfa,
        "n_first_rank_frequent": n_first_rank_frequent,
        "min_first_rank_freq": min_first_rank_freq
    }
    return correlation_stats


def save_hitrate_summary_tables(input_results_dir, main_dataframe):
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
    # region Graph tables
    save_figure(summary_table=production_proportion_per_rfop,
                x_selector=RANK_FREQUENCY_OF_PRODUCTION,
                fig_title="Hitrate per RFOP",
                fig_name=f"hitrate per RFOP {path.basename(input_results_dir)}")
    save_figure(summary_table=production_proportion_per_rfop_restricted,
                x_selector=RANK_FREQUENCY_OF_PRODUCTION,
                fig_title="Hitrate per RFOP (only available categories)",
                fig_name=f"restricted hitrate per RFOP {path.basename(input_results_dir)}")
    save_figure(summary_table=production_proportion_per_rmr,
                x_selector=ROUNDED_MEAN_RANK,
                fig_title="Hitrate per RMR",
                fig_name=f"hitrate per RMR {path.basename(input_results_dir)}")
    save_figure(summary_table=production_proportion_per_rmr_restricted,
                x_selector=ROUNDED_MEAN_RANK,
                fig_title="Hitrate per RMR (only available categories)",
                fig_name=f"restricted hitrate per RMR {path.basename(input_results_dir)}")
    hitrate_stats = {
        "hitrate_fit_rfop": hitrate_fit_rfop,
        "hitrate_fit_rfop_restricted": hitrate_fit_rfop_restricted,
        "hitrate_fit_rmr": hitrate_fit_rmr,
        "hitrate_fit_rmr_restricted": hitrate_fit_rmr_restricted
    }
    return hitrate_stats


def get_sensorimotor_distance_minkowski3(row):
    try:
        category_vector = array(sensorimotor_norms.vector_for_word(row[CPColNames.CategorySensorimotor]))
        response_vector = array(sensorimotor_norms.vector_for_word(row[CPColNames.ResponseSensorimotor]))
        return distance(category_vector, response_vector, DistanceType.Minkowski3)
    except WordNotInNormsError:
        return nan


def hitrate_within_sd_of_mean_frac(df: DataFrame) -> DataFrame:
    # When the model hitrate is within one SD of the production proportion mean
    within = Series(
        (df["Model hitrate"] > df["ProductionProportion Mean"] - df["ProductionProportion SD"])
        & (df["Model hitrate"] < df["ProductionProportion Mean"] + df["ProductionProportion SD"]))
    # The fraction of times this happens
    return within.aggregate('mean')


def save_figure(summary_table, x_selector, fig_title, fig_name):
    """Save a summary table as a figure."""
    # add human bounds
    pyplot.fill_between(x=summary_table.reset_index()[x_selector],
                        y1=summary_table[PRODUCTION_PROPORTION + ' Mean'] - summary_table[PRODUCTION_PROPORTION + ' SD'],
                        y2=summary_table[PRODUCTION_PROPORTION + ' Mean'] + summary_table[PRODUCTION_PROPORTION + ' SD'])
    pyplot.scatter(x=summary_table.reset_index()[x_selector],
                   y=summary_table[PRODUCTION_PROPORTION + ' Mean'])
    # add model performance
    pyplot.scatter(x=summary_table.reset_index()[x_selector],
                   y=summary_table[MODEL_HITRATE])

    pyplot.ylim((0, None))

    pyplot.title(fig_title)
    pyplot.xlabel(x_selector)
    pyplot.ylabel("Production proportion / hitrate")

    pyplot.savefig(
        path.join(Preferences.figures_dir, "hitrates", f"{fig_name}.png"))
    pyplot.clf()
    pyplot.cla()
    pyplot.close()


def get_summary_table(main_dataframe, groupby_column):
    """
    Summarise main dataframe by aggregating production proportion by the stated `groupby_column` column.
    """
    df = DataFrame()
    # Participant columns
    df[PRODUCTION_PROPORTION + ' Mean'] = (
        main_dataframe
            .groupby(groupby_column)
            .mean()[PRODUCTION_PROPORTION])
    df[PRODUCTION_PROPORTION + ' SD'] = (
        main_dataframe
            .groupby(groupby_column)
            .std()[PRODUCTION_PROPORTION])
    df[PRODUCTION_PROPORTION + ' Count'] = (
        main_dataframe
            .groupby(groupby_column)
            .count()[PRODUCTION_PROPORTION])
    df[PRODUCTION_PROPORTION + ' CI95'] = df.apply(lambda row: t_confidence_interval(row[PRODUCTION_PROPORTION + ' SD'],
                                                                                     row[PRODUCTION_PROPORTION + ' Count'],
                                                                                     0.95), axis=1)
    # Model columns
    df[MODEL_HITRATE] = (
        main_dataframe.groupby(groupby_column).mean()[MODEL_HIT])
    # Forget rows with nans
    df = df.dropna().reset_index()
    return df


def add_predictor_column_ttfa(main_dataframe, input_results_dir):
    logger.info("Adding TTFA column")

    # category -> response -> TTFA
    ttfas: Dict[str, Dict[str, int]] = {
        category: get_model_ttfas_for_category_sensorimotor(category, input_results_dir)
        for category in category_production.category_labels_sensorimotor
    }

    def get_min_ttfa_for_multiword_responses(row) -> int:
        """
        Helper function to convert a row in the output into a ttfa when the response is formed either of a single or
        multiple norms terms.
        """
        c = row[CPColNames.CategorySensorimotor]
        r = row[CPColNames.ResponseSensorimotor]

        # If the category is not found in the dictionary, it was not accessed by the model so no TTFA will be present
        # for any response.
        try:
            # response -> TTFA
            c_ttfas: Dict[str, int] = ttfas[c]
        except KeyError:
            return nan

        # If the response was directly found, we can return it
        if r in c_ttfas:
            return c_ttfas[r]

        # Otherwise, try to break the response into components and find any one of them
        else:
            r_ttfas = [c_ttfas[w]
                       for w in modified_word_tokenize(r)
                       if (w not in category_production.ignored_words)
                       and (w in c_ttfas)]

            # The multi-word response is said to be activated the first time any one of its constituent words are
            if len(r_ttfas) > 1:
                return min(r_ttfas)
            # If none of the constituent words have a ttfa, we have no ttfa for the multiword term
            else:
                return nan

    main_dataframe[TTFA] = main_dataframe.apply(get_min_ttfa_for_multiword_responses, axis=1)

    return main_dataframe


def add_predictor_column_distance(main_dataframe):
    logger.info("Adding distance column")
    main_dataframe[distance_column] = main_dataframe.apply(get_sensorimotor_distance_minkowski3, axis=1)
    return main_dataframe


def add_predictor_column_model_hit(main_dataframe):
    logger.info("Adding model hit column")
    main_dataframe[MODEL_HIT] = main_dataframe.apply(lambda row: not isna(row[TTFA]), axis=1)
    return main_dataframe


def add_predictor_column_category_available(main_dataframe):
    logger.info("Adding category availability column")
    main_dataframe[CATEGORY_AVAILABLE] = main_dataframe.apply(lambda row: sensorimotor_norms.has_word(row[CPColNames.CategorySensorimotor]), axis=1)
    return main_dataframe


def add_predictor_column_production_proportion(main_dataframe):
    logger.info("Adding production proportion column")
    # Production proportion is the number of times a response was given,
    # divided by the number of participants who gave it
    main_dataframe[PRODUCTION_PROPORTION] = main_dataframe.apply(lambda row: row[CPColNames.ProductionFrequency] / N_PARTICIPANTS, axis=1)
    return main_dataframe


def add_rfop_column(main_dataframe):
    logger.info("Adding RFoP column")
    main_dataframe[RANK_FREQUENCY_OF_PRODUCTION] = (
        main_dataframe
        # Within each category
        .groupby(CPColNames.CategorySensorimotor)
        # Rank the responses according to production frequency
        [CPColNames.ProductionFrequency]
        .rank(ascending=False,
              # For ties, order alphabetically (i.e. pseudorandomly (?))
              method='first'))
    return main_dataframe


def add_rmr_column(main_dataframe):
    logger.info("Adding RMR column")
    main_dataframe[ROUNDED_MEAN_RANK] = main_dataframe.apply(lambda row: floor(row[CPColNames.MeanRank]), axis=1)
    return main_dataframe


def exclude_idiosyncratic_responses(main_dataframe):
    return main_dataframe[main_dataframe[CPColNames.ProductionFrequency] > 1]


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("path", type=str, help="The path in which to find the results.")
    parser.add_argument("min_frf", type=int, nargs="?", default=None, help="The minimum FRF required for zRT and FRF correlations.")
    args = parser.parse_args()

    main(args.path, args.min_frf)

    logger.info("Done!")
