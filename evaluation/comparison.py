"""
===========================
Functions for comparisons of data.
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

from pandas import DataFrame, Series

from evaluation.column_names import MODEL_HITRATE, PRODUCTION_PROPORTION, MODEL_HIT
from model.utils.maths import t_confidence_interval


def hitrate_within_sd_of_mean_frac(df: DataFrame) -> DataFrame:
    # When the model hitrate is within one SD of the production proportion mean
    within = Series(
        (df[MODEL_HITRATE] > df[PRODUCTION_PROPORTION + " Mean"] - df[PRODUCTION_PROPORTION + " SD"])
        & (df[MODEL_HITRATE] < df[PRODUCTION_PROPORTION + " Mean"] + df[PRODUCTION_PROPORTION + " SD"]))
    # The fraction of times this happens
    return within.aggregate('mean')


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
                                                                                     row[
                                                                                         PRODUCTION_PROPORTION + ' Count'],
                                                                                     0.95), axis=1)
    # Model columns
    df[MODEL_HITRATE] = (
        main_dataframe[[groupby_column, MODEL_HIT]].astype(float).groupby(groupby_column).mean()[MODEL_HIT])
    # Forget rows with nans
    df = df.dropna().reset_index()
    return df
