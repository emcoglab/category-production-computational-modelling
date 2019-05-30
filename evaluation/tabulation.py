"""
===========================
Tabulation and working with pandas.DataFrames.
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


from pandas import pivot_table, DataFrame


def tabulate(all_data, dv, rows, cols):
    # pivot
    p = pivot_table(data=all_data, index=rows, columns=cols, values=dv,
                    # there should be only one value for each group, but we need to use "first" because the
                    # default is "mean", which doesn't work with the "-"s we used to replace nans.
                    aggfunc="first")
    return p


def tabulation_to_csv(pt, csv_path):
    """
    Saves a simple DataFrame.pivot_table to a csv, including columns names.
    Thanks to https://stackoverflow.com/a/55360229/2883198
    :param pt:
    :param csv_path:
    :return:
    """
    csv_df: DataFrame = DataFrame(columns=pt.columns, index=[pt.index.name]).append(pt)
    csv_df.to_csv(csv_path, index_label=pt.columns.name)