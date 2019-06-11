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


def save_tabulation(data: DataFrame, values, rows, cols, path: str):
    """
    Saves a tabulated form of the DataFrame data, where rows are values of `rows`, columns are values of `cols`, and
    values are values of `values`.
    :param data:
    :param values:
    :param rows:
    :param cols:
    :param path:
    :return:
    """
    # pivot
    p = pivot_table(data=data, index=rows, columns=cols, values=values,
                    # there should be only one value for each group, but we need to use "first" because the
                    # default is "mean", which doesn't work with the "-"s we used to replace nans.
                    aggfunc="first")
    _tabulation_to_csv(p, path)


def _tabulation_to_csv(table, csv_path):
    """
    Saves a simple DataFrame.pivot_table to a csv, including columns names.
    Thanks to https://stackoverflow.com/a/55360229/2883198
    """
    csv_df: DataFrame = DataFrame(columns=table.columns, index=[table.index.name]).append(table)
    csv_df.to_csv(csv_path, index_label=table.columns.name)
