"""
===========================
Collate a bunch of category production comparison results and tabulate them

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
import glob
import logging
import sys
from os import path, makedirs

from pandas import concat, read_csv, DataFrame, pivot_table

from model.utils.file import pivot_table_to_csv

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main(results_dir: str) -> None:
    dvs = [
        "FRF corr (-)",
        "zRT corr (+; FRF≥1)",
        'ProdFreq corr (-)',
        'MeanRank corr (+)',
        "FRF N",
        "zRT N",
        'ProdFreq N',
        'Mean Rank N',
    ]

    all_data: DataFrame = collate_data(results_dir)

    # In case there is insufficient data available, there will be nans in the correlation columns.
    # For display purposes, we replace those with "-"s, to be picked up by the pivot table.
    all_data.fillna("-", inplace=True)

    for dv in dvs:

        # pivot
        p = pivot_table(data=all_data, index="Max sphere radius", columns="Bailout", values=dv,
                        # there should be only one value for each group, but we need to use "first" because the
                        # default is "mean", which doesn't work with the "-"s we used to replace nans.
                        aggfunc="first")
        save_dir = path.join(results_dir, " tabulated")
        makedirs(save_dir, exist_ok=True)
        pivot_table_to_csv(p, path.join(save_dir, f"{dv}.csv"))


def collate_data(results_dir: str) -> DataFrame:
    results_paths = glob.iglob(path.join(results_dir, "model_effectiveness*"))
    all_data: DataFrame = concat([read_csv(p, header=0) for p in results_paths])
    all_data.to_csv(path.join(results_dir, "all.csv"), index=False)
    return all_data


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("-p", "--path", required=True, type=str, help="Path in which to find the results.")
    args = parser.parse_args()

    main(args.path)

    logger.info("Done!")