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
from os import path, mkdir

import yaml
from pandas import concat, read_csv, DataFrame, pivot_table, ExcelWriter

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def load_model_spec(response_dir) -> dict:
    with open(path.join(response_dir, " model_spec.yaml"), mode="r", encoding="utf-8") as spec_file:
        return yaml.load(spec_file)


def main(results_dir: str) -> None:
    all_data: DataFrame = collate_data(results_dir)

    models = all_data["Model"].unique()
    dvs = [
        "FRF corr (-)",
        "FRF N",
        "zRT corr (+; FRF≥1)",
        "zRT N",
        'ProdFreq corr (-)',
        'ProdFreq N',
        'MeanRank corr (+)',
        'Mean Rank N',
    ]

    for model in models:
        this_model_data = all_data[all_data["Model"].str.startswith(model)]

        word_counts = this_model_data["Words"].unique()

        for wc in word_counts:

            this_model_data = this_model_data[this_model_data["Words"] == wc]

            # Select only min CAT
            this_model_data = this_model_data[this_model_data["Firing threshold"].eq(this_model_data["CAT"])]

            for dv in dvs:

                # pivot
                p = pivot_table(data=this_model_data, index="SD factor", columns="Firing threshold", values=dv)
                if not path.isdir(path.join(results_dir, "tabulated")):
                    mkdir(path.join(results_dir, "tabulated"))
                # csv
                p.to_csv(path.join(results_dir, "tabulated", f"{model} {wc} {dv}.csv"))
                # xls
                w = ExcelWriter(path.join(results_dir, "tabulated", f"{model} {wc} {dv}.xls"))
                p.to_excel(w, "Sheet 1")
                w.save()

    pass


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
