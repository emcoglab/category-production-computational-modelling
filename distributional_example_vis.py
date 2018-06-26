"""
===========================
Visualisation for the distributional example.
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

import logging
import sys
from os import path

import seaborn
from matplotlib import pyplot
from pandas import read_csv

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main():
    box_root = "/Users/caiwingfield/Box Sync/WIP/"
    csv_location = path.join(box_root, "activated node counts.csv")

    results_df = read_csv(csv_location, header=0, index_col=False)

    save_example(box_root, results_df)


def save_example(fig_location, results_df):
    data = results_df

    for d in [0.85, 0.9, 0.99]:
        for s in [10, 15, 20]:

            ax = seaborn.tsplot(
                data=data[
                    (data["Node decay factor"] == d)
                    & (data["Edge decay SD"] == s)
                ],
                time="clock", value="n_activated", unit="Run",
                condition="Activation threshold",
                err_style="unit_traces",
            )

            ax.set_title(f"d={d} s={s}")

            fname = path.join(fig_location, f"trace d={d} s={s}.png")

            pyplot.savefig(fname)
            pyplot.close()


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
