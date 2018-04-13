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

    save_example(path.join(box_root, "trace.png"), results_df)


def save_example(fig_location, results_df):
    data = results_df
    # data = data[data["Threshold"] == 0.2]
    data = data[data["Node decay factor"] == 0.8]
    data = data[data["Edge decay SD"] == 15]

    seaborn.tsplot(data=data, time="Tick", value="Activated nodes", unit="Run", condition="Threshold")

    pyplot.savefig(fig_location)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
