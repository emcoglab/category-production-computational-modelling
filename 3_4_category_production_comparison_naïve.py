#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Compare naïve model to Briony's category production actual responses.

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
from pathlib import Path

from pandas import DataFrame, read_csv

from category_production.category_production import ColNames as CPColNames
from evaluation.category_production import ModelType, CATEGORY_PRODUCTION, add_rfop_column, add_rmr_column, \
    exclude_idiosyncratic_responses, add_predictor_column_production_proportion, save_item_level_data, \
    save_hitrate_summary_tables, save_naïve_model_performance_stats
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main(input_results_dir: str, model_type: ModelType):
    logger.info(path.basename(input_results_dir))

    main_data = compile_model_data(input_results_dir, model_type)
    process_one_model_output(main_data, input_results_dir, model_type)


def compile_model_data(input_results_dir, model_type: ModelType) -> DataFrame:

    assert model_type in [ModelType.naïve_linguistic, ModelType.naïve_sensorimotor]

    # Main dataframe holds category production data and model response data
    main_data: DataFrame = CATEGORY_PRODUCTION.data.copy()

    main_data.rename(columns={col_name: f"Precomputed {col_name}"
                              for col_name in ['Sensorimotor.distance.cosine.additive', 'Linguistic.PPMI']},
                     inplace=True)
    main_data = exclude_idiosyncratic_responses(main_data)

    add_predictor_column_production_proportion(main_data)
    add_rfop_column(main_data, model_type=model_type)
    add_rmr_column(main_data)

    # Add predictor column naïve model hit:
    model_hits: DataFrame = get_naïve_model_hits(input_results_dir)
    main_data = main_data.merge(model_hits, on=[CPColNames.Category, CPColNames.Response], how="left")

    return main_data


def get_naïve_model_hits(input_results_dir) -> DataFrame:
    return read_csv(path.join(input_results_dir, "hits.csv"), header=0, comment="#", index_col=False)


def process_one_model_output(main_data: DataFrame,
                             input_results_dir: str,
                             model_type: ModelType):
    input_results_path = Path(input_results_dir)
    model_identifier = f"{input_results_path.parent.name} {input_results_path.name}"
    save_item_level_data(main_data, path.join(Preferences.results_dir,
                                              "Category production fit naïve " + ("linguistic" if model_type == ModelType.naïve_linguistic else "sensorimotor"),
                                              f"item-level data"
                                              f" ({model_identifier}).csv"))

    hitrate_stats = save_hitrate_summary_tables(path.basename(input_results_dir), main_data,
                                                model_type, None)

    save_naïve_model_performance_stats(
        results_dir=input_results_dir,
        **hitrate_stats,
        model_type=model_type,
    )


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("path", type=str, help="The path in which to find the results.")
    parser.add_argument("--type", type=str, choices=["linguistic", "sensorimotor"])
    args = parser.parse_args()

    main(args.path,
         model_type=ModelType.naïve_linguistic if args.type == "linguistic" else ModelType.naïve_sensorimotor)

    logger.info("Done!")
