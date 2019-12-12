#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Compare distance-only model to Briony's category production actual responses.

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

from pandas import DataFrame, read_csv

from ldm.utils.logging import date_format, log_message
from category_production.category_production import ColNames as CPColNames
from evaluation.category_production import ModelType, prepare_category_production_data, process_one_model_output_distance_only

logger = logging.getLogger(__name__)


def main(input_results_dir: str, model_type: ModelType):
    logger.info(path.basename(input_results_dir))
    assert model_type in [ModelType.linguistic_distance_only, ModelType.sensorimotor_distance_only]

    main_data = prepare_category_production_data(model_type)

    # Add model hit column
    model_hits: DataFrame = get_distance_only_model_hits(input_results_dir)
    main_data = main_data.merge(model_hits,
                                # Category/Response column names depend on model type
                                # We need to join on the right ones so it matches up compared to
                                on=([CPColNames.Category, CPColNames.Response]
                                    if model_type == ModelType.linguistic_distance_only
                                    else [CPColNames.CategorySensorimotor, CPColNames.ResponseSensorimotor]),
                                how="left")

    process_one_model_output_distance_only(main_data, model_type, input_results_dir, min_first_rank_freq=None)


def get_distance_only_model_hits(input_results_dir) -> DataFrame:
    return read_csv(path.join(input_results_dir, "hits.csv"), header=0, comment="#", index_col=False)


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("path", type=str, help="The path in which to find the results.")
    parser.add_argument("--type", type=str, choices=["linguistic", "sensorimotor"])
    args = parser.parse_args()

    main(args.path,
         model_type=ModelType.linguistic_distance_only if args.type == "linguistic" else ModelType.sensorimotor_distance_only)

    logger.info("Done!")
