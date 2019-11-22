#!/Users/cai/Applications/miniconda3/bin/python
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
from os import path

from ldm.utils.logging import date_format, log_message
from evaluation.category_production import add_model_predictor_columns, CATEGORY_PRODUCTION, \
    get_model_ttfas_for_category_sensorimotor, ModelType, find_output_dirs, prepare_category_production_data, \
    process_one_model_output

logger = logging.getLogger(__name__)


def main(input_results_dir: str,
         min_first_rank_freq: int,
         ):

    model_output_dirs = find_output_dirs(root_dir=input_results_dir)

    for model_output_dir in model_output_dirs:
        logger.info(path.basename(model_output_dir))
        main_data = prepare_category_production_data(ModelType.sensorimotor)

        add_model_predictor_columns(main_data, model_type=ModelType.sensorimotor,
                                    ttfas={
                                        category: get_model_ttfas_for_category_sensorimotor(category, input_results_dir)
                                        for category in CATEGORY_PRODUCTION.category_labels_sensorimotor})

        process_one_model_output(main_data, ModelType.sensorimotor, model_output_dir, min_first_rank_freq, None)


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("path", type=str, help="The path in which to find the results.")
    parser.add_argument("min_frf", type=int, nargs="?", default=1,
                        help="The minimum FRF required for zRT and FRF correlations."
                             " Omit to use 1.")
    args = parser.parse_args()

    main(args.path, args.min_frf)

    logger.info("Done!")
