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
2018
---------------------------
"""

import argparse
import logging
import sys
from os import path
from typing import Optional

from ldm.utils.logging import date_format, log_message
from evaluation.category_production import get_n_words_from_path_linguistic, get_model_ttfas_for_category_linguistic, \
    add_model_predictor_columns, CATEGORY_PRODUCTION, get_firing_threshold_from_path_linguistic, ModelType, \
    find_output_dirs, prepare_category_production_data, process_one_model_output

logger = logging.getLogger(__name__)


def main(input_results_dir: str,
         min_first_rank_freq: int,
         conscious_access_threshold: Optional[float] = None,
         ):

    model_output_dirs = find_output_dirs(root_dir=input_results_dir)

    for model_output_dir in model_output_dirs:
        logger.info(path.basename(model_output_dir))
        main_data = prepare_category_production_data(ModelType.linguistic)

        if conscious_access_threshold is None:
            ft = get_firing_threshold_from_path_linguistic(model_output_dir)
            logger.info(f"No CAT provided, using FT instead ({ft})")
            this_conscious_access_threshold = ft
        else:
            this_conscious_access_threshold = conscious_access_threshold

        n_words = get_n_words_from_path_linguistic(model_output_dir)

        add_model_predictor_columns(main_data, model_type=ModelType.linguistic,
                                    ttfas={
                                        category: get_model_ttfas_for_category_linguistic(category, model_output_dir, n_words, this_conscious_access_threshold)
                                        for category in CATEGORY_PRODUCTION.category_labels})

        process_one_model_output(main_data, ModelType.linguistic, model_output_dir, min_first_rank_freq, this_conscious_access_threshold)


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("path", type=str, help="The path in which to find the results.")
    parser.add_argument("min_frf", type=int, nargs="?", default=1,
                        help="The minimum FRF required for zRT and FRF correlations."
                             " Omit to use 1.")
    parser.add_argument("cat", type=float, nargs="?", default=None,
                        help="The conscious-access threshold."
                             " Omit to use CAT = firing threshold.")
    args = parser.parse_args()

    main(args.path, args.min_frf, args.cat)

    logger.info("Done!")
