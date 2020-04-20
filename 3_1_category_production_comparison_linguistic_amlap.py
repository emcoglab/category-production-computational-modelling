"""
===========================
Compare model to Briony's category production actual responses.
Also works for the one-hop linguistic model.

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
from typing import Optional

from pandas import DataFrame

from sensorimotor_norms.config.config import Config as SMConfig; SMConfig(use_config_overrides_from_file=path.join(path.dirname(__file__), "config_sensorimotor.yaml"))
from evaluation.column_names import TTFA
from ldm.utils.logging import date_format, log_message
from category_production.category_production import CategoryProduction

from evaluation.category_production import get_n_words_from_path_linguistic, get_model_ttfas_for_category_linguistic, \
    add_ttfa_column, get_firing_threshold_from_path_linguistic, ModelType, find_output_dirs, \
    prepare_category_production_data, add_model_hit_column, save_item_level_data, get_hitrate_summary_tables, \
    save_hitrate_summary_tables, frac_within_sd_of_hitrate_mean, MODEL_HITRATE, drop_missing_data_to_add_types, \
    save_hitrate_graphs

logger = logging.getLogger(__name__)

CP = CategoryProduction()


def main(input_results_dir: str,
         min_first_rank_freq: int,
         variant: str,
         conscious_access_threshold: Optional[float] = None,
         ):

    if variant == "full":
        model_type = ModelType.linguistic
    elif variant == "one-hop":
        model_type = ModelType.linguistic_one_hop
    else:
        raise NotImplementedError()

    model_output_dirs = find_output_dirs(root_dir=input_results_dir)

    for model_output_dir in model_output_dirs:
        logger.info(path.basename(model_output_dir))
        main_data = prepare_category_production_data(model_type)

        if conscious_access_threshold is None:
            ft = get_firing_threshold_from_path_linguistic(model_output_dir)
            logger.info(f"No CAT provided, using FT instead ({ft})")
            this_cat = ft
        else:
            this_cat = conscious_access_threshold

        n_words = get_n_words_from_path_linguistic(model_output_dir)

        ttfas = {
            category: get_model_ttfas_for_category_linguistic(category, model_output_dir, n_words, this_cat)
            for category in CP.category_labels
        }
        add_ttfa_column(main_data, model_type=model_type, ttfas=ttfas)
        add_model_hit_column(main_data)

        process_one_model_output(main_data, model_type, model_output_dir, min_first_rank_freq, this_cat)

def process_one_model_output(main_data: DataFrame,
                             model_type: ModelType,
                             input_results_dir: str,
                             min_first_rank_freq: Optional[int],
                             conscious_access_threshold: Optional[float],
                             ):
    assert model_type in [ModelType.linguistic, ModelType.sensorimotor, ModelType.linguistic_one_hop,
                          ModelType.sensorimotor_one_hop]
    input_results_path = Path(input_results_dir)
    model_identifier = f"{input_results_path.parent.name} {input_results_path.name}"
    save_item_level_data(main_data, path.join("/Users/cai/Box Sync/LANGBOOT Project/Conferences/AMLaP 2020/graphs",
                                              model_type.model_output_dirname,
                                              f"item-level data ({model_identifier})"
                                              + (
                                                  f" CAT={conscious_access_threshold}" if conscious_access_threshold is not None else "") +
                                              ".csv"))

    if conscious_access_threshold is not None:
        file_suffix = f"({model_identifier}) CAT={conscious_access_threshold}"
    else:
        file_suffix = f"({model_identifier})"

    hitrates_per_rpf, hitrates_per_rmr = get_hitrate_summary_tables(main_data, model_type)

    # Compute hitrate fits
    hitrate_fit_rpf = frac_within_sd_of_hitrate_mean(hitrates_per_rpf, test_column=MODEL_HITRATE,
                                                     only_before_sd_includes_0=False)
    hitrate_fit_rmr = frac_within_sd_of_hitrate_mean(hitrates_per_rmr, test_column=MODEL_HITRATE,
                                                     only_before_sd_includes_0=False)
    hitrate_fit_rpf_head = frac_within_sd_of_hitrate_mean(hitrates_per_rpf, test_column=MODEL_HITRATE,
                                                          only_before_sd_includes_0=True)
    hitrate_fit_rmr_head = frac_within_sd_of_hitrate_mean(hitrates_per_rmr, test_column=MODEL_HITRATE,
                                                          only_before_sd_includes_0=True)

    drop_missing_data_to_add_types(main_data, {TTFA: int})

    save_hitrate_graphs(hitrates_per_rpf, hitrates_per_rmr, model_type, file_suffix, figures_dir="/Users/cai/Box Sync/LANGBOOT Project/Conferences/AMLaP 2020/graphs")


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
    parser.add_argument("-v-", "--variant", type=str, choices=["one-hop", "full"])
    args = parser.parse_args()

    main(args.path, args.min_frf, args.variant, args.cat)

    logger.info("Done!")
