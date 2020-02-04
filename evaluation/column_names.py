"""
===========================
Column names.
Names ending with _f require .format()-ing.
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

RANKED_PRODUCTION_FREQUENCY = "RankedProductionFreq"
ROUNDED_MEAN_RANK           = "RoundedMeanRank"
PRODUCTION_PROPORTION       = "ProductionProportion"

MODEL_HIT                   = "ModelHit"
# hitrate is among all categories
# (e.g. 42nd response within all categories, whether or not they had ≥42 responses)
MODEL_HITRATE               = "Model hitrate"
MODEL_HITRATE_PER_CATEGORY  = "Model hitrate per category"
# PARTICIPANT_*_f columns require .format()-ing with participant ID
PARTICIPANT_SAW_CATEGORY_f  = "Participant {0} saw category"
PARTICIPANT_RESPONSE_HIT_f  = "Participant {0} response hit"
PARTICIPANT_HITRATE_All_f   = "Participant {0} hitrate"
PARTICIPANT_HITRATE_PER_CATEGORY_f = "Participant {0} hitrate per category"

TTFA                        = "TTFA"
TICK_ON_WHICH_ACTIVATED     = "Tick on which activated"

ITEM_ENTERED_BUFFER         = "Item entered WM buffer"
REACHED_CAT                 = "Reached conc.acc. θ"

RESPONSE                    = "Response"
ACTIVATION                  = "Activation"

CAT                         = "CAT"
