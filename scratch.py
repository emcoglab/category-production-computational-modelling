"""
===========================
Proof of concept for linguistic spreading activation model.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2017
---------------------------
"""

import sys
import logging
from ldm.core.model.count import LogCoOccurrenceCountModel


logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main():
    pass


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat,evel=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")

