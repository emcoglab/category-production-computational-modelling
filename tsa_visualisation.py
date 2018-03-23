"""
===========================
Visualisation for TemporalSpreadingActivations.
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

from matplotlib.backends.backend_pdf import PdfPages

from temporal_spreading_activation import TemporalSpreadingActivation

logger = logging.getLogger(__name__)


def run_with_pdf_output(tsa: TemporalSpreadingActivation, n_steps: int, path: str):

    with PdfPages(path) as pdf:

        i = 0
        pos = tsa.draw_graph(pdf=pdf, frame_label=str(i))

        for i in range(1, n_steps+1):
            logger.info(f"CLOCK = {i}")
            tsa.tick()
            tsa.draw_graph(pdf=pdf, pos=pos, frame_label=str(i))
