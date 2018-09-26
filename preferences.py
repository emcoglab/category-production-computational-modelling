"""
===========================
Analysis preferences for spreading activation models.
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


class Preferences(object):
    """
    Global preferences for spreading activation models.
    """

    # Paths

    email_connection_details_path = "/Users/caiwingfield/Box Sync/Admin dox/notify@cwcomplex.net.txt"
    target_email_address = "c.wingfield@lancaster.ac.uk"

    graphs_dir = "/Users/caiwingfield/data/graphs/"

    node_distributions_dir = "/Users/caiwingfield/data/node_distributions/"

    output_dir = "/Volumes/Data/spreading activation model/Model output/"
    results_dir = "/Volumes/Data/spreading activation model/Evaluation/"

    figures_dir = "/Volumes/Data/spreading activation model/Figures/"

    graph_sizes = [
        1_000,
        3_000,
        10_000,
        15_000,
        20_000,
        30_000,
        35_000,
        40_000,
    ]

    min_edges_per_node = 10

    sensorimotor_norms_path = "/Users/caiwingfield/Box Sync/LANGBOOT Project/Model/sensorimotor_norms_for_39731_words_low_N_known_removed.csv"
