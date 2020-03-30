from os import path
from threading import Thread
from typing import Optional

from cli.lookups import get_corpus_from_name, get_model_from_params
from ldm.corpus.indexing import FreqDist
from ldm.model.base import DistributionalSemanticModel
from ldm.utils.maths import DistanceType
from model.utils.job import SensorimotorSASpec, Job, LinguisticSASpec, NoninteractiveCombinedSpec
from model.sensorimotor_propagator import NormAttenuationStatistic
from model.version import VERSION
from preferences import Preferences

logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


# noinspection PyPep8Naming
class Job_3_3(Job):

    def __init__(self, sm_spec: SensorimotorSASpec, ling_spec: LinguisticSASpec,
                 sm_attenuate: NormAttenuationStatistic,
                 # TODO: is there a reason these aren't part of the LingusticSASpec and SensorimotorSASpec?
                 ling_rft: int, sm_rft: int, ling_bail: Optional[int], sm_bail: Optional[int],
                 manual_cut_off: Optional[int] = None
                 ):
        super().__init__(
            script_number="3_3",
            script_name="3_3_cp_comparison_combined_noninteractive.py",
            spec=NoninteractiveCombinedSpec(linguistic_spec=ling_spec, sensorimotor_spec=sm_spec))
        self.sm_attenuate: NormAttenuationStatistic = sm_attenuate
        self.ling_rft: int = ling_rft
        self.sm_rft: int = sm_rft
        self.ling_bail: int = ling_bail
        self.sm_bail: int = sm_bail
        self.manual_cut_off: Optional[int] = manual_cut_off

        assert isinstance(self.spec, NoninteractiveCombinedSpec)

        # TODO: there is now the Spec classes, the model_spec dictionaries and the models themselves.  And it's not
        #  obvious where something like this should live, but it should be accessible in more than one place, because
        #  right now it's just copied from the associated script.
        self._sm_dir: str = path.join(Preferences.output_dir,
                                      "Category production",
                                      f"Sensorimotor {VERSION}",
                                      f"{self.spec.sensorimotor_spec.distance_type.name}"
                                      f" length {self.spec.sensorimotor_spec.length_factor}"
                                      f" attenuate {self.sm_attenuate.name}",
                                      f"max-r {self.spec.sensorimotor_spec.max_radius};"
                                      f" n-decay-median {self.spec.sensorimotor_spec.node_decay_median:.1f};"
                                      f" n-decay-sigma {self.spec.sensorimotor_spec.node_decay_sigma};"
                                      f" as-θ {self.spec.sensorimotor_spec.accessible_set_threshold};"
                                      f" as-cap {self.spec.sensorimotor_spec.accessible_set_capacity:,};"
                                      f" buff-θ {self.spec.sensorimotor_spec.buffer_threshold};"
                                      f" buff-cap {self.spec.sensorimotor_spec.buffer_capacity};"
                                      f" run-for {self.sm_rft};"
                                      f" bail {self.sm_bail}")

        # TODO: This is absurd
        corpus = get_corpus_from_name(self.spec.linguistic_spec.corpus_name)
        freq_dist = FreqDist.load(corpus.freq_dist_path)
        distributional_model: DistributionalSemanticModel = get_model_from_params(corpus, freq_dist, self.spec.linguistic_spec.model_name, self.spec.linguistic_spec.model_radius)
        # TODO: here too
        self._ling_dir = path.join(Preferences.output_dir,
                                   "Category production",
                                   f"Linguistic {VERSION}",
                                   f"{distributional_model.name}"
                                   f" {self.spec.linguistic_spec.graph_size:,} words, length {self.spec.linguistic_spec.length_factor}",
                                   f"firing-θ {self.spec.linguistic_spec.firing_threshold};"
                                   f" n-decay-f {self.spec.linguistic_spec.node_decay_factor};"
                                   f" e-decay-sd {self.spec.linguistic_spec.edge_decay_sd:.1f};"
                                   f" imp-prune-θ {self.spec.linguistic_spec.impulse_pruning_threshold};"
                                   f" run-for {self.ling_rft};"
                                   f" bail {self.ling_bail}")

    @property
    def command(self) -> str:
        assert isinstance(self.spec, NoninteractiveCombinedSpec)
        cmd = self.script_name
        # script args
        cmd += f" \"{self._ling_dir}\""
        cmd += f" \"{self._sm_dir}\""
        if self.manual_cut_off is not None:
            cmd += f" --manual-cut-off {self.manual_cut_off}"
        return cmd

    @property
    def _ram_requirement_g(self):
        return 5


if __name__ == '__main__':


    sm_length_factor = 100
    sm_distance_type = DistanceType.Minkowski3
    attenuate = NormAttenuationStatistic.Prevalence
    sm_buffer_capacity = 10
    sm_accessible_set_capacity = 3_000
    sm_rft = 10_000
    sm_bail = None

    sm_specs = [
        SensorimotorSASpec(max_radius=150, node_decay_median=75.0,  node_decay_sigma=0.9, accessible_set_threshold=0.3, buffer_threshold=0.7, buffer_capacity=sm_buffer_capacity, accessible_set_capacity=sm_accessible_set_capacity, distance_type=sm_distance_type, length_factor=sm_length_factor),
        SensorimotorSASpec(max_radius=150, node_decay_median=100.0, node_decay_sigma=0.9, accessible_set_threshold=0.3, buffer_threshold=0.7, buffer_capacity=sm_buffer_capacity, accessible_set_capacity=sm_accessible_set_capacity, distance_type=sm_distance_type, length_factor=sm_length_factor),
        SensorimotorSASpec(max_radius=150, node_decay_median=500.0, node_decay_sigma=0.3, accessible_set_threshold=0.3, buffer_threshold=0.7, buffer_capacity=sm_buffer_capacity, accessible_set_capacity=sm_accessible_set_capacity, distance_type=sm_distance_type, length_factor=sm_length_factor),
        SensorimotorSASpec(max_radius=150, node_decay_median=500.0, node_decay_sigma=0.9, accessible_set_threshold=0.3, buffer_threshold=0.7, buffer_capacity=sm_buffer_capacity, accessible_set_capacity=sm_accessible_set_capacity, distance_type=sm_distance_type, length_factor=sm_length_factor),
        SensorimotorSASpec(max_radius=198, node_decay_median=500.0, node_decay_sigma=0.3, accessible_set_threshold=0.5, buffer_threshold=0.7, buffer_capacity=sm_buffer_capacity, accessible_set_capacity=sm_accessible_set_capacity, distance_type=sm_distance_type, length_factor=sm_length_factor),
        SensorimotorSASpec(max_radius=198, node_decay_median=500.0, node_decay_sigma=0.3, accessible_set_threshold=0.5, buffer_threshold=0.9, buffer_capacity=sm_buffer_capacity, accessible_set_capacity=sm_accessible_set_capacity, distance_type=sm_distance_type, length_factor=sm_length_factor),
        SensorimotorSASpec(max_radius=198, node_decay_median=500.0, node_decay_sigma=0.9, accessible_set_threshold=0.3, buffer_threshold=0.9, buffer_capacity=sm_buffer_capacity, accessible_set_capacity=sm_accessible_set_capacity, distance_type=sm_distance_type, length_factor=sm_length_factor),
    ]

    ling_graph_size = 40_000
    ling_impulse_pruning_threshold = 0.05
    ling_node_decay_factor = 0.99
    ling_model_radius = 5
    ling_corpus_name = "bbc"
    ling_rft = 3_000
    ling_bail = int(ling_graph_size / 2)

    ling_specs = [
        LinguisticSASpec(model_name="ppmi_ngram", firing_threshold=0.7, edge_decay_sd=15, impulse_pruning_threshold=ling_impulse_pruning_threshold, node_decay_factor=ling_node_decay_factor, model_radius=ling_model_radius, corpus_name=ling_corpus_name, pruning=None, graph_size=ling_graph_size, length_factor=10, ),
        LinguisticSASpec(model_name="ppmi_ngram", firing_threshold=0.8, edge_decay_sd=15, impulse_pruning_threshold=ling_impulse_pruning_threshold, node_decay_factor=ling_node_decay_factor, model_radius=ling_model_radius, corpus_name=ling_corpus_name, pruning=None, graph_size=ling_graph_size, length_factor=10, ),
        LinguisticSASpec(model_name="ppmi_ngram", firing_threshold=0.9, edge_decay_sd=15, impulse_pruning_threshold=ling_impulse_pruning_threshold, node_decay_factor=ling_node_decay_factor, model_radius=ling_model_radius, corpus_name=ling_corpus_name, pruning=None, graph_size=ling_graph_size, length_factor=10, ),
    ]

    threads = [
        Thread(target=Job_3_3(sm_spec, ling_spec, attenuate, ling_rft, sm_rft, ling_bail, sm_bail).run_locally)
        for ling_spec in ling_specs
        for sm_spec in sm_specs
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]

    from model.utils.email import Emailer
    Emailer(Preferences.email_connection_details_path).send_email(f"Done running {path.basename(__file__)} parallel batch.", Preferences.target_email_address)
