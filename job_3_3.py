from os import path
from threading import Thread
from typing import Optional

from cli.lookups import get_corpus_from_name, get_model_from_params
from ldm.corpus.indexing import FreqDist
from ldm.model.base import DistributionalSemanticModel
from ldm.utils.maths import DistanceType
from model.sensorimotor_components import NormAttenuationStatistic
from model.utils.job import SensorimotorPropagationJobSpec, Job, LinguisticPropagationJobSpec, NoninteractiveCombinedJobSpec
from model.version import VERSION
from preferences import Preferences

logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


# noinspection PyPep8Naming
class Job_3_3(Job):

    def __init__(self, sm_spec: SensorimotorPropagationJobSpec, ling_spec: LinguisticPropagationJobSpec,
                 sm_attenuate: NormAttenuationStatistic,
                 manual_cut_off: Optional[int] = None
                 ):
        super().__init__(
            script_number="3_3",
            script_name="3_3_cp_comparison_combined_noninteractive.py",
            spec=NoninteractiveCombinedJobSpec(linguistic_spec=ling_spec, sensorimotor_spec=sm_spec))
        self.sm_attenuate: NormAttenuationStatistic = sm_attenuate
        self.manual_cut_off: Optional[int] = manual_cut_off

        assert isinstance(self.spec, NoninteractiveCombinedJobSpec)

        # TODO: there is now the JobSpec classes, the model_spec dictionaries and the models themselves.  And it's not
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
                                      f" run-for {self.spec.sensorimotor_spec.run_for_ticks};"
                                      f" bail {self.spec.sensorimotor_spec.bailout}")

        # TODO: This is absurd
        corpus = get_corpus_from_name(self.spec.linguistic_spec.corpus_name)
        freq_dist = FreqDist.load(corpus.freq_dist_path)
        distributional_model: DistributionalSemanticModel = get_model_from_params(corpus, freq_dist, self.spec.linguistic_spec.model_name, self.spec.linguistic_spec.model_radius)
        # TODO: here too
        self._ling_dir = path.join(Preferences.output_dir,
                                   "Category production",
                                   f"Linguistic {VERSION}",
                                   f"{distributional_model.name}"
                                   f" {self.spec.linguistic_spec.n_words:,} words, length {self.spec.linguistic_spec.length_factor}",
                                   f"firing-θ {self.spec.linguistic_spec.firing_threshold};"
                                   f" n-decay-f {self.spec.linguistic_spec.node_decay_factor};"
                                   f" e-decay-sd {self.spec.linguistic_spec.edge_decay_sd:.1f};"
                                   f" imp-prune-θ {self.spec.linguistic_spec.impulse_pruning_threshold};"
                                   f" run-for {self.spec.linguistic_spec.run_for_ticks};"
                                   f" bail {self.spec.linguistic_spec.bailout}")

    @property
    def command(self) -> str:
        cmd = super().command
        # script args
        cmd += f"--linguistic_path \"{self._ling_dir}\""
        cmd += f"--sensorimotor_path \"{self._sm_dir}\""
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
        SensorimotorPropagationJobSpec(max_radius=150, node_decay_median=75.0,  node_decay_sigma=0.9, accessible_set_threshold=0.3, buffer_threshold=0.7, buffer_capacity=sm_buffer_capacity, accessible_set_capacity=sm_accessible_set_capacity, distance_type=sm_distance_type, length_factor=sm_length_factor, bailout=sm_bail, run_for_ticks=sm_rft, attenuation_statistic=attenuate),
        SensorimotorPropagationJobSpec(max_radius=150, node_decay_median=100.0, node_decay_sigma=0.9, accessible_set_threshold=0.3, buffer_threshold=0.7, buffer_capacity=sm_buffer_capacity, accessible_set_capacity=sm_accessible_set_capacity, distance_type=sm_distance_type, length_factor=sm_length_factor, bailout=sm_bail, run_for_ticks=sm_rft, attenuation_statistic=attenuate),
        SensorimotorPropagationJobSpec(max_radius=150, node_decay_median=500.0, node_decay_sigma=0.3, accessible_set_threshold=0.3, buffer_threshold=0.7, buffer_capacity=sm_buffer_capacity, accessible_set_capacity=sm_accessible_set_capacity, distance_type=sm_distance_type, length_factor=sm_length_factor, bailout=sm_bail, run_for_ticks=sm_rft, attenuation_statistic=attenuate),
        SensorimotorPropagationJobSpec(max_radius=150, node_decay_median=500.0, node_decay_sigma=0.9, accessible_set_threshold=0.3, buffer_threshold=0.7, buffer_capacity=sm_buffer_capacity, accessible_set_capacity=sm_accessible_set_capacity, distance_type=sm_distance_type, length_factor=sm_length_factor, bailout=sm_bail, run_for_ticks=sm_rft, attenuation_statistic=attenuate),
        SensorimotorPropagationJobSpec(max_radius=198, node_decay_median=500.0, node_decay_sigma=0.3, accessible_set_threshold=0.5, buffer_threshold=0.7, buffer_capacity=sm_buffer_capacity, accessible_set_capacity=sm_accessible_set_capacity, distance_type=sm_distance_type, length_factor=sm_length_factor, bailout=sm_bail, run_for_ticks=sm_rft, attenuation_statistic=attenuate),
        SensorimotorPropagationJobSpec(max_radius=198, node_decay_median=500.0, node_decay_sigma=0.3, accessible_set_threshold=0.5, buffer_threshold=0.9, buffer_capacity=sm_buffer_capacity, accessible_set_capacity=sm_accessible_set_capacity, distance_type=sm_distance_type, length_factor=sm_length_factor, bailout=sm_bail, run_for_ticks=sm_rft, attenuation_statistic=attenuate),
        SensorimotorPropagationJobSpec(max_radius=198, node_decay_median=500.0, node_decay_sigma=0.9, accessible_set_threshold=0.3, buffer_threshold=0.9, buffer_capacity=sm_buffer_capacity, accessible_set_capacity=sm_accessible_set_capacity, distance_type=sm_distance_type, length_factor=sm_length_factor, bailout=sm_bail, run_for_ticks=sm_rft, attenuation_statistic=attenuate),
    ]

    ling_n_words = 40_000
    ling_impulse_pruning_threshold = 0.05
    ling_node_decay_factor = 0.99
    ling_model_radius = 5
    ling_corpus_name = "bbc"
    ling_rft = 3_000
    ling_bail = int(ling_n_words / 2)

    ling_specs = [
        LinguisticPropagationJobSpec(model_name="ppmi_ngram", firing_threshold=0.7, edge_decay_sd=15, impulse_pruning_threshold=ling_impulse_pruning_threshold, node_decay_factor=ling_node_decay_factor, model_radius=ling_model_radius, corpus_name=ling_corpus_name, pruning=None, pruning_type=None, n_words=ling_n_words, length_factor=10, bailout=ling_bail, run_for_ticks=ling_rft),
        LinguisticPropagationJobSpec(model_name="ppmi_ngram", firing_threshold=0.8, edge_decay_sd=15, impulse_pruning_threshold=ling_impulse_pruning_threshold, node_decay_factor=ling_node_decay_factor, model_radius=ling_model_radius, corpus_name=ling_corpus_name, pruning=None, pruning_type=None, n_words=ling_n_words, length_factor=10, bailout=ling_bail, run_for_ticks=ling_rft),
        LinguisticPropagationJobSpec(model_name="ppmi_ngram", firing_threshold=0.9, edge_decay_sd=15, impulse_pruning_threshold=ling_impulse_pruning_threshold, node_decay_factor=ling_node_decay_factor, model_radius=ling_model_radius, corpus_name=ling_corpus_name, pruning=None, pruning_type=None, n_words=ling_n_words, length_factor=10, bailout=ling_bail, run_for_ticks=ling_rft),
    ]

    threads = [
        Thread(target=Job_3_3(sm_spec, ling_spec, attenuate).run_locally)
        for ling_spec in ling_specs
        for sm_spec in sm_specs
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]

    from model.utils.email import Emailer
    Emailer(Preferences.email_connection_details_path).send_email(f"Done running {path.basename(__file__)} parallel batch.", Preferences.target_email_address)
