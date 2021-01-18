from os import path
from pathlib import Path
from threading import Thread
from typing import Optional

from framework.cli.lookups import get_corpus_from_name, get_model_from_params
from framework.cognitive_model.ldm.corpus.indexing import FreqDist
from framework.cognitive_model.ldm.model.base import DistributionalSemanticModel
from framework.cognitive_model.version import VERSION
from framework.cognitive_model.preferences.preferences import Preferences
from framework.cli.job import BufferedSensorimotorPropagationJobSpec, Job, LinguisticPropagationJobSpec, NoninteractiveCombinedJobSpec

logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


# noinspection PyPep8Naming
class Job_3_3(Job):

    def __init__(self, sm_spec: BufferedSensorimotorPropagationJobSpec, ling_spec: LinguisticPropagationJobSpec,
                 manual_cut_off: Optional[int] = None
                 ):
        super().__init__(
            script_number="3_3",
            script_name="3_3_cp_comparison_combined_noninteractive.py",
            spec=NoninteractiveCombinedJobSpec(linguistic_spec=ling_spec, sensorimotor_spec=sm_spec))
        self.manual_cut_off: Optional[int] = manual_cut_off

        assert isinstance(self.spec, NoninteractiveCombinedJobSpec)
        assert isinstance(self.spec.sensorimotor_spec, BufferedSensorimotorPropagationJobSpec)

        # TODO: there is now the JobSpec classes, the model_spec dictionaries and the models themselves.  And it's not
        #  obvious where something like this should live, but it should be accessible in more than one place, because
        #  right now it's just copied from the associated script.
        self._sm_dir: str = path.join(Preferences.output_dir,
                                      "Category production",
                                      f"Sensorimotor {VERSION}",
                                      f"{self.spec.sensorimotor_spec.distance_type.name}"
                                      f" length {self.spec.sensorimotor_spec.length_factor}"
                                      f" att {self.spec.sensorimotor_spec.attenuation_statistic.name};"
                                      f" max-r {self.spec.sensorimotor_spec.max_radius};"
                                      f" n-decay-m {self.spec.sensorimotor_spec.node_decay_median:.1f};"
                                      f" n-decay-σ {self.spec.sensorimotor_spec.node_decay_sigma};"
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
                                   f" {self.spec.linguistic_spec.n_words:,} words, length {self.spec.linguistic_spec.length_factor};"
                                   f" firing-θ {self.spec.linguistic_spec.firing_threshold};"
                                   f" n-decay-f {self.spec.linguistic_spec.node_decay_factor};"
                                   f" e-decay-sd {self.spec.linguistic_spec.edge_decay_sd:.1f};"
                                   f" imp-prune-θ {self.spec.linguistic_spec.impulse_pruning_threshold};"
                                   f" run-for {self.spec.linguistic_spec.run_for_ticks};"
                                   f" bail {self.spec.linguistic_spec.bailout}")

    @property
    def command(self) -> str:
        # No call to super().command, as that contains the specs' individual args
        cmd = self.script_name
        cmd += " "  # separates args from script name
        # script args
        cmd += f"--linguistic_path \"{self._ling_dir}\" "
        cmd += f"--sensorimotor_path \"{self._sm_dir}\" "
        if self.manual_cut_off is not None:
            cmd += f" --manual-cut-off {self.manual_cut_off} "
        return cmd

    @property
    def _ram_requirement_g(self):
        return 5


if __name__ == '__main__':

    sm_specs = jobs = [
        s
        for s in BufferedSensorimotorPropagationJobSpec.load_multiple(
            Path(Path(__file__).parent, "job_specifications/job_cognition_paper_sensorimotor.yaml"))
    ]

    ling_specs = [
        s
        for s in LinguisticPropagationJobSpec.load_multiple(
            Path(Path(__file__).parent, "job_specifications/job_cognition_paper_linguistic.yaml"))
    ]

    threads = [
        Thread(target=Job_3_3(sm_spec, ling_spec).run_locally)
        for ling_spec in ling_specs
        for sm_spec in sm_specs
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]

    from framework.cognitive_model.utils.email import Emailer
    Emailer(Preferences.email_connection_details_path).send_email(f"Done running {path.basename(__file__)} parallel batch.", Preferences.target_email_address)
