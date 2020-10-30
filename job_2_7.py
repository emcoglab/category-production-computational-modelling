from typing import Dict

from model.ldm.utils.maths import DistanceType
from model.attenuation_statistic import AttenuationStatistic
from model.utils.job import InteractiveCombinedJob, InteractiveCombinedJobSpec, LinguisticPropagationJobSpec, \
    SensorimotorPropagationJobSpec

logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class Job_2_7(InteractiveCombinedJob):

    # max_sphere_radius -> RAM/G
    SM_RAM: Dict[int, int] = {
        100: 5,
        150: 30,
        198: 55,  # 198 is the largest min edge length, so the threshold below which the graph becomes disconnected
        200: 60,
        250: 120,
    }
    LING_RAM: Dict[str, Dict[int, int]] = {
        "pmi_ngram": {
            1_000: 2,
            3_000: 3,
            10_000: 7,
            30_000: 11,
            40_000: 15,
            60_000: 20,
        },
        "ppmi_ngram": {
            1_000: 2,
            3_000: 3,
            10_000: 5,
            30_000: 7,
            40_000: 9,
            60_000: 11,
        }
    }

    def __init__(self, spec: InteractiveCombinedJobSpec):
        super().__init__(
            script_number="2_7",
            script_name="2_7_interactive_combined.py",
            spec=spec)

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, InteractiveCombinedJobSpec)
        return self.SM_RAM[self.spec.sensorimotor_spec.max_radius] + self.LING_RAM[self.spec.linguistic_spec.model_name][self.spec.linguistic_spec.n_words]


if __name__ == '__main__':

    linguistic_n_words = 60_000
    linguistic_impulse_pruning_threshold = 0.05
    linguistic_node_decay_factor = 0.99
    linguistic_model_radius = 5
    linguistic_corpus_name = "bbc"
    linguistic_accessible_set_capacity = 3_000

    sensorimotor_length_factor = 100
    sensorimotor_max_radius = 150
    sensorimotor_distance_type = DistanceType.Minkowski3
    sensorimotor_accessible_set_capacity = 3_000
    sensorimotor_attenuation = AttenuationStatistic.Prevalence

    bailout = 20_000
    run_for_ticks = 1_000

    specs = [
        InteractiveCombinedJobSpec(
            linguistic_spec=LinguisticPropagationJobSpec(
                accessible_set_threshold=0.3,
                accessible_set_capacity=linguistic_accessible_set_capacity,
                corpus_name=linguistic_corpus_name,
                firing_threshold=0.9,
                impulse_pruning_threshold=linguistic_impulse_pruning_threshold,
                length_factor=10,
                model_name="ppmi_ngram",
                node_decay_factor=linguistic_node_decay_factor,
                model_radius=linguistic_model_radius,
                edge_decay_sd=15,
                n_words=linguistic_n_words,
                pruning=None,
                pruning_type=None,
                bailout=bailout,
                run_for_ticks=run_for_ticks,
            ),
            sensorimotor_spec=SensorimotorPropagationJobSpec(
                accessible_set_threshold=0.3,
                accessible_set_capacity=sensorimotor_accessible_set_capacity,
                distance_type=sensorimotor_distance_type,
                length_factor=sensorimotor_length_factor,
                node_decay_median=500.0,
                node_decay_sigma=0.9,
                attenuation_statistic=sensorimotor_attenuation,
                max_radius=sensorimotor_max_radius,
                bailout=bailout,
                run_for_ticks=run_for_ticks,
            ),
            buffer_threshold=0.7,
            buffer_capacity_linguistic_items=12,
            buffer_capacity_sensorimotor_items=9,
            lc_to_smc_delay=100,
            smc_to_lc_delay=150,
            inter_component_attenuation=0.7,
        )
    ]

    for job in [Job_2_7(spec) for spec in specs]:
        job.run_locally(extra_arguments="--sensorimotor_use_prepruned")
