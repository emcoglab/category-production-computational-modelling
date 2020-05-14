from model.sensorimotor_components import NormAttenuationStatistic
from model.utils.job import SensorimotorPropagationJob, BufferedSensorimotorPropagationJobSpec
from ldm.utils.maths import DistanceType


logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class Job_2_4(SensorimotorPropagationJob):

    # max_sphere_radius -> RAM/G
    RAM = {
        100: 5,
        150: 30,
        198: 55,  # 198 is the largest min edge length, so the threshold below which the graph becomes disconnected
        200: 60,
        250: 120,
    }

    def __init__(self, spec: BufferedSensorimotorPropagationJobSpec):
        super().__init__(
            script_number="2_4",
            script_name="2_4_sensorimotor_tsp.py",
            spec=spec)

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, BufferedSensorimotorPropagationJobSpec)
        return self.RAM[self.spec.max_radius]


if __name__ == '__main__':

    length_factor = 100
    distance_type = DistanceType.Minkowski3
    buffer_capacity = 10
    accessible_set_capacity = 3_000
    attenuation = NormAttenuationStatistic.Prevalence

    specs = [
        BufferedSensorimotorPropagationJobSpec(max_radius=150, buffer_threshold=0.7, accessible_set_threshold=0.3, node_decay_median=500.0, node_decay_sigma=0.9, buffer_capacity=buffer_capacity, accessible_set_capacity=accessible_set_capacity, distance_type=distance_type, length_factor=length_factor, run_for_ticks=10_000, bailout=None, attenuation_statistic=attenuation),
    ]

    for job in [Job_2_4(spec) for spec in specs]:
        job.submit()
