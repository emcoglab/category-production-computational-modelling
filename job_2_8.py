import logging

from model.utils.job import SensorimotorSAJob, SensorimotorSASpec
from ldm.utils.maths import DistanceType


class Job_2_8(SensorimotorSAJob):

    # max_sphere_radius -> RAM/G
    RAM = {
        100: 5,
        150: 30,
        198: 55,  # 198 is the largest min edge length, so the threshold below which the graph becomes disconnected
        200: 60,
        250: 120,
    }

    def __init__(self, spec: SensorimotorSASpec):
        super().__init__(
            script_number="2_8",
            script_name="2_8_sensorimotor_one_hop.py",
            spec=spec)

    @property
    def command(self) -> str:
        cmd = self.script_name
        # script args
        cmd += f" --distance_type {self.spec.distance_type.name}" if self.spec.distance_type else ""
        cmd += f" --max_sphere_radius {self.spec.max_radius}"
        cmd += f" --buffer_threshold {self.spec.buffer_threshold}"
        cmd += f" --accessible_set_threshold {self.spec.accessible_set_threshold}"
        cmd += f" --length_factor {self.spec.length_factor}"
        cmd += f" --node_decay_median {self.spec.median}"
        cmd += f" --node_decay_sigma {self.spec.sigma}"
        return cmd

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, SensorimotorSASpec)
        return self.RAM[self.spec.max_radius]


logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)

    length_factor = 100
    distance_type = DistanceType.Minkowski3

    specs = [
        SensorimotorSASpec(max_radius=198, buffer_threshold=0.9, accessible_set_threshold=0.5, median=500, sigma=0.3, buffer_capacity=None, accessible_set_capacity=None, distance_type=distance_type, length_factor=length_factor),
        SensorimotorSASpec(max_radius=150, buffer_threshold=0.7, accessible_set_threshold=0.3, median=500, sigma=0.3, buffer_capacity=None, accessible_set_capacity=None, distance_type=distance_type, length_factor=length_factor),
        SensorimotorSASpec(max_radius=198, buffer_threshold=0.9, accessible_set_threshold=0.3, median=500, sigma=0.9, buffer_capacity=None, accessible_set_capacity=None, distance_type=distance_type, length_factor=length_factor),
        SensorimotorSASpec(max_radius=150, buffer_threshold=0.7, accessible_set_threshold=0.3, median=100, sigma=0.9, buffer_capacity=None, accessible_set_capacity=None, distance_type=distance_type, length_factor=length_factor),
    ]

    for job in [Job_2_8(spec) for spec in specs]:
        job.submit()
