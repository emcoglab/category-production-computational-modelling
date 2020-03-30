from model.utils.job import SensorimotorSAJob, SensorimotorSASpec
from ldm.utils.maths import DistanceType


logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class Job_2_6(SensorimotorSAJob):

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
            script_number="2_6",
            script_name="2_6_sensorimotor_one_hop.py",
            spec=spec)

    @property
    def command(self) -> str:
        cmd = self.script_name
        # script args
        cmd += f" --distance_type {self.spec.distance_type.name}" if self.spec.distance_type else ""
        cmd += f" --max_sphere_radius {self.spec.max_radius}"
        cmd += f" --accessible_set_capacity {self.spec.accessible_set_capacity}"
        cmd += f" --buffer_capacity {self.spec.buffer_capacity}"
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


if __name__ == '__main__':


    length_factor = 100
    distance_type = DistanceType.Minkowski3
    buffer_capacity = 10
    accessible_set_capacity = 3_000

    specs = [
        SensorimotorSASpec(max_radius=198, buffer_threshold=0.9, accessible_set_threshold=0.5, node_decay_median=500, node_decay_sigma=0.3, buffer_capacity=buffer_capacity, accessible_set_capacity=accessible_set_capacity, distance_type=distance_type, length_factor=length_factor),
        SensorimotorSASpec(max_radius=150, buffer_threshold=0.7, accessible_set_threshold=0.3, node_decay_median=500, node_decay_sigma=0.3, buffer_capacity=buffer_capacity, accessible_set_capacity=accessible_set_capacity, distance_type=distance_type, length_factor=length_factor),
        SensorimotorSASpec(max_radius=198, buffer_threshold=0.9, accessible_set_threshold=0.3, node_decay_median=500, node_decay_sigma=0.9, buffer_capacity=buffer_capacity, accessible_set_capacity=accessible_set_capacity, distance_type=distance_type, length_factor=length_factor),
        SensorimotorSASpec(max_radius=150, buffer_threshold=0.7, accessible_set_threshold=0.3, node_decay_median=100, node_decay_sigma=0.9, buffer_capacity=buffer_capacity, accessible_set_capacity=accessible_set_capacity, distance_type=distance_type, length_factor=length_factor),
        SensorimotorSASpec(max_radius=150, buffer_threshold=0.7, accessible_set_threshold=0.3, node_decay_median=500, node_decay_sigma=0.9, buffer_capacity=buffer_capacity, accessible_set_capacity=accessible_set_capacity, distance_type=distance_type, length_factor=length_factor),
        SensorimotorSASpec(max_radius=150, buffer_threshold=0.7, accessible_set_threshold=0.3, node_decay_median=75,  node_decay_sigma=0.9, buffer_capacity=buffer_capacity, accessible_set_capacity=accessible_set_capacity, distance_type=distance_type, length_factor=length_factor),
        SensorimotorSASpec(max_radius=198, buffer_threshold=0.7, accessible_set_threshold=0.5, node_decay_median=500, node_decay_sigma=0.3, buffer_capacity=buffer_capacity, accessible_set_capacity=accessible_set_capacity, distance_type=distance_type, length_factor=length_factor),
    ]

    for job in [Job_2_6(spec) for spec in specs]:
        job.submit()
