from model.utils.job import SensorimotorPropagationJob, SensorimotorPropagationSpec
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

    def __init__(self, spec: SensorimotorPropagationSpec, run_for_ticks: int, bailout: int = None):
        super().__init__(
            script_number="2_4",
            script_name="2_4_sensorimotor_tsp.py",
            spec=spec,
            run_for_ticks=run_for_ticks,
            bailout=bailout)

    @property
    def command(self) -> str:
        cmd = self.script_name
        cmd += f" --distance_type {self.spec.distance_type.name}" if self.spec.distance_type else ""
        cmd += f" --max_sphere_radius {self.spec.max_radius}"
        cmd += f" --accessible_set_capacity {self.spec.accessible_set_capacity}"
        cmd += f" --buffer_capacity {self.spec.buffer_capacity}"
        cmd += f" --buffer_threshold {self.spec.buffer_threshold}"
        cmd += f" --accessible_set_threshold {self.spec.accessible_set_threshold}"
        cmd += f" --length_factor {self.spec.length_factor}"
        cmd += f" --node_decay_median {self.spec.node_decay_median}"
        cmd += f" --node_decay_sigma {self.spec.node_decay_sigma}"
        cmd += f" --run_for_ticks {self.run_for_ticks}"
        cmd += f" --bailout {self.bailout}" if self.bailout else ""
        return cmd

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, SensorimotorPropagationSpec)
        return self.RAM[self.spec.max_radius]


if __name__ == '__main__':


    length_factor = 100
    distance_type = DistanceType.Minkowski3
    buffer_capacity = 10
    accessible_set_capacity = 3_000

    specs = [
        SensorimotorPropagationSpec(max_radius=150, buffer_threshold=0.7, accessible_set_threshold=0.3, node_decay_median=500.0, node_decay_sigma=0.9, buffer_capacity=buffer_capacity, accessible_set_capacity=accessible_set_capacity, distance_type=distance_type, length_factor=length_factor),
    ]

    for job in [Job_2_4(spec, run_for_ticks=10_000) for spec in specs]:
        job.submit()
