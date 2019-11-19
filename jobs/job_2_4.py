from jobs.job import SensorimotorSAJob, SensorimotorSASpec


class Job_2_4(SensorimotorSAJob):

    # max_sphere_radius -> RAM/G
    RAM = {
        100: 5,
        150: 30,
        198: 55,  # 198 is the largest min edge length, so the threshold below which the graph becomes disconnected
        200: 60,
        250: 120,
    }

    def __init__(self, spec: SensorimotorSASpec, run_for_ticks: int, bailout: int = None):
        super().__init__(
            script_number="2_4",
            script_name="2_4_sensorimotor_tsp.py",
            spec=spec,
            run_for_ticks=run_for_ticks,
            bailout=bailout)

    @property
    def qsub_command(self) -> str:
        cmd = f"qsub"
        # qsub args
        cmd += f" -S {self._python_location}"
        cmd += f" -q {self._queue}"
        cmd += f" -N {self.name}"
        cmd += f" -m e -M c.wingfield@lancaster.ac.uk"
        cmd += f" -l h_vmem={self.RAM[self.spec.max_radius]}G"
        # script
        cmd += f" {self.script_name}"
        # script args
        cmd += f" --accessible_set_capacity {self.spec.accessible_set_capacity}"
        cmd += f" --distance_type {self.spec.distance_type.name}" if self.spec.distance_type else ""
        cmd += f" --max_sphere_radius {self.spec.max_radius}"
        cmd += f" --buffer_capacity {self.spec.buffer_capacity}"
        cmd += f" --buffer_threshold {self.spec.buffer_threshold}"
        cmd += f" --accessible_set_threshold {self.spec.accessible_set_threshold}"
        cmd += f" --length_factor {self.spec.length_factor}"
        cmd += f" --node_decay_median {self.spec.median}"
        cmd += f" --node_decay_sigma {self.spec.sigma}"
        cmd += f" --run_for_ticks {self.run_for_ticks}"
        cmd += f" --bailout {self.bailout}" if self.bailout else ""
        return cmd