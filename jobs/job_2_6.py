from jobs.job import Job, NaïveSensorimotorSpec


class Job_2_6(Job):

    # RAM/G
    RAM = 60

    def __init__(self, spec: NaïveSensorimotorSpec):
        super().__init__(
            script_number="2_6",
            script_name="2_6_naïve_sensorimotor.py",
            spec=spec)

    @property
    def qsub_command(self) -> str:
        cmd = f"qsub"
        # qsub args
        cmd += f" -S {self._python_location}"
        cmd += f" -q {self._queue}"
        cmd += f" -N {self.name}"
        cmd += f" -m e -M c.wingfield@lancaster.ac.uk"
        cmd += f" -l h_vmem={self.RAM}G"
        # script
        cmd += f" {self.script_name}"
        # script args
        cmd += f" --distance_type {self.spec.distance_type.name}" if self.spec.distance_type else ""
        cmd += f" --length_factor {self.spec.length_factor}"
        return cmd
