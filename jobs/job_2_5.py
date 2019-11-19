from jobs.job import Job, NaïveLinguisticSpec


class Job_2_5(Job):

    # n_words -> RAM/G
    RAM = {
        3_000:    3,
        10_000:  20,
        20_000:  65,
        30_000: 120,
        40_000: 120,
    }

    def __init__(self, spec: NaïveLinguisticSpec):
        super().__init__(
            script_number="2_5",
            script_name="2_5_naïve_linguistic.py",
            spec=spec)

    @property
    def qsub_command(self) -> str:
        cmd = f"qsub"
        # qsub args
        cmd += f" -S {self._python_location}"
        cmd += f" -q {self._queue}"
        cmd += f" -N {self.name}"
        cmd += f" -m e -M c.wingfield@lancaster.ac.uk"
        cmd += f" -l h_vmem={self.RAM[self.spec.n_words]}G"
        # script
        cmd += f" {self.script_name}"
        # script args
        cmd += f" --corpus_name {self.spec.corpus_name}"
        cmd += f" --model_name {self.spec.model_name}"
        cmd += f" --radius {self.spec.model_radius}"
        cmd += f" --distance_type {self.spec.distance_type.name}" if self.spec.distance_type else ""
        cmd += f" --length_factor {self.spec.length_factor}"
        return cmd
