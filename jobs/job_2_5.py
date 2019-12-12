import logging
from functools import partial

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
            script_name="2_5_linguistic_distance_only.py",
            spec=spec)
        assert isinstance(self.spec, NaïveLinguisticSpec)
        # For running locally
        self.main = partial(__import__(self.module_name).main,
                            n_words=self.spec.n_words,
                            corpus_name=self.spec.corpus_name,
                            model_name=self.spec.model_name,
                            radius=self.spec.model_radius,
                            distance_type=self.spec.distance_type)

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


logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"

if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)

    n_words = 40_000
    model_radius = 5
    corpus_name = "bbc"

    specs = [
        NaïveLinguisticSpec(n_words=n_words, model_name="pmi_ngram", model_radius=model_radius, corpus_name=corpus_name),
        NaïveLinguisticSpec(n_words=n_words, model_name="ppmi_ngram", model_radius=model_radius, corpus_name=corpus_name),
        NaïveLinguisticSpec(n_words=n_words, model_name="log_ngram", model_radius=model_radius, corpus_name=corpus_name),
    ]

    for job in [Job_2_5(spec) for spec in specs]:
        job.main()
