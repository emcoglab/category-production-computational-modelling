from pathlib import Path

from model.utils.job import SensorimotorPropagationJob, BufferedSensorimotorPropagationJobSpec


logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class Job_2_4(SensorimotorPropagationJob):

    # max_sphere_radius -> RAM/G
    RAM = {
        1.00: 5,
        1.50: 30,
        1.98: 55,  # 198 is the largest min edge length, so the threshold below which the graph becomes disconnected
        2.00: 60,
        2.50: 120,
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
    job = Job_2_4(BufferedSensorimotorPropagationJobSpec.load(
        Path(Path(__file__).parent, "job_specifications/job_cognition_paper_sensorimotor.yaml")))
    job.run_locally(extra_arguments=["--use_prepruned", "--multiword_divide"])
