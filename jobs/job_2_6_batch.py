"""
Submits a job batch
"""
from jobs.job import NaïveLinguisticSpec, NaïveSensorimotorSpec
from jobs.job_2_6 import Job_2_6
from ldm.utils.maths import DistanceType

if __name__ == '__main__':

    length_factor = 100

    specs = [
        NaïveSensorimotorSpec(length_factor=length_factor, distance_type=DistanceType.correlation),
        NaïveSensorimotorSpec(length_factor=length_factor, distance_type=DistanceType.cosine),
        NaïveSensorimotorSpec(length_factor=length_factor, distance_type=DistanceType.Euclidean),
        NaïveSensorimotorSpec(length_factor=length_factor, distance_type=DistanceType.Minkowski3),
    ]

    for job in [Job_2_6(spec) for spec in specs]:
        job.submit()
