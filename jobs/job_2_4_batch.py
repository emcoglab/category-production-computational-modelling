"""
Submits a job batch
"""
from jobs.job import SensorimotorSASpec
from jobs.job_2_4 import Job_2_4
from ldm.utils.maths import DistanceType

if __name__ == '__main__':

    length_factor = 100
    distance_type = DistanceType.Minkowski3
    buffer_capacity = 10
    accessible_set_capacity = 3_000

    specs = [
        SensorimotorSASpec(max_radius=198, buffer_threshold=0.9, accessible_set_threshold=0.5, median=500, sigma=0.3, buffer_capacity=buffer_capacity, accessible_set_capacity=accessible_set_capacity, distance_type=distance_type, length_factor=length_factor),
        SensorimotorSASpec(max_radius=150, buffer_threshold=0.7, accessible_set_threshold=0.3, median=500, sigma=0.3, buffer_capacity=buffer_capacity, accessible_set_capacity=accessible_set_capacity, distance_type=distance_type, length_factor=length_factor),
        SensorimotorSASpec(max_radius=198, buffer_threshold=0.9, accessible_set_threshold=0.3, median=500, sigma=0.9, buffer_capacity=buffer_capacity, accessible_set_capacity=accessible_set_capacity, distance_type=distance_type, length_factor=length_factor),
        SensorimotorSASpec(max_radius=150, buffer_threshold=0.7, accessible_set_threshold=0.3, median=100, sigma=0.9, buffer_capacity=buffer_capacity, accessible_set_capacity=accessible_set_capacity, distance_type=distance_type, length_factor=length_factor),
    ]

    for job in [Job_2_4(spec, run_for_ticks=10_000) for spec in specs]:
        job.submit()
