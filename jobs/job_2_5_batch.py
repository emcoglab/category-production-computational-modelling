"""
Submits a job batch
"""
import logging

from jobs.job import NaïveLinguisticSpec
from jobs.job_2_5 import Job_2_5

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
