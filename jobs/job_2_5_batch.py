"""
Submits a job batch
"""
from jobs.job import NaïveLinguisticSpec
from jobs.job_2_5 import Job_2_5

if __name__ == '__main__':

    n_words = 40_000
    length_factor = 100
    model_radius = 5
    corpus_name = "bbc"

    specs = [
        NaïveLinguisticSpec(n_words=n_words, model_name="pmi_ngram", length_factor=length_factor, model_radius=model_radius, corpus_name=corpus_name),
        NaïveLinguisticSpec(n_words=n_words, model_name="ppmi_ngram", length_factor=length_factor, model_radius=model_radius, corpus_name=corpus_name),
    ]

    for job in [Job_2_5(spec) for spec in specs]:
        job.submit()
