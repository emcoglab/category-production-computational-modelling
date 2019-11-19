"""
Submits a job batch
"""
from jobs.job import LinguisticSASpec
from jobs.job_2_3 import Job_2_3

if __name__ == '__main__':

    graph_size = 40_000
    length_factor = 100
    bailout = graph_size / 2
    impulse_pruning_threshold = 0.05
    node_decay_factor = 0.99
    model_radius = 5
    corpus_name = "bbc"

    specs = [
        LinguisticSASpec(graph_size=graph_size, model_name="pmi_ngram", length_factor=length_factor, firing_threshold=0.7, edge_decay_sd=35, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name),
        LinguisticSASpec(graph_size=graph_size, model_name="pmi_ngram", length_factor=length_factor, firing_threshold=0.6, edge_decay_sd=25, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name),
        LinguisticSASpec(graph_size=graph_size, model_name="pmi_ngram", length_factor=length_factor, firing_threshold=0.7, edge_decay_sd=30, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name),
        LinguisticSASpec(graph_size=graph_size, model_name="pmi_ngram", length_factor=length_factor, firing_threshold=0.6, edge_decay_sd=30, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name),
        LinguisticSASpec(graph_size=graph_size, model_name="ppmi_ngram", length_factor=length_factor, firing_threshold=0.7, edge_decay_sd=35, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name),
        LinguisticSASpec(graph_size=graph_size, model_name="ppmi_ngram", length_factor=length_factor, firing_threshold=0.5, edge_decay_sd=15, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name),
        LinguisticSASpec(graph_size=graph_size, model_name="ppmi_ngram", length_factor=length_factor, firing_threshold=0.7, edge_decay_sd=20, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name),
        LinguisticSASpec(graph_size=graph_size, model_name="ppmi_ngram", length_factor=length_factor, firing_threshold=0.5, edge_decay_sd=20, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name),
        LinguisticSASpec(graph_size=graph_size, model_name="ppmi_ngram", length_factor=length_factor, firing_threshold=0.5, edge_decay_sd=30, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name),
    ]

    for job in [Job_2_3(spec, run_for_ticks=10_000) for spec in specs]:
        job.submit()
