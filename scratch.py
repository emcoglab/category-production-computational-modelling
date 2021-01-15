from framework.cognitive_model.combined_cognitive_model import InterComponentMapping
from framework.cognitive_model.sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

sensorimotor_vocab = set(SensorimotorNorms(use_breng_translation=True).iter_words())
linguistic_vocab = {"part", "of", "a", "pharmacy", "chemist"}
mapping = InterComponentMapping(linguistic_vocab=linguistic_vocab, sensorimotor_vocab=sensorimotor_vocab, ignore_identity_mapping=True)

pass
