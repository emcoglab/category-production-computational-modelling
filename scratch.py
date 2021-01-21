from logging import basicConfig, INFO

from framework.cognitive_model.sensorimotor_norms.sensorimotor_norms import SensorimotorNorms
basicConfig(format='%(asctime)s | %(levelname)s | %(funcName)s @ %(module)s:%(lineno)d |➤ %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=INFO)

SensorimotorNorms(use_breng_translation=True, verbose=True, test_word = "tranquillization")
SensorimotorNorms(use_breng_translation=True, verbose=True, test_word = "tranquilization")
SensorimotorNorms(use_breng_translation=True, verbose=True, test_word = "tranquillisation")
SensorimotorNorms(use_breng_translation=True, verbose=True, test_word = "tranquilisation")

pass
