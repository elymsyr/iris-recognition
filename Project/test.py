from iris_database import IrisSystem
from iris_recognition import IrisRecognizer

recognizer = IrisRecognizer()

system = IrisSystem(db_path='Database/iris_db_syn_orb_0_100', recognizer=recognizer)
system.compare_iris(image_tag_1='S6003S00', image_tag_2='S6003S07', show=True)