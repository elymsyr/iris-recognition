from iris_database import IrisSystem
from iris_recognition import IrisRecognizer

recognizer = IrisRecognizer()
system = IrisSystem(db_path='Database/iris_db_syn_orb_0_100', recognizer=recognizer)
# system.create_tables()
system.process_and_store_iris(path='IrisDB/CASIA-Iris-Syn/')
