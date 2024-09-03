from iris_database import IrisSystem
from iris_recognition import IrisRecognizer
from os.path import exists

parameters = {
    'detector': 'ORB',
    'kp_size_min': 0,
    'kp_size_max': 100
}

db_path = f"Database/iris_db_syn_{parameters['detector'].lower()}_{parameters['kp_size_min']}_{parameters['kp_size_max']}"

recognizer = IrisRecognizer(**parameters)
system = IrisSystem(db_path=db_path, recognizer=recognizer)

if not exists(db_path):
    system.create_tables()

system.process_and_store_iris(path='IrisDB/CASIA-Iris-Syn/')
