from iris_database import IrisSystem
from iris_recognition import IrisRecognizer
from os.path import exists

parameters = {
    'detector': 'ORB',
    'kp_size_min': 0,
    'kp_size_max': 100
}

db_path_to_check = f"Database/iris_db_syn_{parameters['detector'].lower()}_{parameters['kp_size_min']}_{parameters['kp_size_max']}"
db_path = f"Database/iris_db_syn_{parameters['detector'].lower()}_{parameters['kp_size_min']}_{parameters['kp_size_max']}_test"

recognizer = IrisRecognizer(**parameters)
system = IrisSystem(db_path=db_path, recognizer=recognizer)

if not exists(db_path):
    system.create_tables()

system.process_and_store_iris(path='IrisDB/CASIA-Iris-Syn/', db_names_to_check=db_path_to_check)
