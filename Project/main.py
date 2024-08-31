from iris_database import IrisSystem

parameters = {
    'detector': 'ORB',
    'kp_size_max': 70
}

system = IrisSystem(db_path='Database/iris_db_syn_orb_0_70', recognizer_parameters=parameters)
system.create_tables()
system.process_and_store_iris(path='IrisDB/casia-iris-syn-200mb/CASIA-Iris-Syn/')
