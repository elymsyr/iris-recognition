from iris_database import create_system

# db_path_to_check = f"Database/iris_db_syn_{parameters['detector'].lower()}_{parameters['kp_size_min']}_{parameters['kp_size_max']}"
db_path = f"Database/iris_db_syn_xgboost"

system, recognizer = create_system(db_path=db_path, model_path='Scripts/xgboost_model.json', scaler_path='Scripts/scaler.pkl')

system.process_and_store_iris(path='IrisDB/CASIA-Iris-Syn/') # db_names_to_check=db_path_to_check
