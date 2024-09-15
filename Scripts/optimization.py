from iris_database import IrisSystemOptimizationTest, IrisSystem
from iris_recognition import IrisRecognizer
import json, csv
from os.path import exists
import joblib
from xgboost import Booster

model_path='Scripts/xgboost_model.json'
scaler_path='Scripts/scaler.pkl'

db_path = f"Database/iris_db_syn_xgboost"
"""
Database: Database/iris_db_syn_xgboost
Match avg: 66.02 - 428 - 0
False match avg: 9.67077649527807 - 48 - 0
KP number: 411.66631667366653 - 1269 - 7
"""

# db_path = f"Database/iris_db_syn_orb_0_100"
"""
Database: Database/iris_db_syn_orb_0_100
Match avg: 193.161 - 1063 - 4
False match avg: 20.95075 - 64 - 0
KP number: 898.0443 - 1413 - 102
"""

model = Booster()
model.load_model(model_path)
scaler = joblib.load(scaler_path)

parameters = {
    'detector': 'ORB',
    'kp_size_min': 0,
    'kp_size_max': 1000,
    'model': model,
    'scaler': scaler,
    'model_threshold': 0.5
}

recognizer = IrisRecognizer(**parameters)
system = IrisSystemOptimizationTest(db_path=db_path, recognizer=recognizer)

if not exists(db_path):
    system.create_tables()

parameters = {
    "test_size_diff" : 5000,
    "test_size_same" : 1000,
    "dratio_list" : [0.92],
    "stdev_angle_list" : [10],
    "stdev_dist_list":  [0.08]
    }

results = system.optimization_test(**parameters)

# with open(f'test.json', 'w') as json_file:
#     json.dump(results, json_file, indent=4)

system.read_results({'results':results})

test = system.key_points_classify(results, 0)

with open('kp_classify.csv', 'w') as csvfile: 
    writer = csv.DictWriter(csvfile, fieldnames = test[0].keys())
    writer.writeheader()
    writer.writerows(test)
