from iris_database import IrisSystemOptimizationTest
from iris_recognition import IrisRecognizer
import json

recognizer = IrisRecognizer()
system = IrisSystemOptimizationTest(db_path='Database/iris_db_syn_orb_0_100', recognizer=recognizer)

parameters = {
    "test_size_diff" : 40,
    "test_size_same" : 40,
    "dratio_list" : [0.88, ],
    "stdev_angle_list" : [10],
    "stdev_dist_list":  [0.1]
    }

results = system.optimization_test(**parameters)

with open(f'test.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)
