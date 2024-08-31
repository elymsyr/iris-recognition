from iris_database import IrisSystemOptimizationTest
from iris_recognition import IrisRecognizer
import json

recognizer = IrisRecognizer()
system = IrisSystemOptimizationTest(db_path='Database/iris_db_syn_orb_0_100', recognizer=recognizer)

parameters = {
    "test_size_diff" : 5,
    "test_size_same" : 5,
    "dratio_list" : [0.88, 0.92],
    "stdev_angle_list" : [8,10,12],
    "stdev_dist_list":  [0.1,0.08]
    }

results = system.optimization_test(**parameters)

with open(f'test.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

system.read_results({'results':results})