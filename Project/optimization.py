from iris_database import IrisSystem
import json

system = IrisSystem(db_path='Database/iris_db_syn')
# system.create_tables()

parameters = {
    "detectors" : ['ORB'],
    "kp_size_min" : [0, 3],
    "kp_size_max" : [70],
    "test_size_diff" : 40,
    "test_size_same" : 40,
    "dratio_list" : [0.88, 0.95],
    "stdev_angle_list" : [10],
    "stdev_dist_list":  [0.07, 0.1],
    "from_db": True
    }

results = system.optimization_test(**parameters)

with open(f'test3.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)
