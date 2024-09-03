from iris_database import IrisSystemOptimizationTest
from iris_recognition import IrisRecognizer
import json, csv

recognizer = IrisRecognizer()
system = IrisSystemOptimizationTest(db_path='Database/iris_db_syn_orb_0_100', recognizer=recognizer)

parameters = {
    "test_size_diff" : 1000,
    "test_size_same" : 1000,
    "dratio_list" : [0.92],
    "stdev_angle_list" : [10],
    "stdev_dist_list":  [0.08]
    }

results = system.optimization_test(**parameters)

# with open(f'test.json', 'w') as json_file:
#     json.dump(results, json_file, indent=4)

# system.read_results({'results':results})

test = system.key_points_classify(results, 0)

with open('test.csv', 'w') as csvfile: 
    writer = csv.DictWriter(csvfile, fieldnames = test[0].keys()) 
    writer.writeheader()
    writer.writerows(test)