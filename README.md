# Iris Recognition System

The Iris Recognition System is a project aimed at developing a robust tool for iris image analysis and recognition. It includes planned features such as advanced extraction and comparison of iris data, performance optimization using Random Forest Classifiers over keypoints, and improvements for challenging conditions. The system includes a database for efficient data management and possible future enhancements for real-time applications with a GUI or mobile app. Forked from [**andreibercu/iris-recognition**](https://github.com/andreibercu/iris-recognition), the project has been updated to Python 3.9.x and is designed to enhance iris recognition accuracy and speed.

## Table of Contents

- [Development](#development)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Development
- [ ] **Improved Iris Analysis**: Improving analysis speed and extraction of features from iris images and creating a dictionary for recognition. Iris recognition accuracy is planned to be improved through keypoint classifying using Random Forest Classifiers or else, see [Match or No Match: Keypoint Filtering Based on Matching Probability Youtube Video](https://www.youtube.com/watch?v=4jV3S04ejFc&t=521s) and [Keypoint Recognition Using Random Forests and Random Ferns by V. Lepetit & P. Fua](https://link.springer.com/chapter/10.1007/978-1-4471-4929-3_9). In order to create a test data, functions is applied, see [Usage](#usage)
- [ ] **Performance Improvements**: Enhancements for recognition performance and speed, especially under suboptimal camera conditions.
- [x] **Database Integration**: Save and retrieve iris data using a database.
  - [ ] Optimization for faster query.
- [ ] **Future Developments**: Plans to create a GUI or mobile app for dynamic and in real-time use.

## Installation

For the latest release:
```
pip install iris_recognition_system
```
or

#### 1. Clone the repository:
```
git clone https://github.com/elymsyr/iris-recognition.git
cd iris-recognition
```

#### 1. Create a virtual environment:
```
conda create -n venv_name python=3.9
```

#### 1. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage
- **`IrisSystem` Class**: This class provides a comprehensive solution for managing an iris recognition database, including the storage, analysis, and comparison of iris data.

#### Example Usage for Creating Iris Database
```
from iris_system.iris_database import IrisSystem
from iris_system.iris_recognition import IrisRecognizer
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
```

- **`IrisSystemOptimizationTest` Class**: This class extends the `IrisSystem` to provide advanced optimization testing for iris recognition parameters. It includes functionality to:

  - Randomly select and analyze iris data from the database.
  - Test various parameters to identify optimal settings for improved iris recognition accuracy.
  - Process and analyze test results to determine the best configuration.
  
  - **Key Methods**:
    - `random_iris_tag(iris_id: int)`: Retrieves a random iris tag associated with a given ID.
    - `optimization_test(...)`: Conducts optimization tests by analyzing random selections from the current database or images.
    - `read_results(results: dict)`: Analyzes and interprets the optimization test results.
    - `get_unique_iris_ids()`: Retrieves a list of unique iris IDs from the database.
    - `key_points_classify(...)`: Classifies key points and prepares data for export.

#### Example Usage for Creating Test Data
```
from iris_system.iris_database import IrisSystemOptimizationTest
from iris_system.iris_recognition import IrisRecognizer
import json, csv

recognizer = IrisRecognizer()
system = IrisSystemOptimizationTest(db_path='Database/iris_db_syn_orb_0_100', recognizer=recognizer)

parameters = {
    "test_size_diff" : 100,
    "test_size_same" : 100,
    "dratio_list" : [0.92, 0.88, 0.7],
    "stdev_angle_list" : [5, 10, 15],
    "stdev_dist_list":  [0.08, 0.1]
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
```

See `iris_system.iris_database` for more...

- **`IrisRecognizer` Class**: This class focuses on detecting and analyzing the iris and pupil from eye images using advanced image processing techniques. It provides several methods to handle the entire process of iris recognition, from loading images to detecting boundaries and finding key points.

See `iris_system.iris_recognition` for more...

## Contributing
Contributions are welcome! Please submit a pull request with your changes or improvements. Ensure that you follow the project's coding standards and provide relevant documentation. See the [CONTRIBUTING](CONTRIBUTING.md) file for details.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

## Contact
For questions or further information, please contact [orhun868@gmail.com].
