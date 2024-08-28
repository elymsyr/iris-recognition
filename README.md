# Iris Recognition System

This project is an iris recognition system ([*See original project.*](https://github.com/andreibercu/iris-recognition)) to analyze and recognize iris images. It includes functionality for feature extraction, comparison of iris features, and performance improvements for various scenarios, including suboptimal camera conditions. The project has been updated to run on Python 3.9.x and includes functionality to save iris data to a database. There are also plans to develop a dynamic system with a GUI or mobile app for real-time use.

## Table of Contents

- [Development](#development)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Development
- [ ] **Improved Iris Analysis**: Improving analyze speed and extraction of features from iris images and creating a dictionary for recognition.
- [ ] **Performance Improvements**: Enhancements for recognition performance and speed, especially under suboptimal camera conditions.
- [x] **Database Integration**: Save and retrieve iris data using a database.
  - [ ] Optimization for fast query.
- [ ] **Future Developments**: Plans to create a GUI or mobile app for dynamic and in real-time use.

## Installation

For latest release:
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
See [iris_recognition.py](iris_recognition.py).

## Contributing
Contributions are welcome! Please submit a pull request with your changes or improvements. Ensure that you follow the project's coding standards and provide relevant documentation. See the [CONTRIBUTING](CONTRIBUTING.md) file for details.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

## Contact
For questions or further information, please contact [orhun868@gmail.com].
