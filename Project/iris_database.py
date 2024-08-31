from sqlite3 import connect
from os import listdir
from random import choice
from pickle import dumps, loads
from gzip import open
import cv2
from random import shuffle
from iris_recognition import IrisRecognizer
from decorators import counter, suppress_print, capture_prints_to_file

class IrisSystem():
    def __init__(self, db_path: str = None, recognizer_parameters: dict = None, log_path: str = 'log.txt') -> None:
        """Iris Database Control System 

        Args:
            db_path (str): Create new db using self.create_tables() or use existing db.
            recognizer_parameters (dict, optional): Create recognizer with defult parameters. Defaults to None.
            log_path (str, optional): Defaults to 'log.txt'.
        """
        self.db_path = db_path
        self.log_path = log_path
        # if not exists(f'{self.db_path}.db'): self.create_tables()
        self.recognizer = IrisRecognizer(**recognizer_parameters) if recognizer_parameters else IrisRecognizer()

    @counter
    def load_to_db(self, image_name: str, rois_id: int, img_path: str = None, show: bool = False, iris: dict = None):
        """Analyze iris and import data to db.

        Args:
            image_name (str): feature_tag
            rois_id (int) 
            img_path (str)
            show (bool, optional): Show while analyzing. Defaults to False.
        Returns:
            rois (dict)
        """
        if iris: rois = iris
        else: rois = self.recognizer.load_rois_from_image(img_path, show)
        self.insert_iris(image_name, rois_id, rois)
        return rois

    def get_unique_iris_ids(self):
        # Connect to the SQLite database
        conn = connect(f'{self.db_path}.db')
        c = conn.cursor()

        # SQL query to select distinct iris_id values from the iris table
        c.execute('SELECT DISTINCT iris_id FROM iris')

        # Fetch all the results from the executed query
        unique_iris_ids = [row[0] for row in c.fetchall()]

        # Close the database connection
        conn.close()

        return unique_iris_ids

    def create_tables(self):
        conn = connect(f'{self.db_path}.db')
        c = conn.cursor()

        # Create iris table
        c.execute('''
        CREATE TABLE IF NOT EXISTS iris (
            feature_tag TEXT PRIMARY KEY,
            iris_id INTEGER,
            kp_len INT,
            kp_filtered_len INT,
            desc_len INT
        )
        ''') # add feature numbers found here

        # Create feature tables
        feature_tables = ['right_side', 'left_side', 'bottom', 'complete']
        for table_name in feature_tables:
            c.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                feature_tag TEXT PRIMARY KEY,
                iris_id INTEGER,
                img BLOB,
                kp BLOB,
                pupil_circle BLOB,
                ext_circle BLOB,            
                des BLOB,
                FOREIGN KEY (iris_id) REFERENCES iris (iris_id)
            )
            ''')
        
        for table_name in feature_tables:
            c.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name}_img (
                feature_tag TEXT PRIMARY KEY,
                iris_id INTEGER,
                img_kp_init BLOB,
                img_kp_filtered BLOB,
                FOREIGN KEY (iris_id) REFERENCES iris (iris_id)
            )
            ''')

        conn.commit()
        conn.close()

    @counter
    def insert_iris(self, feature_tag: str, iris_id: int, feature_data: dict, save_img: bool = False) -> bool:
        """Inserts iris data to database.

        Args:
            feature_tag (string)
            iris_id (int)
            feature_data (dict)
            save_img (bool, optional): Defaults to False.
        Returns:
            bool: True when no exception found.            
        """
        try:
            conn = connect(f'{self.db_path}.db')
            c = conn.cursor()

            # Insert into iris table
            c.execute('''
            INSERT INTO iris (iris_id, feature_tag, kp_len, kp_filtered_len, desc_len) VALUES (?, ?, ?, ?, ?)
            ''', (iris_id, feature_tag, int(feature_data['kp_len']), int(feature_data['kp_filtered_len']), int(feature_data['desc_len'])))

            # Insert into feature tables
            if save_img:
                feature_tables = ['right_side', 'left_side', 'bottom', 'complete']
                for table_name in feature_tables:
                    data = feature_data.get(table_name.replace('_', '-'), {})
                    table_name = f"{table_name}_img"
                    if data:
                        c.execute(f'''
                        INSERT INTO {table_name} (iris_id, feature_tag, img_kp_init, img_kp_filtered)
                        VALUES (?, ?, ?, ?)
                        ''', (
                            iris_id,
                            feature_tag,
                            dumps(data['img_kp_init']),
                            dumps(data['img_kp_filtered']),
                        ))
                    
            feature_tables = ['right_side', 'left_side', 'bottom', 'complete']
            for table_name in feature_tables:
                data = feature_data.get(table_name.replace('_', '-'), {})
                if data:
                    serialized_kp = dumps(self.serialize_keypoints(data['kp']))
                    serialized_pupil_circle = dumps(data['pupil_circle'])
                    serialized_ext_circle = dumps(data['ext_circle'])
                    c.execute(f'''
                    INSERT INTO {table_name} (iris_id, feature_tag, img, kp, pupil_circle, ext_circle, des)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        iris_id,
                        feature_tag,
                        dumps(data['img']),
                        serialized_kp,
                        serialized_pupil_circle,
                        serialized_ext_circle,                
                        dumps(data['des'])
                    ))

            conn.commit()
            conn.close()
            print(f'Iris {feature_tag} is inserted to {self.db_path}...')
            return True
        except: return False

    @counter
    def retrieve_iris(self, feature_tag: str, get_img: bool=False) -> dict:
        """Retrieves the iris data with the desired image tag.

        Args:
            feature_tag (str)
            get_img (bool, optional): Defaults to False.

        Returns:
            dict: Iris data
        """
        conn = connect(f'{self.db_path}.db')
        c = conn.cursor()

        # Initialize dictionary to store iris data
        iris_data = {}

        # Retrieve metadata from iris table
        c.execute('SELECT * FROM iris WHERE feature_tag = ?', (feature_tag,))
        iris_metadata = c.fetchone()
        if iris_metadata:
            iris_data['iris_metadata'] = iris_metadata

            # Retrieve feature data from specified tables
            feature_tables = ['right_side', 'left_side', 'bottom', 'complete']
            for table_name in feature_tables:
                dict_table_name = table_name.replace('_', '-')
                iris_data[dict_table_name] = {}

                # Retrieve keypoints and descriptors
                c.execute(f'SELECT * FROM {table_name} WHERE feature_tag = ?', (feature_tag,))
                rows = c.fetchall()
                for row in rows:
                    # Deserialize the feature data
                    img = loads(row[2])
                    kp = loads(row[3])
                    pupil_circle = loads(row[4])
                    ext_circle = loads(row[5])                
                    des = loads(row[6])
                    iris_data[dict_table_name]['img'] = img
                    iris_data[dict_table_name]['kp'] = self.deserialize_keypoints(kp)
                    iris_data[dict_table_name]['des'] = des
                    iris_data[dict_table_name]['pupil_circle'] = pupil_circle
                    iris_data[dict_table_name]['ext_circle'] = ext_circle

                # Retrieve image and related data if requested
                if get_img:
                    c.execute(f'SELECT * FROM {table_name}_img WHERE feature_tag = ?', (feature_tag,))
                    img_rows = c.fetchall()
                    for img_row in img_rows:
                        img_kp_init = loads(img_row[3])
                        img_kp_filtered = loads(img_row[4])
                        iris_data[dict_table_name]['img_kp_init'] = img_kp_init
                        iris_data[dict_table_name]['img_kp_filtered'] = img_kp_filtered

            # Retrieve additional information from the iris table
            c.execute('SELECT * FROM iris WHERE feature_tag = ?', (feature_tag,))
            iris_additional_info = c.fetchall()
            if iris_additional_info:
                for row in iris_additional_info:
                    # Deserialize additional feature data
                    kp_len = int(row[2])
                    kp_filtered_len = int(row[3])
                    desc_len = int(row[4])

                    iris_data['kp_len'] = kp_len
                    iris_data['kp_filtered_len'] = kp_filtered_len
                    iris_data['desc_len'] = desc_len

        conn.close()
        return iris_data

    def serialize_keypoints(self, keypoints) -> list[tuple]:
        """Convert list of cv2.KeyPoint objects to a serializable format."""
        return [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

    def deserialize_keypoints(self, serialized_keypoints) -> list[cv2.KeyPoint]:
        """Convert serialized keypoints back to list of cv2.KeyPoint objects."""
        return [cv2.KeyPoint(x, y, size, angle, response, octave, class_id)
                for (x, y, size, angle, response, octave, class_id) in serialized_keypoints]

    def print_rois(self, data) -> None:
        """Print rois data to see dictionary and value types.

        Args:
            data (dict): Rois data
        """
        print("Dict data:")
        for key, value in data.items():
            if type(value) == dict:
                print(f"  {key}")
                for s_key,s_value in value.items():
                    print(f"    {s_key} : {type(s_value) if type(s_value) != tuple else {type(item) for item in s_value}}")
            else: print(f"{key} : {type(value) if type(value) != tuple else {type(item) for item in value}}")

    def check_exists(self, feature_tag: str) -> bool:
        """Check if the iris with the feature_tag exist in db. 

        Args:
            feature_tag (str): Iris tag

        Returns:
            bool: True if not exist
        """
        conn = connect(f'{self.db_path}.db')
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM iris WHERE feature_tag = ?", (feature_tag,))
        return cursor.fetchone() is None

    @counter
    @suppress_print
    def compare_iris(self, image_tag_1: str, image_tag_2: str, image_id_1: str = None, image_id_2: str = None, recognizer_parameters: dict = None,from_db: bool = False, path: str = r'IrisDB/casia-iris-syn-200mb/CASIA-Iris-Syn', dratio: float = 0.8, stdev_angle: int = 10, stdev_dist: float = 0.15, show = False) -> tuple[dict, dict, dict]:
        """Compare two irises from db or file.

        Args:
            image_tag_1 (str)
            image_tag_2 (str)
            image_id_1 (str, optional): Defaults to None.
            image_id_2 (str, optional): Defaults to None.
            recognizer_parameters (dict, optional): Recreate IrisRecognizer class if iris read from file. Defaults to None.
            from_db (bool, optional): Take iris data from db. Defaults to False.
            path (str, optional): Take iris data from file. Defaults to r'IrisDB/casia-iris-syn-200mb/CASIA-Iris-Syn'.
            Parameters:
                dratio (float, optional): Defaults to 0.8.
                stdev_angle (int, optional): Defaults to 10.
                stdev_dist (float, optional): Defaults to 0.15.
            show (bool, optional): Defaults to False.

        Returns:
            tuple[dict, dict, dict]: rois for iris 1, rois for iris 2,  match counts for each side
        """
        if from_db:
            rois_1 = self.retrieve_iris(image_tag_1)
            print(rois_1.keys())
            rois_2 = self.retrieve_iris(image_tag_2)
        else:
            if recognizer_parameters: self.recognizer = IrisRecognizer(**recognizer_parameters)
            rois_1 = self.recognizer.load_rois_from_image(filepath=f"{path}/{image_id_1}/{image_tag_1}", show=False)
            self.load_to_db(image_name=image_tag_1.replace('.jpg', ''), rois_id=image_id_1, iris=rois_1)
            rois_2 = self.recognizer.load_rois_from_image(filepath=f"{path}/{image_id_2}/{image_tag_2}", show=False)
            self.load_to_db(image_name=image_tag_2.replace('.jpg', ''), rois_id=image_id_2, iris=rois_2)
        return rois_1, rois_2, self.recognizer.getall_matches(rois_1=rois_1, rois_2=rois_2, dratio=dratio, stdev_angle=stdev_angle, stdev_dist=stdev_dist, show=show)

    def random_iris_tag(self, iris_id: int, from_db: bool = False, path: str = r'IrisDB/casia-iris-syn-200mb/CASIA-Iris-Syn') -> str:
        """Get random iris tag with iris_id.

        Args:
            iris_id (int)

        Returns:
            str: Iris tag (feature_tag)
        """
        if from_db:
            conn = connect(f'{self.db_path}.db')
            cursor = conn.cursor()

            # Query to select a random row where feature_tag is 'x'
            query = """
            SELECT * FROM complete
            WHERE iris_id = ?
            ORDER BY RANDOM()
            LIMIT 1;
            """

            # Execute the query
            cursor.execute(query, (iris_id,))

            # Fetch the result
            random_row = cursor.fetchone()

            # Close the connection
            conn.close()
            return random_row[0]
        else:
            iris_id = str(iris_id)
            while len(iris_id) < 3:
                iris_id = f"0{iris_id}"
            images = listdir(f'{path}/{iris_id}')
            return choice(images)

    @counter
    def optimization_parameters(self, db_size: int, test_decorater_number: int = None, recognizer_parameters: dict = None, from_db: bool = False, path: str = r'IrisDB/casia-iris-syn-200mb/CASIA-Iris-Syn', save_name: str = None, test_size_diff: int = 10, test_size_same: int = 10, dratio_list: list = [0.9, 0.95, 0.8, 0.75], stdev_angle_list: list = [10, 20, 5, 25], stdev_dist_list: list = [0.10, 0.15, 0.20]) -> dict:
        """Tests parameters over current db or from images by randomly selection.

        Args:
            db_size (int): Uniqe ID count in db or folder system. (Assuming ids starts from 0 and increases one by one.)
            recognizer_parameters (dict, optional): _description_. Defaults to None.
            detectors (list[str], optional): _description_. Defaults to ['SIFT', 'ORB'].
            from_db (bool, optional): While True, iris data will be retrieved from current db. Defaults to False.
            path (str, optional): Folder path of images. 
                Assuming folder system: folder->id->images
                Defaults to r'IrisDB/casia-iris-syn-200mb/CASIA-Iris-Syn'.
            save_name (str, optional): Where to save the test results. Defaults to None.
            test_size_diff (int, optional): Number of random rows to analyze for false data. Defaults to 10.
            test_size_same (int, optional): Number of random rows to analyze for true data. Defaults to 10.
            Parameters:
                dratio_list (list, optional): Defaults to [0.9, 0.95, 0.8, 0.75, 0.7].
                stdev_angle_list (list, optional): Defaults to [10, 20, 5, 25].
                stdev_dist_list (list, optional): Defaults to [0.10, 0.15, 0.20, 0.30].

        Returns:
            dict: Results
        """
        test_data_len = (test_size_diff+test_size_same)*len(dratio_list)*len(stdev_angle_list)*len(stdev_dist_list)
        test_number = 0
        possible_parameters = []

        for dratio in dratio_list:
            for stdev_angle in stdev_angle_list:
                for stdev_dist in stdev_dist_list:
                    possible_parameters.append({'dratio': dratio, 'stdev_angle': stdev_angle, 'stdev_dist': stdev_dist})

        results_dif = {}
        results_same = {}
        
        param_dict = {}
        param_dict['false_match'] = {}
        param_dict['true_match'] = {}
        param_dict['parameters'] = {}

        for param_id, parameter in enumerate(possible_parameters):
            print(f"\n    Running for mathcing parameters\n      {parameter['dratio']=}\n      {parameter['stdev_angle']=}\n      {parameter['stdev_dist']=}")
            param_dict['parameters'][param_id] = parameter
            results_dif[param_id] = {}
            results_same[param_id] = {}
            param_dict['false_match'][param_id] = []
            param_dict['true_match'][param_id] = []
            
            test_order = 0
            
            for test_size in [test_size_diff, test_size_same]:
                for test_id in range(test_size):
                    test_number += 1
                    print(f"\n      Test Parameter Number {test_decorater_number}.{test_number} of {test_data_len}\n")
                    new_test = {}
                    number_list = list(range(db_size)) if not from_db else self.get_unique_iris_ids()
                    first_class = (choice(number_list))
                    if test_order == 0:
                        number_list.remove(first_class)
                        second_class = (choice(number_list))
                    else:
                        second_class = first_class
                    first_class = str(first_class)
                    while len(first_class) < 3:
                        first_class = f"0{first_class}"
                    second_class = str(second_class)
                    while len(second_class) < 3:
                        second_class = f"0{second_class}"
                    rois_1 = self.random_iris_tag(iris_id=first_class, from_db=from_db, path=path)
                    rois_2 = self.random_iris_tag(iris_id=second_class, from_db=from_db, path=path)                        
                    if test_order != 0:
                        while rois_1 == rois_2:
                            rois_2 = self.random_iris_tag(iris_id=first_class, from_db=from_db, path=path)                        
                    print(f"        Analysing {first_class}/{rois_1} {second_class}/{rois_2}...")
                    try:
                        iris_1, iris_2, matches = self.compare_iris(image_id_1=first_class, image_id_2=second_class, image_tag_1=rois_1, image_tag_2=rois_2, from_db=from_db, path=path, **parameter, recognizer_parameters=recognizer_parameters if recognizer_parameters else None)
                        new_test['tags'] = [rois_1, rois_2]
                        new_test['classes'] = [first_class, second_class]
                        new_test['keypoints'] = [{side: len(iris_1[side]['kp']) for side in ['right-side','left-side','bottom','complete']}, {side: len(iris_2[side]['kp']) for side in ['right-side','left-side','bottom','complete']}]
                        new_test['matches'] = matches
                        if test_order == 0:
                            results_dif[param_id][test_id] = new_test
                        else:
                            results_same[param_id][test_id] = new_test
                        if test_order == 0:
                            test_class = 'False'
                        else: test_class = 'True'
                        sum_point = int(sum([point for _, point in matches.items()]))
                        param_dict[f'{test_class.lower()}_match'][param_id].append(sum_point)
                        print(f"\n        {test_class} match point for {test_number} -> {sum_point}")
                    except Exception as exception:
                        exception_str = f"      Exception {path} -> {first_class}/{rois_1} or {second_class}/{rois_2} :\n          {type(exception).__name__} : {str(exception)}"
                        print(exception_str)
                test_order += 1

        param_dict['false_match']['details'] = results_dif
        param_dict['true_match']['details'] = results_same



        return param_dict

    @capture_prints_to_file('logger.txt')
    def process_and_store_iris(self, path: str):
        """Load data to db.

        Args:
            path (str): Data folder path.
        """
        ids = []
        for folder in listdir(path=path):
            ids.append(int(folder))
        shuffle(ids)
        for id in ids:
            id_text = str(id).strip()
            while len(id_text) < 3:
                id_text = f"0{id_text}"
            print(f'\nChecking {id_text}...\n')
            for image in listdir(path+f"{id_text}/"):
                iris_path = path+f"{id_text}/{image}"
                image_name = image.replace('.jpg','')
                if self.check_exists(image_name):
                    self.load_to_db(image_name, id, iris_path)
                else: print(f'{image_name} found in db.')

    @capture_prints_to_file('logger.txt')
    def optimization_test(self, from_db: bool = False, db_size: int = 999, detectors: list[str] = ['SIFT', 'ORB'], kp_size_min: list[int] = [0,2,4,8,12], kp_size_max: list[int] = [30,40,50], test_size_diff: int = 10, test_size_same: int = 10, dratio_list: list = [0.9, 0.95, 0.8, 0.75], stdev_angle_list: list = [10, 20, 5, 25], stdev_dist_list: list = [0.10, 0.15, 0.20]):
        test_data_len: int =  len(detectors)*len(kp_size_min)*len(kp_size_max)   
        results = {}
        test_number = 0
        print(f"\n\nTest started for {test_data_len} number of recognizer tests and {(test_size_diff+test_size_same)*len(dratio_list)*len(stdev_angle_list)*len(stdev_dist_list)} number of matching tests per each recognizer test.")
        for detector in detectors:
            for kp_min in kp_size_min:
                for kp_max in kp_size_max:
                    recognizer_parameters = {
                        'detector': detector,
                        'kp_size_max': kp_max,
                        'kp_size_min': kp_min
                    }
                    test_number += 1
                    print(f"\n  Test Decorator Number {test_number} of {test_data_len}")
                    print(f"  Running for decorater parameters\n  {recognizer_parameters['detector']=}\n  {recognizer_parameters['kp_size_max']=}\n  {recognizer_parameters['kp_size_min']=}")
                    results[f"{detector}_{kp_min}_{kp_max}"] = self.optimization_parameters(from_db=from_db, test_decorater_number = test_number,db_size=db_size, recognizer_parameters=recognizer_parameters, save_name=f"{detector}_{kp_min}_{kp_max}", test_size_diff=test_size_diff, test_size_same=test_size_same, dratio_list = dratio_list, stdev_angle_list = stdev_angle_list, stdev_dist_list = stdev_dist_list)
        return results

    def log(self, message: str):
        """
        Appends a message to the specified log file.

        param message (str): The string to append to the log file.
        param file_path (str): The path to the log file. Defaults to 'log.txt'.
        """
        assert type(message) == str
        try:
            with open(self.log_path, 'a') as file:
                file.write(message + '\n')
        except IOError as e:
            print(f"An error occurred while writing to the log file: {e}")

    def read_results(self, results: dict):
        def calculate_average_difference(false_points, true_points):
            total_difference = 0
            count = 0
            
            for id_ in true_points:
                if id_ in false_points:
                    false_value = false_points[id_]
                    true_value = true_points[id_]
                    difference = true_value - false_value
                    total_difference += difference
                    count += 1
                    
            return total_difference / count if count > 0 else 0
        list_avg_differences = {}
        parameter_best_differences = {}

        # Process each dataset
        for key, result in results.items():
            print(key)
            
            # Calculate average values
            false_points = {id_: sum(value)/len(value) for id_, value in result['false_match'].items() if id_ != 'details'}
            true_points = {id_: sum(value)/len(value) for id_, value in result['true_match'].items() if id_ != 'details'}
            
            # Store average points in the data dictionary
            data = {
                'False matches': false_points,
                'True matches': true_points
            }
            
            # Print the processed values
            print(f"  False matches: {[(id_, false_points[id_]) for id_ in false_points]}")
            print(f"  True matches: {[(id_, true_points[id_]) for id_ in true_points]}")
            
            # Calculate average differences for each parameter
            param_differences = {}
            for id_ in true_points:
                if id_ in false_points:
                    false_value = false_points[id_]
                    true_value = true_points[id_]
                    difference = true_value - false_value
                    param_differences[id_] = difference
            
            # Find the best parameter for this list
            best_param = max(param_differences, key=param_differences.get, default=None)
            best_param_diff = param_differences.get(best_param, 0)
            
            # Store the best parameter and its difference
            parameter_best_differences[key] = (best_param, best_param_diff)
            
            # Calculate the average difference for the list
            list_avg_differences[key] = calculate_average_difference(false_points, true_points)

        # Determine the better list based on average differences
        best_list = max(list_avg_differences, key=list_avg_differences.get)
        best_param, best_param_diff = parameter_best_differences[best_list]

        # Print results
        print(f"The better list is: {best_list}")
        print(f"The best parameter in this list is: {best_param} with an average difference of {best_param_diff:.2f}")
        return best_list, results[best_list]['parameters'][best_param], results