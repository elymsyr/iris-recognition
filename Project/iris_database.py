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
    def __init__(self, db_path: str = None, recognizer: IrisRecognizer = None) -> None:
        """Iris Database Control System 

        Args:
            db_path (str): Create new db using self.create_tables() or use existing db.
            recognizer_parameters (dict, optional): Create recognizer with defult parameters. Defaults to None.
            log_path (str, optional): Defaults to 'log.txt'.
        """
        self.db_path = db_path
        # if not exists(f'{self.db_path}.db'): self.create_tables()
        self.recognizer = recognizer

    @counter
    def load_to_db(self, image_name: str, rois_id: int, rois: dict = None):
        """Analyze iris and import data to db.

        Args:
            image_name (str): feature_tag
            rois_id (int) 
            img_path (str)
            show (bool, optional): Show while analyzing. Defaults to False.
        Returns:
            rois (dict)
        """
        self.insert_iris(image_name, rois_id, rois)

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

        conn.commit()
        conn.close()

    def insert_iris(self, feature_tag: str, iris_id: int, feature_data: dict) -> bool:
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

    def retrieve_iris(self, feature_tag: str) -> dict:
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

    def check_db_free(self, feature_tag: str) -> bool:
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
                if self.check_db_free(image_name):
                    rois = self.recognizer.load_rois_from_image(iris_path, False)
                    self.load_to_db(image_name, id, rois)
                else: print(f'{image_name} found in db.')


class IrisSystemOptimizationTest(IrisSystem):
    @counter
    @suppress_print
    def compare_iris(self, image_tag_1: str, image_tag_2: str, dratio: float = 0.8, stdev_angle: int = 10, stdev_dist: float = 0.15, show = False) -> tuple[dict, dict, dict]:
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
        rois_1 = self.retrieve_iris(image_tag_1)
        rois_2 = self.retrieve_iris(image_tag_2)
        return rois_1, rois_2, self.recognizer.getall_matches(rois_1=rois_1, rois_2=rois_2, dratio=dratio, stdev_angle=stdev_angle, stdev_dist=stdev_dist, show=show)

    def random_iris_tag(self, iris_id: int) -> str:
        """Get random iris tag with iris_id.

        Args:
            iris_id (int)

        Returns:
            str: Iris tag (feature_tag)
        """
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

    @counter
    @capture_prints_to_file('log.txt')
    def optimization_test(self, test_size_diff: int = 10, test_size_same: int = 10, dratio_list: list = [0.9, 0.95, 0.8, 0.75], stdev_angle_list: list = [10, 20, 5, 25], stdev_dist_list: list = [0.10, 0.15, 0.20]) -> dict:
        """Tests parameters over current db or from images by randomly selection.

        Args:
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
            print(f"\nRunning for mathcing parameters\n  {parameter['dratio']=}\n  {parameter['stdev_angle']=}\n  {parameter['stdev_dist']=}")
            param_dict['parameters'][param_id] = parameter
            results_dif[param_id] = {}
            results_same[param_id] = {}
            param_dict['false_match'][param_id] = []
            param_dict['true_match'][param_id] = []
            
            test_order = 0
            number_list = self.get_unique_iris_ids()
            
            for test_size in [test_size_diff, test_size_same]:
                for test_id in range(test_size):
                    test_number += 1
                    print(f"\nTest Parameter Number {test_number} of {test_data_len}\n")
                    new_test = {}
                    id_list = [id for id in number_list]
                    random_iris = True
                    while random_iris:
                        first_class = (choice(id_list))
                        if test_order == 0:
                            id_list.remove(first_class)
                            second_class = (choice(id_list))
                        else:
                            second_class = first_class
                        first_class = str(first_class)
                        while len(first_class) < 3:
                            first_class = f"0{first_class}"
                        second_class = str(second_class)
                        while len(second_class) < 3:
                            second_class = f"0{second_class}"
                        rois_1 = self.random_iris_tag(iris_id=first_class)
                        rois_2 = self.random_iris_tag(iris_id=second_class)
                        random_iris = False
                        if test_order != 0:
                            tag_counter = 0
                            while rois_1 == rois_2:
                                rois_2 = self.random_iris_tag(iris_id=second_class)
                                if tag_counter > 5:
                                    random_iris = True
                                    break
                                tag_counter += 1
                    print(f"Analysing {first_class}/{rois_1} {second_class}/{rois_2}...")
                    try:
                        iris_1, iris_2, matches = self.compare_iris(image_tag_1=rois_1, image_tag_2=rois_2, **parameter)
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
                        print(f"{test_class} match point for {test_number} -> {sum_point}")
                    except Exception as exception:
                        exception_str = f"Exception -> {first_class}/{rois_1} or {second_class}/{rois_2} :\n          {type(exception).__name__} : {str(exception)}"
                        print(exception_str)
                test_order += 1
        param_dict['false_match']['details'] = results_dif
        param_dict['true_match']['details'] = results_same
        return param_dict

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
