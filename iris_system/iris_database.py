from sqlite3 import connect
from os import listdir
from random import choice
from pickle import dumps, loads
from gzip import open
from json import dump
import cv2
from os import listdir
from .iris_recognition import IrisRecognizer

class IrisDatabaseSystem():
    def __init__(self, db_name: str) -> None:
        """Iris Database Control System 

        Args:
            db_name (str): Create new db using self.create_tables() or use existing db.
        """
        self.db_name = db_name
        self.recognizer = IrisRecognizer()

    def load_to_db(self, image_name: str, rois_id: int, img_path: str, show: bool = False):
        """Analyze iris and import data to db.

        Args:
            image_name (str): feature_tag
            rois_id (int) 
            img_path (str)
            show (bool, optional): Show while analyzing. Defaults to False.
        """
        rois = self.recognizer.load_rois_from_image(img_path, show)
        self.insert_iris(image_name, rois_id, rois)

    def create_tables(self):
        conn = connect(f'{self.db_name}.db')
        c = conn.cursor()

        # Create iris table
        c.execute('''
        CREATE TABLE IF NOT EXISTS iris (
            feature_tag TEXT PRIMARY KEY,
            iris_id INTEGER,
            kp_len INT,
            kp_filtered_len INT,
            desc_len INT,
            kp_desc_len INT
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
            conn = connect(f'{self.db_name}.db')
            c = conn.cursor()

            # Insert into iris table
            c.execute('''
            INSERT INTO iris (iris_id, feature_tag, kp_len, kp_filtered_len, desc_len, kp_desc_len) VALUES (?, ?, ?, ?, ?, ?)
            ''', (iris_id, feature_tag, int(feature_data['kp_len']), int(feature_data['kp_filtered_len']), int(feature_data['desc_len']), int(feature_data['kp_desc_len'])))

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
            print(f'Iris {feature_tag} is inserted to {self.db_name}...')
            return True
        except: return False

    def retrieve_iris(self, feature_tag: str, get_img: bool=False) -> dict:
        """Retrieves the iris data with the desired image tag.

        Args:
            feature_tag (str)
            get_img (bool, optional): Defaults to False.

        Returns:
            dict: Iris data
        """
        conn = connect(f'{self.db_name}.db')
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
                    kp_desc_len = int(row[5])

                    iris_data['kp_len'] = kp_len
                    iris_data['kp_filtered_len'] = kp_filtered_len
                    iris_data['desc_len'] = desc_len
                    iris_data['kp_desc_len'] = kp_desc_len

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

    def check_if_not_exists(self, feature_tag: str) -> bool:
        """Check if the iris with the feature_tag exist in db. 

        Args:
            feature_tag (str): _description_

        Returns:
            bool: True if not exist
        """
        conn = connect(f'{self.db_name}.db')
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM iris WHERE feature_tag = ?", (feature_tag,))
        return cursor.fetchone() is None

    def compare_retireved_images(self, image_tag_1: str, image_tag_2: str, dratio: float = 0.8, stdev_angle: int = 10, stdev_dist: float = 0.15, show = False) -> dict:
        """Compare two irises from db.

        Args:
            image_tag_1 (str)
            image_tag_2 (str)
            Parameters:
                dratio (float, optional): Defaults to 0.8.
                stdev_angle (int, optional): Defaults to 10.
                stdev_dist (float, optional): Defaults to 0.15.
                show (bool, optional): Defaults to False.

        Returns:
            dict: Matche counts for each side
        """
        print(f"Analysing {image_tag_1} {image_tag_2}...")
        if not rois_1 and image_tag_1: rois_1 = self.retrieve_iris(image_tag_1)
        if not rois_2 and image_tag_2: rois_2 = self.retrieve_iris(image_tag_2)
        return self.recognizer.getall_matches(rois_1=rois_1, rois_2=rois_2, dratio=dratio, stdev_angle=stdev_angle, stdev_dist=stdev_dist, show=show)

    def get_random_row_with_id(self, iris_id: int) -> str:
        """Get random iris tag with iris_id.

        Args:
            iris_id (int)

        Returns:
            str: Iris tag (feature_tag)
        """
        conn = connect(f'{self.db_name}.db')
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

    def test_parameters(self, db_size: int, write_data: bool = True, test_size_diff: int = 10, test_size_same: int = 10, dratio_list: list = [0.9, 0.95, 0.8, 0.75, 0.7], stdev_angle_list: list = [10, 20, 5, 25], stdev_dist_list: list = [0.10, 0.15, 0.20, 0.30]) -> dict:
        """Tests paramteres over current db.

        Args:
            db_size (int): Uniqe ID count in db. (Assuming ids starts from 0 and increases one by one.)
            write_data (bool, optional): Writes founded data to json. Defaults to True.
            test_size_diff (int, optional): Number of random rows to analyze for false data. Defaults to 10.
            test_size_same (int, optional): Number of random rows to analyze for true data. Defaults to 10.
            Parameters:
                dratio_list (list, optional): Defaults to [0.9, 0.95, 0.8, 0.75, 0.7].
                stdev_angle_list (list, optional): Defaults to [10, 20, 5, 25].
                stdev_dist_list (list, optional): Defaults to [0.10, 0.15, 0.20, 0.30].

        Returns:
            dict
        """
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
            param_dict['parameters'][param_id] = parameter
            results_dif[param_id] = {}
            results_same[param_id] = {}
            param_dict['false_match'][param_id] = []
            param_dict['true_match'][param_id] = []        
            for test_id in range(test_size_diff):
                try:
                    new_test = {}
                    number_list = list(range(db_size))
                    first_class = choice(number_list)
                    number_list.remove(first_class)
                    second_class = choice(number_list)
                    rois_1 = self.get_random_row_with_id(first_class)
                    rois_2 = self.get_random_row_with_id(second_class)
                    matches = self.compare_retireved_images(image_tag_1=rois_1[0], image_tag_2=rois_2[0], **parameter)
                    new_test['tags'] = [rois_1[0], rois_2[0]]
                    new_test['classes'] = [first_class, second_class]
                    new_test['matches'] = matches
                    new_test['parameter'] = parameter
                    results_dif[param_id][test_id] = new_test
                    param_dict['false_match'][param_id].append(int(matches['complete']))
                except: pass

            for test_id in range(test_size_same):
                try:
                    new_test = {}
                    number_list = list(range(6))
                    first_class = choice(number_list)
                    rois_1 = self.get_random_row_with_id(first_class)
                    rois_2 = self.get_random_row_with_id(first_class)
                    while rois_1[0] == rois_2[0]:
                        rois_2 = self.get_random_row_with_id(first_class)
                    matches = self.compare_retireved_images(image_tag_1=rois_1[0], image_tag_2=rois_2[0], **parameter)
                    new_test['tags'] = [rois_1[0], rois_2[0]]
                    new_test['classes'] = [first_class, first_class]
                    new_test['matches'] = matches
                    new_test['parameter'] = parameter
                    results_same[param_id][test_id] = new_test
                    param_dict['true_match'][param_id].append(int(matches['complete']))
                except: pass

        param_dict['false_match']['details'] = results_dif
        param_dict['true_match']['details'] = results_same
        
        with open('output.json', 'w') as json_file:
            dump(param_dict, json_file, indent=4)

        return param_dict

    def load_from_thousand(self, path: str = r'IrisDB/casia-iris-thousand-500mb/CASIA-Iris-Thousand/'):
        """Load data taken from https://www.kaggle.com/datasets/sondosaabed/casia-iris-thousand to db

        Args:
            path (str): Data folder path.
        """
        for id in range(0,1000):
            id_text = str(id).strip()
            while len(id_text) < 3:
                id_text = f"0{id_text}"
            print(f'\nChecking {id_text}...\n')
            for image in listdir(path+f"{id_text}/R"):
                iris_path = path+f"{id_text}/R/{image}"
                image_name = image.replace('.jpg','')
                if self.check_if_not_exists(image_name):
                    self.load_to_db(image_name, id, iris_path)
                else: print(f'{image_name} found in db.')
