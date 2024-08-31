from os import remove
import numpy as np
from os.path import exists
import cv2, sys
sys.path.append('Project')
sys.path.append('Project/Test')
sys.path.append('Database')
from iris_database import IrisSystem

DB_PATH = 'Project/Test/test'
DB_PATH_EXT = 'Project/Test/test.db'
IMAGE = 'Project/Test/test_image.jpg'

def compare_rois(rois_export: dict, rois_import: dict):
    del rois_import['iris_metadata']
    comparison = []
    comparison.append(rois_export.keys() == rois_import.keys())
    comparison.append(rois_export['kp_len'] == rois_import['kp_len'])
    comparison.append(rois_export['desc_len'] == rois_import['desc_len'])
    comparison.append(rois_export['kp_filtered_len'] == rois_import['kp_filtered_len'])
    for pos in ['right-side','left-side','bottom','complete']:
        comparison.append(np.array_equal(rois_export[pos]['img'], rois_import[pos]['img']) and np.array_equal(rois_export[pos]['des'], rois_import[pos]['des'])     )
        comparison.append(are_keypoints_identical(rois_export[pos]['kp'], rois_import[pos]['kp']))
        comparison.append(rois_export[pos]['pupil_circle'] == rois_import[pos]['pupil_circle'])
        comparison.append(rois_export[pos]['ext_circle'] == rois_import[pos]['ext_circle'])
    if False in comparison: print(comparison)
    return(all(comparison))

def fake_rois():
    per_pos = {
        'img': np.array([23,24,12,5,61,12,5]),
        'kp': (cv2.KeyPoint(size=1, x=10, y= 20, angle=10, response=20, octave=20, class_id=1), cv2.KeyPoint(size=2, x=20, y= 30, angle=4, response=10, octave=10, class_id=2)),
        'pupil_circle' : (204, 23, 234),
        'ext_circle': (26, 83, 197),
        'des': np.array([23,24,12,5,61,12,5])
        }

    rois_export = {key: per_pos for key in ['right-side','left-side','bottom','complete']}
    rois_export['kp_len'] = 200
    rois_export['desc_len'] = 300
    rois_export['kp_filtered_len'] = 400
    return rois_export

def are_keypoints_identical(kp_tuple1, kp_tuple2):
    if len(kp_tuple1) != len(kp_tuple2):
        return False

    for kp1, kp2 in zip(kp_tuple1, kp_tuple2):
        if (kp1.pt != kp2.pt or
            kp1.size != kp2.size or
            kp1.angle != kp2.angle or
            kp1.response != kp2.response or
            kp1.octave != kp2.octave or
            kp1.class_id != kp2.class_id):
            return False

    return True    

def start_db():
    if exists(DB_PATH_EXT): remove(DB_PATH_EXT)
    system = IrisSystem(DB_PATH)    
    system.create_tables()
    return system
