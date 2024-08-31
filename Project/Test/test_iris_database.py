# pytest -v Project/Test/test_iris_database.py --html=report.html --self-contained-html

import sys
from os.path import exists
sys.path.append('Project')
sys.path.append('Project/Test')
sys.path.append('Database')
from test_functions import *

def test_create_tables():
    start_db()

    assert exists(DB_PATH_EXT)

def test_database_process():
    system = start_db()
    rois_export = fake_rois()
    system.insert_iris('tag_test_1', 1, rois_export)
    rois_import = system.retrieve_iris('tag_test_1')
    print(rois_export.keys(), rois_import.keys())    
    assert compare_rois(rois_export, rois_import)
    
def test_check_exists():
    system = start_db()
    rois_export = fake_rois()
    system.insert_iris('tag_test_1', 1, rois_export)   
    assert not system.check_db_free(feature_tag='tag_test_1')


