# pytest -v Project/Test/test_iris_database.py --html=report.html --self-contained-html

from os.path import exists
from Project.Test.test_functions import *

def test_create_tables():
    start_db()
    assert exists(DB_PATH_EXT)

def test_database_process():
    system = start_db()
    rois_export = fake_rois()
    system.insert_iris('tag_test_1', 1, rois_export)
    rois_import = system.retrieve_iris('tag_test_1')
    assert compare_rois(rois_export, rois_import)

def test_load_to_db():
    system = start_db()
    rois_export = system.load_to_db('test_image', 0, IMAGE)
    rois_import = system.retrieve_iris('test_image')
    assert compare_rois(rois_export, rois_import)
    
def test_check_exists():
    system = start_db()
    rois_export = fake_rois()
    system.insert_iris('tag_test_1', 1, rois_export)   
    assert not system.check_exists(feature_tag='tag_test_1')


