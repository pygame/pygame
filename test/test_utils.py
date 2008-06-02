############################### INCOMPLETE TESTS ###############################

fail_incomplete_tests = 0

def test_not_implemented():
    return not fail_incomplete_tests
    
################################## TEMP FILES ##################################

import tempfile

def get_tmp_dir():
    return tempfile.mkdtemp()