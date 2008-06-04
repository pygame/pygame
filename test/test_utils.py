#################################### IMPORTS ###################################

import tempfile, sys

############################### INCOMPLETE TESTS ###############################

fail_incomplete_tests = 0

def test_not_implemented():
    return not fail_incomplete_tests

def get_cl_fail_incomplete_opt():    
    for arg in "--incomplete", "-i":
        if  arg in sys.argv:

            # Remove the flag or it will mess up unittest cmd line arg parser
            del sys.argv[sys.argv.index(arg)]

            return True

################################## TEMP FILES ##################################

def get_tmp_dir():
    return tempfile.mkdtemp()
        
#################################### HELPERS ###################################

def unordered_equality(seq1, seq2):
    """ 
    Tests to see if the contents of one sequence is contained in the other
    and that they are of the same length.
    """
    
    if len(seq1) != len(seq2):
        return False
    
    for val in seq1:
        if val not in seq2:
            return False
        
    return True

################################################################################