#################################### IMPORTS ###################################

import tempfile, sys, pygame

############################### INCOMPLETE TESTS ###############################

fail_incomplete_tests = 0

def test_not_implemented():
    return not fail_incomplete_tests

def get_fail_incomplete_tests_option():
    global fail_incomplete_tests

    for arg in "--incomplete", "-i":
        if  arg in sys.argv:

            # Remove the flag or it will mess up unittest cmd line arg parser
            del sys.argv[sys.argv.index(arg)]
            
            fail_incomplete_tests = 1
            return

################################## TEMP FILES ##################################

def get_tmp_dir():
    return tempfile.mkdtemp()
        
#################################### HELPERS ###################################

def rgba_between(value, minimum=0, maximum=255):
    if value < minimum: return minimum
    elif value > maximum: return maximum
    else: return value

def gradient(width, height):
    """

    Yields a pt and corresponding RGBA tuple, for every (width, height) combo.
    Useful for generating gradients.
    
    Actual gradient may be changed, no tests rely on specific values.
    
    Used in transform.rotate lossless tests to generate a fixture.

    """

    for l in xrange(width):
        for t in xrange(height):
            yield (l,t), tuple(map(rgba_between, (l, t, l, l+t)))

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

def rect_area_pts(rect):
    for l in xrange(rect.left, rect.right):
        for t in xrange(rect.top, rect.bottom):
            yield l, t

def rect_perimeter_pts(rect):
    """
    
    Returns pts ((L, T) tuples) encompassing the perimeter of a rect.
    
    The order is clockwise:
        
          topleft to topright
         topright to bottomright
      bottomright to bottomleft
       bottomleft to topleft
    
    Duplicate pts are not returned

    """
    clock_wise_from_top_left = (
      ((l,       rect.top) for l in xrange(rect.left,      rect.right)      ),
      ((rect.right -1,  t) for t in xrange(rect.top   + 1, rect.bottom)     ), 
      ((l, rect.bottom -1) for l in xrange(rect.right  -2, rect.left -1, -1)), 
      ((rect.left,      t) for t in xrange(rect.bottom -2, rect.top,     -1))
    )
    
    for line in clock_wise_from_top_left:
        for pt in line: yield pt
    
def rect_outer_bounds(rect):
    """

    Returns topleft outerbound if possible and then the other pts, that are 
    "exclusive" bounds of the rect
        
   ?------O     
    |RECT|      ?|0)uterbound
    |----|     
   O      O

    """
    return (
         (rect.left is not 0 and [(rect.left-1, rect.top)] or []) +
        [ rect.topright,                                          
          rect.bottomleft,                                             
          rect.bottomright]  
    ) 

def helpers_test():
    """
    
    Lightweight test for helpers
    
    """

    r = pygame.Rect(0, 0, 10, 10)
    assert (
        rect_outer_bounds ( r ) == [(10,  0), # tr
                                    ( 0, 10), # bl
                                    (10, 10)] # br
    )
    
    assert len(list(rect_area_pts(r))) == 100 
    
    
    r = pygame.Rect(0, 0, 3, 3)
    assert list(rect_perimeter_pts(r)) == [
        (0, 0), (1, 0), (2, 0),                 # tl -> tr
        (2, 1), (2, 2),                         # tr -> br  
        (1, 2), (0, 2),                         # br -> bl
        (0, 1)                                  # bl -> tl
    ]
        
    return 'Tests: OK'
if __name__ == '__main__':
    print helpers_test()

################################################################################