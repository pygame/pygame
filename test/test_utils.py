#################################### IMPORTS ###################################

import tempfile, sys, pygame, time, os

################################################################################

trunk_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

def trunk_relative_path(relative):
    return os.path.normpath(os.path.join(trunk_dir, relative))

sys.path.insert(0, trunk_relative_path('.'))

import test.unittest as unittest

############################### INCOMPLETE TESTS ###############################
# TODO: PHASE THIS OUT
# Just prefix TODO test names with todo_. 
# eg def todo_test_sanity__is_overrated(self): self.fail()
# Change test loader to load test_ and todo_ TestCase callables as tests

fail_incomplete_tests = 0

def test_not_implemented():
    return not fail_incomplete_tests

################################## TEMP FILES ##################################

def get_tmp_dir():
    return tempfile.mkdtemp()

################################################################################

def question(q):
    return raw_input('%s ' % q.rstrip(' ')).lower().strip() == 'y'

def prompt(p):
    return raw_input('%s (and press enter to continue) ' % p.rstrip(' '))

#################################### HELPERS ###################################

def rgba_between(value, minimum=0, maximum=255):
    if value < minimum: return minimum
    elif value > maximum: return maximum
    else: return value

def combinations(seqs):
    """
    
    Recipe 496807 from ActiveState Python CookBook
    
    Non recursive technique for getting all possible combinations of a sequence 
    of sequences.
    
    """

    r=[[]]
    for x in seqs:
        r = [ i + [y] for y in x for i in r ]
    return r

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
        
    print 'Tests: OK'

if __name__ == '__main__':
    helpers_test()

################################################################################
