#################################### IMPORTS ###################################

import test_utils, unittest
from test_utils import test_not_implemented, unordered_equality

import pygame
import pygame.draw as draw 

pygame.init()

################################################################################

def rect_area_pts(rect):
    for l in xrange(rect.left, rect.right):
        for t in xrange(rect.top, rect.bottom):
            yield l, t

def rect_perimeter_pts(rect):
    """
    
    Returns a list of pts ((L, T) tuples) encompassing the perimeter of a rect.
    
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
    
    """
    return (
         (rect.left is not 0 and [(rect.left-1, rect.top)] or []) +
        [ rect.topright,                                          
          rect.bottomleft,                                             
          rect.bottomright]  
    ) 

def test_helpers():
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

if 1:
    test_helpers()

################################################################################

class DrawModuleTest(unittest.TestCase):
    def setUp(self):
        (self.surf_w, self.surf_h) = self.surf_size = (320, 200)
        self.surf = pygame.Surface(self.surf_size, pygame.SRCALPHA)
        self.color = (1, 13, 24, 205)
    
    def test_rect__fill(self):
        # __doc__ (as of 2008-06-25) for pygame.draw.rect:
    
          # pygame.draw.rect(Surface, color, Rect, width=0): return Rect
          # draw a rectangle shape

        rect = pygame.Rect(10, 10, 25, 20)
        drawn = draw.rect(self.surf, self.color, rect, 0)

        self.assert_(drawn == rect)

        #Should be colored where it's supposed to be
        for pt in rect_area_pts(rect):
            color_at_pt = self.surf.get_at(pt)
            self.assert_(color_at_pt == self.color)

        #And not where it shouldn't
        for pt in rect_outer_bounds(rect):
            color_at_pt = self.surf.get_at(pt)
            self.assert_(color_at_pt != self.color)
    
    def test_rect__one_pixel_lines(self):
        # __doc__ (as of 2008-06-25) for pygame.draw.rect:

          # pygame.draw.rect(Surface, color, Rect, width=0): return Rect
          # draw a rectangle shape
        rect = pygame.Rect(10, 10, 56, 20)
    
        drawn = draw.rect(self.surf, self.color, rect, 1)
        self.assert_(drawn == rect)

        #Should be colored where it's supposed to be
        for pt in rect_perimeter_pts(drawn):
            color_at_pt = self.surf.get_at(pt)
            self.assert_(color_at_pt == self.color)

        #And not where it shouldn't
        for pt in rect_outer_bounds(drawn):
            color_at_pt = self.surf.get_at(pt)
            self.assert_(color_at_pt != self.color)
        
    def test_aaline(self):
        # __doc__ (as of 2008-06-25) for pygame.draw.aaline:

          # pygame.draw.aaline(Surface, color, startpos, endpos, blend=1): return Rect
          # draw fine antialiased lines

        self.assert_(test_not_implemented()) 

    def test_aalines(self):

        # __doc__ (as of 2008-06-25) for pygame.draw.aalines:

          # pygame.draw.aalines(Surface, color, closed, pointlist, blend=1): return Rect

        self.assert_(test_not_implemented()) 

    def test_arc(self):

        # __doc__ (as of 2008-06-25) for pygame.draw.arc:

          # pygame.draw.arc(Surface, color, Rect, start_angle, stop_angle, width=1): return Rect
          # draw a partial section of an ellipse

        self.assert_(test_not_implemented()) 

    def test_circle(self):
        # __doc__ (as of 2008-06-25) for pygame.draw.circle:

          # pygame.draw.circle(Surface, color, pos, radius, width=0): return Rect
          # draw a circle around a point

        self.assert_(test_not_implemented()) 

    def test_ellipse(self):

        # __doc__ (as of 2008-06-25) for pygame.draw.ellipse:

          # pygame.draw.ellipse(Surface, color, Rect, width=0): return Rect
          # draw a round shape inside a rectangle

        self.assert_(test_not_implemented()) 

    def test_line(self):

        # __doc__ (as of 2008-06-25) for pygame.draw.line:

          # pygame.draw.line(Surface, color, start_pos, end_pos, width=1): return Rect
          # draw a straight line segment

        drawn = draw.line(self.surf, self.color, (1, 0), (200, 0)) #(l, t), (l, t)
        self.assert_(drawn.right == 201,
            "end point arg should be (or at least was) inclusive"
        )

        #Should be colored where it's supposed to be
        for pt in rect_area_pts(drawn):
            self.assert_(self.surf.get_at(pt) == self.color)

        #And not where it shouldn't
        for pt in rect_outer_bounds(drawn):
            self.assert_(self.surf.get_at(pt) != self.color)
        
    def test_lines(self):

        # __doc__ (as of 2008-06-25) for pygame.draw.lines:

          # pygame.draw.lines(Surface, color, closed, pointlist, width=1): return Rect
          # draw multiple contiguous line segments

        self.assert_(test_not_implemented())

    def test_polygon(self):
        # __doc__ (as of 2008-06-25) for pygame.draw.polygon:

          # pygame.draw.polygon(Surface, color, pointlist, width=0): return Rect
          # draw a shape with any number of sides

        self.assert_(test_not_implemented())

################################################################################

if __name__ == '__main__':
    test_utils.get_fail_incomplete_tests_option()
    unittest.main()