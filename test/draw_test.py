#################################### IMPORTS ###################################

import test_utils
import test.unittest as unittest
from test_utils import test_not_implemented, unordered_equality

import pygame
import pygame.draw as draw 

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
        for pt in test_utils.rect_area_pts(rect):
            color_at_pt = self.surf.get_at(pt)
            self.assert_(color_at_pt == self.color)

        #And not where it shouldn't
        for pt in test_utils.rect_outer_bounds(rect):
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
        for pt in test_utils.rect_perimeter_pts(drawn):
            color_at_pt = self.surf.get_at(pt)
            self.assert_(color_at_pt == self.color)

        #And not where it shouldn't
        for pt in test_utils.rect_outer_bounds(drawn):
            color_at_pt = self.surf.get_at(pt)
            self.assert_(color_at_pt != self.color)

    def test_line(self):

        # __doc__ (as of 2008-06-25) for pygame.draw.line:

          # pygame.draw.line(Surface, color, start_pos, end_pos, width=1): return Rect
          # draw a straight line segment

        drawn = draw.line(self.surf, self.color, (1, 0), (200, 0)) #(l, t), (l, t)
        self.assert_(drawn.right == 201,
            "end point arg should be (or at least was) inclusive"
        )

        #Should be colored where it's supposed to be
        for pt in test_utils.rect_area_pts(drawn):
            self.assert_(self.surf.get_at(pt) == self.color)

        #And not where it shouldn't
        for pt in test_utils.rect_outer_bounds(drawn):
            self.assert_(self.surf.get_at(pt) != self.color)

        
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
    unittest.main()