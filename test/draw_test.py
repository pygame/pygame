#################################### IMPORTS ###################################

import test_utils, unittest
from test_utils import test_not_implemented

################################################################################

class DrawModuleTest(unittest.TestCase):
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

    def test_rect(self):

        # __doc__ (as of 2008-06-25) for pygame.draw.rect:

          # pygame.draw.rect(Surface, color, Rect, width=0): return Rect
          # draw a rectangle shape

        self.assert_(test_not_implemented()) 


################################################################################

if __name__ == '__main__':
    test_utils.get_fail_incomplete_tests_option()
    unittest.main()
