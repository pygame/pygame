#################################### IMPORTS ###################################

if __name__ == '__main__':
    import sys
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

if is_pygame_pkg:
    from pygame.tests import test_utils
    from pygame.tests.test_utils \
         import test_not_implemented, unordered_equality, unittest
else:
    from test import test_utils
    from test.test_utils \
         import test_not_implemented, unordered_equality, unittest
import pygame
from pygame import draw

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

    def todo_test_aaline(self):

        # __doc__ (as of 2008-08-02) for pygame.draw.aaline:

          # pygame.draw.aaline(Surface, color, startpos, endpos, blend=1): return Rect
          # draw fine antialiased lines
          # 
          # Draws an anti-aliased line on a surface. This will respect the
          # clipping rectangle. A bounding box of the affected area is returned
          # returned as a rectangle. If blend is true, the shades will be be
          # blended with existing pixel shades instead of overwriting them. This
          # function accepts floating point values for the end points.
          # 

        self.fail() 

    def todo_test_aalines(self):

        # __doc__ (as of 2008-08-02) for pygame.draw.aalines:

          # pygame.draw.aalines(Surface, color, closed, pointlist, blend=1): return Rect
          # 
          # Draws a sequence on a surface. You must pass at least two points in
          # the sequence of points. The closed argument is a simple boolean and
          # if true, a line will be draw between the first and last points. The
          # boolean blend argument set to true will blend the shades with
          # existing shades instead of overwriting them. This function accepts
          # floating point values for the end points.
          # 

        self.fail() 

    def todo_test_arc(self):

        # __doc__ (as of 2008-08-02) for pygame.draw.arc:

          # pygame.draw.arc(Surface, color, Rect, start_angle, stop_angle,
          # width=1): return Rect
          # 
          # draw a partial section of an ellipse
          # 
          # Draws an elliptical arc on the Surface. The rect argument is the
          # area that the ellipse will fill. The two angle arguments are the
          # initial and final angle in radians, with the zero on the right. The
          # width argument is the thickness to draw the outer edge.
          # 

        self.fail() 

    def todo_test_circle(self):

        # __doc__ (as of 2008-08-02) for pygame.draw.circle:

          # pygame.draw.circle(Surface, color, pos, radius, width=0): return Rect
          # draw a circle around a point
          # 
          # Draws a circular shape on the Surface. The pos argument is the
          # center of the circle, and radius is the size. The width argument is
          # the thickness to draw the outer edge. If width is zero then the
          # circle will be filled.
          # 

        self.fail() 

    def todo_test_ellipse(self):

        # __doc__ (as of 2008-08-02) for pygame.draw.ellipse:

          # pygame.draw.ellipse(Surface, color, Rect, width=0): return Rect
          # draw a round shape inside a rectangle
          # 
          # Draws an elliptical shape on the Surface. The given rectangle is the
          # area that the circle will fill. The width argument is the thickness
          # to draw the outer edge. If width is zero then the ellipse will be
          # filled.
          # 

        self.fail() 

    def todo_test_lines(self):

        # __doc__ (as of 2008-08-02) for pygame.draw.lines:

          # pygame.draw.lines(Surface, color, closed, pointlist, width=1): return Rect
          # draw multiple contiguous line segments
          # 
          # Draw a sequence of lines on a Surface. The pointlist argument is a
          # series of points that are connected by a line. If the closed
          # argument is true an additional line segment is drawn between the
          # first and last points.
          # 
          # This does not draw any endcaps or miter joints. Lines with sharp
          # corners and wide line widths can have improper looking corners.
          # 

        self.fail() 

    def todo_test_polygon(self):

        # __doc__ (as of 2008-08-02) for pygame.draw.polygon:

          # pygame.draw.polygon(Surface, color, pointlist, width=0): return Rect
          # draw a shape with any number of sides
          # 
          # Draws a polygonal shape on the Surface. The pointlist argument is
          # the vertices of the polygon. The width argument is the thickness to
          # draw the outer edge. If width is zero then the polygon will be
          # filled.
          # 
          # For aapolygon, use aalines with the 'closed' parameter. 

        self.fail() 

################################################################################

if __name__ == '__main__':
    unittest.main()
