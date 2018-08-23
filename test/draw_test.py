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

import unittest
if is_pygame_pkg:
    from pygame.tests import test_utils
else:
    from test import test_utils
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

        #Issue #310: Cannot draw rectangles that are 1 pixel high
        bgcolor = pygame.Color('black')
        self.surf.fill(bgcolor)
        hrect = pygame.Rect(1, 1, self.surf_w - 2, 1)
        vrect = pygame.Rect(1, 3, 1, self.surf_h - 4)
        drawn = draw.rect(self.surf, self.color, hrect, 0)
        self.assert_(drawn == hrect)
        x, y = hrect.topleft
        w, h = hrect.size
        self.assertEqual(self.surf.get_at((x - 1, y)), bgcolor)
        self.assertEqual(self.surf.get_at((x + w, y)), bgcolor)
        for i in range(x, x + w):
            self.assertEqual(self.surf.get_at((i, y)), self.color)
        drawn = draw.rect(self.surf, self.color, vrect, 0)
        self.assertEqual(drawn, vrect)
        x, y = vrect.topleft
        w, h = vrect.size
        self.assertEqual(self.surf.get_at((x, y - 1)), bgcolor)
        self.assertEqual(self.surf.get_at((x, y + h)), bgcolor)
        for i in range(y, y + h):
            self.assertEqual(self.surf.get_at((x, i)), self.color)
    
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

        #Line width greater that 1
        line_width = 2
        offset = 5
        a = (offset, offset)
        b = (self.surf_size[0] - offset, a[1])
        c = (a[0], self.surf_size[1] - offset)
        d = (b[0], c[1])
        e = (a[0] + offset, c[1])
        f = (b[0], c[0] + 5)
        lines = [(a, d), (b, c), (c, b), (d, a),
                 (a, b), (b, a), (a, c), (c, a),
                 (a, e), (e, a), (a, f), (f, a),
                 (a, a),]
        for p1, p2 in lines:
            msg = "%s - %s" % (p1, p2)
            if p1[0] <= p2[0]:
                plow = p1
                phigh = p2
            else:
                plow = p2
                phigh = p1
            self.surf.fill((0, 0, 0))
            rec = draw.line(self.surf, (255, 255, 255), p1, p2, line_width)
            xinc = yinc = 0
            if abs(p1[0] - p2[0]) > abs(p1[1] - p2[1]):
                yinc = 1
            else:
                xinc = 1
            for i in range(line_width):
                p = (p1[0] + xinc * i, p1[1] + yinc * i)
                self.assert_(self.surf.get_at(p) == (255, 255, 255), msg)
                p = (p2[0] + xinc * i, p2[1] + yinc * i)
                self.assert_(self.surf.get_at(p) == (255, 255, 255), msg)
            p = (plow[0] - 1, plow[1])
            self.assert_(self.surf.get_at(p) == (0, 0, 0), msg)
            p = (plow[0] + xinc * line_width, plow[1] + yinc * line_width)
            self.assert_(self.surf.get_at(p) == (0, 0, 0), msg)
            p = (phigh[0] + xinc * line_width, phigh[1] + yinc * line_width)
            self.assert_(self.surf.get_at(p) == (0, 0, 0), msg)
            if p1[0] < p2[0]:
                rx = p1[0]
            else:
                rx = p2[0]
            if p1[1] < p2[1]:
                ry = p1[1]
            else:
                ry = p2[1]
            w = abs(p2[0] - p1[0]) + 1 + xinc * (line_width - 1)
            h = abs(p2[1] - p1[1]) + 1 + yinc * (line_width - 1)
            msg += ", %s" % (rec,)
            self.assert_(rec == (rx, ry, w, h), msg)
        
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

    def test_ellipse(self):
        """ |tags: ignore| 
        """
        # __doc__ (as of 2008-08-02) for pygame.draw.ellipse:

          # pygame.draw.ellipse(Surface, color, Rect, width=0): return Rect
          # draw a round shape inside a rectangle
          # 
          # Draws an elliptical shape on the Surface. The given rectangle is the
          # area that the circle will fill. The width argument is the thickness
          # to draw the outer edge. If width is zero then the ellipse will be
          # filled.
          # 

        sizes = [(4, 4), (5, 4), (4, 5), (5, 5)]
        color = (1, 13, 24, 255)
    
        for width, height in sizes:
            for border_width in (0, 1):
                surface = pygame.Surface((width, height))

                draw.ellipse(
                    surface, color, (0, 0, height, width), border_width)

                # Get all positions of the surface's borders
                border_top = []
                border_left = []
                border_right = []
                border_bottom = []
                for x in range(width):
                    for y in range(height):
                        try:
                            surface.get_at((x, y - 1))
                        except IndexError:
                            border_top.append((x, y))

                        try:
                            surface.get_at((x - 1, y))
                        except IndexError:
                            border_left.append((x, y))

                        try:
                            surface.get_at((x + 1, y))
                        except IndexError:
                            border_right.append((x, y))

                        try:
                            surface.get_at((x, y + 1))
                        except IndexError:
                            border_bottom.append((x, y))

                # For each of the four borders check if it contains the color
                borders = [border_top, border_left, border_right, border_bottom]
                for border in borders:
                    colors = [surface.get_at(position) for position in border]
                    self.assertTrue(color in colors)

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
