#!/usr/bin/env python

'''Pygame module for drawing shapes.

Draw several simple shapes to a Surface. These functions will work for
rendering to any format of Surface. Rendering to hardware Surfaces will
be slower than regular software Surfaces.

Most of the functions take a width argument to represent the size of
stroke around the edge of the shape. If a width of 0 is passed the
function will actually solid fill the entire shape.

All the drawing functions respect the clip area for the Surface, and
will be constrained to that area. The functions return a rectangle
representing the bounding area of changed pixels.

Most of the arguments accept a color argument that is an RGB triplet.
These can also accept an RGBA quadruplet. The alpha value will be written
directly into the Surface if it contains pixel alphas, but the draw
function will not draw transparently. The color argument can also be an
integer pixel value that is already mapped to the Surface's pixel format.

These functions must temporarily lock the Surface they are operating on.
Many sequential drawing calls can be sped up by locking and unlocking
the Surface object around the draw calls.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from copy import copy

from SDL import *

import pygame.base
import pygame.rect

def _get_color(color, surface):
    rgba = pygame.base._rgba_from_obj(color)
    if rgba:
        color = SDL_MapRGBA(surface._surf.format, 
                            rgba[0], rgba[1], rgba[2], rgba[3])
    if type(color) not in (int, long):
        raise 'invalid color argument'
    return color

def _get_rect(rect):
    rect = copy(rect)
    rect.normalize()
    return rect._r

def rect(surface, color, rect, width=0):
    '''Draw a rectangle shape.

    Draws a rectangular shape on the Surface. The given Rect is the area of
    the rectangle. The width argument is the thickness to draw the outer edge.
    If width is zero then the rectangle will be filled.
     
    Keep in mind the Surface.fill() method works just as well for drawing
    filled rectangles. In fact the Surface.fill() can be hardware accelerated
    on some platforms with both software and hardware display modes.

    :Parameters:
        `surface` : `Surface`
            Surface to draw on.
        `color` : int, int, int
            RGB fill color.
        `rect` : `Rect` or int, int, int, int
            Rectangle boundary.
        `width` : int
            Edge thickness, or zero to fill region.

    :rtype: `Rect`
    :return: Affected bounding box.
    '''    
    color = _get_color(color, surface)
    rect = _get_rect(rect)
    if width == 0:
        SDL_FillRect(surface._surf, rect, color)
    else:
        hw = width / 2
        r = SDL_Rect(rect.x, rect.y - hw, rect.w, width)
        SDL_FillRect(surface._surf, r, color)
        r.y += rect.h
        SDL_FillRect(surface._surf, r, color)
        r.x -= hw
        r.w = width
        r.y = rect.y
        r.h = rect.h
        SDL_FillRect(surface._surf, r, color)
        r.x += rect.w
        SDL_FillRect(surface._surf, r, color)
        # XXX returned clip rectangle wrong
    return pygame.rect.Rect(rect)

def polygon(surface, color, pointlist, width=0):
    '''Draw a shape with any number of sides.

    Draws a polygonal shape on the Surface. The pointlist argument is the
    vertices of the polygon. The width argument is the thickness to draw the
    outer edge. If width is zero then the polygon will be filled.

    For aapolygon, use aalines with the 'closed' parameter.
    
    :Parameters:
        `surface` : `Surface`
            Surface to draw on.
        `color` : int, int, int
            RGB fill color.
        `pointlist` : list of (int, int)
            Each list element is a vertex of the polygon.
        `width` : int
            Edge thickness, or zero to fill region.

    :rtype: `Rect`
    :return: Affected bounding box.
    '''    

def circle(surface, color, pos, radius, width=0):
    '''Draw a circle around a point.

    Draws a circular shape on the Surface. The pos argument is the center of
    the circle, and radius is the size. The width argument is the thickness to
    draw the outer edge. If width is zero then the circle will be filled.

    :Parameters:
        `surface` : `Surface`
            Surface to draw on.
        `color` : int, int, int
            RGB fill color.
        `pos` : int, int
            Center point of circle.
        `radius` : int
            Radius of circle, in pixels.
        `width` : int
            Edge thickness, or zero to fill region.

    :rtype: `Rect`
    :return: Affected bounding box.
    '''     

def ellipse(surface, color, rect, width=0):
    '''Draw a round shape inside a rectangle.

    Draws an elliptical shape on the Surface. The given rectangle is the area
    that the circle will fill. The width argument is the thickness to draw the
    outer edge. If width is zero then the ellipse will be filled.
    
    :Parameters:
        `surface` : `Surface`
            Surface to draw on.
        `color` : int, int, int
            RGB fill color.
        `rect` : `Rect` or int, int, int, int
            Ellipse boundary.
        `width` : int
            Edge thickness, or zero to fill region.

    :rtype: `Rect`
    :return: Affected bounding box.
    '''     

def arc(surface, color, rect, start_angle, stop_angle, width=1):
    '''Draw a partial section of an ellipse.

    Draws an elliptical arc on the Surface. The rect argument is the area that
    the ellipse will fill. The two angle arguments are the initial and final
    angle in radians, with the zero on the right. The width argument is the
    thickness to draw the outer edge.

    :Parameters:
        `surface` : `Surface`
            Surface to draw on.
        `color` : int, int, int
            RGB fill color.
        `rect` : `Rect` or int, int, int, int
            Ellipse boundary.
        `start_angle` : float
            Angle to start drawing arc at, in radians.
        `stop_angle` : float
            Angle to stop drawing arc at, in radians.
        `width` : int
            Edge thickness.

    :rtype: `Rect`
    :return: Affected bounding box.
    '''     

def line(surface, color, start_pos, end_pos, width=1):
    '''Draw a straight line segment.

    Draw a straight line segment on a Surface. There are no endcaps, the
    ends are squared off for thick lines.

    :Parameters:
        `surface` : `Surface`
            Surface to draw on.
        `color` : int, int, int
            RGB fill color.
        `start_pos` : int, int
            X, Y coordinates of first vertex.
        `end_pos` : int, int
            X, Y coordinates of second vertex.
        `width` : int
            Line thickness.

    :rtype: `Rect`
    :return: Affected bounding box.
    '''
    
def lines(surface, color, closed, pointlist, width=1):
    '''Draw multiple contiguous line segments.

    Draw a sequence of lines on a Surface. The pointlist argument is a series
    of points that are connected by a line. If the closed argument is true an
    additional line segment is drawn between the first and last points.

    This does not draw any endcaps or miter joints. Lines with sharp corners
    and wide line widths can have improper looking corners.
    
    :Parameters:
        `surface` : `Surface`
            Surface to draw on.
        `color` : int, int, int
            RGB fill color.
        `closed` : bool
            True if line segments form a closed polygon, otherwise False.
        `pointlist` : list of (int, int)
            List of X, Y coordinates giving vertices.
        `width` : int
            Line thickness.

    :rtype: `Rect`
    :return: Affected bounding box.
    '''

def aaline(surface, color, startpos, endpos, blend=1):
    '''Draw a line with antialiasing.

    Draws an anti-aliased line on a surface. This will respect the clipping
    rectangle. A bounding box of the affected area is returned returned as a
    rectangle. If blend is true, the shades will be be blended with existing
    pixel shades instead of overwriting them. This function accepts floating
    point values for the end points. 
    
    :Parameters:
        `surface` : `Surface`
            Surface to draw on.
        `color` : int, int, int
            RGB fill color.
        `start_pos` : float, float
            X, Y coordinates of first vertex.
        `end_pos` : float, float
            X, Y coordinates of second vertex.
        `width` : int
            Line thickness.

    :rtype: `Rect`
    :return: Affected bounding box.
    '''

def aalines(surface, color, closed, pointlist, blend=1):
    '''Draw multiple contiguous line segments with antialiasing.

    Draws a sequence on a surface. You must pass at least two points in the
    sequence of points. The closed argument is a simple boolean and if true, a
    line will be draw between the first and last points. The boolean blend
    argument set to true will blend the shades with existing shades instead of
    overwriting them. This function accepts floating point values for the end
    points. 

    :Parameters:
        `surface` : `Surface`
            Surface to draw on.
        `color` : int, int, int
            RGB fill color.
        `closed` : bool
            True if line segments form a closed polygon, otherwise False.
        `pointlist` : list of (float, float)
            List of X, Y coordinates giving vertices.
        `width` : int
            Line thickness.

    :rtype: `Rect`
    :return: Affected bounding box.
    '''
