'''Pygame Drawing algorithms written in Python (Work in Progress)

'''
# FIXME : the import of the builtin math module is broken, even with :
# from __future__ import relative_imports
# from math import floor, ceil, trunc

#   H E L P E R   F U N C T I O N S    #

# fractional part of x

def fpart(x):
    '''return fractional part of x'''
    return x - floor(x)

def rfpart(x):
    '''return inverse fractional part of x'''
    return 1 - (x - floor(x)) # eg, 1 - fpart(x)


#   D R A W   L I N E   F U N C T I O N S    #
def drawhorzlineclip(surf, color, x_from, y, x_to):
    '''draw clipped horizontal line.'''
    # check Y inside surf
    clip = surf.get_clip()
    if y < clip.y or y >= clip.y + clip.h:
        return

    x_from = max(x_from, clip.x)
    x_to = min(x_to, clip.x + clip.w - 1)

    # check any x inside surf
    if x_to < clip.x or x_from >= clip.x + clip.w:
        return

    drawhorzline(surf, color, x_from, y, x_to)


def drawvertlineclip(surf, color, x, y_from, y_to):
    '''draw clipped vertical line.'''
    # check X inside surf
    clip = surf.get_clip()

    if x < clip.x or x >= clip.x + clip.w:
        return

    y_from = max(y_from, clip.y)
    y_to = min(y_to, clip.y + clip.h - 1)

    # check any y inside surf
    if y_to < clip.y or y_from >= clip.y + clip.h:
        return

    drawvertline(surf, color, x, y_from, y_to)


def draw_aaline(surface, color, from_point, to_point):
    'TODO !'

#   M U L T I L I N E   F U N C T I O N S   #

def draw_polygon(surface, color, points, width):
    num_points = len(points)
    point_x = [x for x, y in points]
    point_y = [y for x, y in points]

    miny = min(point_y)
    maxy = max(point_y)

    if miny == maxy:
        minx = min(point_x)
        maxx = max(point_x)
        drawhorzlineclip(surface, color, minx, miny, maxx)
        return

    for y in range(miny, maxy + 1):
        x_intersect = []
        for i in range(num_points):
            i_prev = i - 1 if i else num_points - 1

            y1 = point_y[i_prev]
            y2 = point_y[i]

            if y1 < y2:
                x1 = point_x[i_prev]
                x2 = point_x[i]
            elif y1 > y2:
                y2 = point_y[i_prev]
                y1 = point_y[i]
                x2 = point_x[i_prev]
                x1 = point_x[i]
            else:  # special case handled below
                continue

            if ( ((y >= y1) and (y < y2))  or ((y == maxy) and (y <= y2))) :
                x_sect = (y - y1) * (x2 - x1) // (y2 - y1) + x1
                x_intersect.append(x_sect)

        x_intersect.sort()
        for i in range(0, len(x_intersect), 2):
            drawhorzlineclip(surface, color, x_intersect[i], y,
                             x_intersect[i + 1])

    # special case : horizontal border lines
    for i in range(num_points):
        i_prev = i - 1 if i else num_points - 1
        y = point_y[i]
        if miny < y == point_y[i_prev] < maxy:
            drawhorzlineclip(surface, color, point_x[i], y, point_x[i_prev])


#     L O W   L E V E L   F U N C T I O N S   #
# (too low-level to be translated into python, right?)

def set_at(surf, x, y, color):
    surf.set_at((x, y), color)


def drawhorzline(surf, color, x_from, y, x_to):
    if x_from == x_to:
        surf.set_at((x_from, y), color)
        return

    start, end = (x_from, x_to) if x_from <= x_to else (x_to, x_from)
    for x in range(x_from, x_to + 1):
        surf.set_at((x, y), color)


def drawvertline(surf, color, x, y_from, y_to):
    if y_from == y_to:
        surf.set_at((x, y_from), color)
        return

    start, end = (y_from, y_to) if y_from <= y_to else (y_to, y_from)
    for y in range(y_from, y_to + 1):
        surf.set_at((x, y_to), color)
