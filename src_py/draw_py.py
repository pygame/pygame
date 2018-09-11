'''Pygame Drawing algorithms written in Python. (Work in Progress)

Implement Pygame's Drawing Algorithms in a Python version for testing
and debugging.
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


#   L O W   L E V E L   D R A W   F U N C T I O N S   #
# (They are too low-level to be translated into python, right?)

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


LEFT_EDGE = 0x1
RIGHT_EDGE = 0x2
BOTTOM_EDGE = 0x4
TOP_EDGE = 0x8

def encode(x, y, left, top, right, bottom):
    return ((x < left) *  LEFT_EDGE +
            (x > right) * RIGHT_EDGE +
            (y < top) * TOP_EDGE +
            (y > bottom) * BOTTOM_EDGE)


INSIDE = lambda a: not a
ACCEPT = lambda a, b: not (a or b)
REJECT = lambda a, b: a and b


def clip_line(pts, left, top, right, bottom):
    assert isinstance(pts, list)
    x1, y1, x2, y2 = pts

    while True:
        code1 = encode(x1, y1, left, top, right, bottom)
        code2 = encode(x2, y2, left, top, right, bottom)

        if ACCEPT(code1, code2):
            pts[:] = x1, y1, x2, y2
            return True
        if REJECT(code1, code2):
            return False

        if INSIDE(code1):
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            code1, code2 = code2, code1
        if (x2 != x1):
            m = (y2 - y1) / float(x2 - x1)
        else:
            m = 1.0
        if code1 & LEFT_EDGE:
            y1 += int((left - x1) * m)
            x1 = left
        elif code1 & RIGHT_EDGE:
            y1 += int((right - x1) * m)
            x1 = right
        elif code1 & BOTTOM_EDGE:
            if x2 != x1:
                x1 += int((bottom - y1) / m)
            y1 = bottom
        elif code1 & TOP_EDGE:
            if x2 != x1:
                x1 += int((top - y1) / m)
            y1 = top


def clip_and_draw_line(surf, rect, color, pts):
    if not clipline(pts, rect.x, rect.y, rect.x + rect.w - 1,
                    rect.y + rect.h - 1):
        # not crossing the rectangle...
        return 0
    # pts ==  x1, y1, x2, y2 ...
    if pts[1] == pts[3]:
        drawhorzline(surf, color, pts[0], pts[1], pts[2])
    elif pts[0] == pts[2]:
        drawvertline(surf, color, pts[0], pts[1], pts[3])
    else:
        drawline(surf, color, pts[0], pts[1], pts[2], pts[3])
    return 1


# Variant of https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
# This strongly differs from craw.c implementation, because we do not
# handle BytesPerPixel, and we use "slope" and "error" variables.
def drawline(surf, color, x1, y1, x2, y2):

    if x1 == x2:
        # This case should not happen...
        raise ValueError

    slope = abs((y2 - y1) / (x2 - x1))
    error = 0.0

    if slope < 1:
        # Here, it's a rather horizontal line
        # 1. check in which octants we are & set init values
        if x2 < x1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        y = y1
        dy_sign = 1 if (y1 < y2) else -1
        # 2. step along x coordinate
        for x in range(x1, x2 + 1):
            set_at(surf, x, y, color)
            error += slope
            if error >= 0.5:
                y += dy_sign
                error -= 1
    else:
        # Case of a rather vertical line
        # 1. check in which octants we are & set init values
        if y1 > y2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        x = x1
        slope = 1 / slope
        dx_sign = 1 if (x1 < x2) else -1

        # 2. step along y coordinate
        for y in range(y1, y2 + 1):
            set_at(surf, x, y, color)
            error += slope
            if error >= 0.5:
                x += dx_sign
                error -= 1


def clip_and_draw_line_width(surf, rect, color, width, line):
    yinc = xinc = 0
    if abs(line[0] - line[2]) > abs(line[1] - line[3]):
        yinc = 1
    else:
        xinc = 1
    newpts = line[:]
    if clip_and_draw_line(surf, rect, color, newpts):
        anydrawn = 1
        frame = newpts[:]
    else:
        anydrawn = 0
        frame = [10000, 10000, -10000, -10000]

    for loop in range(1, width // 2 + 1):
        newpts[0] = line[0] + xinc * loop
        newpts[1] = line[1] + yinc * loop
        newpts[2] = line[2] + xinc * loop
        newpts[3] = line[3] + yinc * loop
        if clip_and_draw_line(surf, rect, color, newpts):
            anydrawn = 1
            frame[0] = min(newpts[0], frame[0])
            frame[1] = min(newpts[1], frame[1])
            frame[2] = max(newpts[2], frame[2])
            frame[3] = max(newpts[3], frame[3])

        if loop * 2 < width:
            newpts[0] = line[0] - xinc * loop
            newpts[1] = line[1] - yinc * loop
            newpts[2] = line[2] - xinc * loop
            newpts[3] = line[3] - yinc * loop
            if clip_and_draw_line(surf, rect, color, newpts):
                anydrawn = 1
                frame[0] = min(newpts[0], frame[0])
                frame[1] = min(newpts[1], frame[1])
                frame[2] = max(newpts[2], frame[2])
                frame[3] = max(newpts[3], frame[3])

    return anydrawn


def draw_aaline(surf, color, from_point, to_point, blend):
    '''draw anti-alisiased line between two endpoints.'''
    # TODO


#   M U L T I L I N E   F U N C T I O N S   #

def draw_lines(surf, color, closed, points, width):

    length = len(points)
    if length <= 2:
        raise TypeError
    line = [0] * 4  # store x1, y1 & x2, y2 of the lines to be drawn

    x, y = points[0]
    left = right = line[0] = x
    top = bottom = line[1] = y

    for loop in range(1, length):
        line[0] = x
        line[1] = y
        x, y = points[loop]
        line[2] = x
        line[3] = y
        if clip_and_draw_line_width(surf, surf.get_clip(), color, width, line):
            left = min(line[2], left)
            top = min(line[3], top)
            right = max(line[2], right)
            bottom = max(line[3], bottom)

    if closed:
        line[0] = x
        line[1] = y
        x, y = points[0]
        line[2] = x
        line[3] = y
        clip_and_draw_line_width(surf, surf.get_clip(), color, width, line)

    return  # TODO Rect(...)


def draw_polygon(surface, color, points, width):
    if width:
        draw_lines(surface, color, 1, points, width)
        return  # TODO Rect(...)
    num_points = len(points)
    point_x = [x for x, y in points]
    point_y = [y for x, y in points]

    miny = min(point_y)
    maxy = max(point_y)

    if miny == maxy:
        minx = min(point_x)
        maxx = max(point_x)
        drawhorzlineclip(surface, color, minx, miny, maxx)
        return  # TODO Rect(...)

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

    return  # TODO Rect(...)
