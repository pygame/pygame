'''Pygame Drawing algorithms written in Python (Work in Progress)

'''
# horizontal line
def drawhorzline(surf, color, x_from, y, x_to):
    if x_from == x_to:
        surf.set_at((x_from, y), color)
        return

    start, end = (x_from, x_to) if x_from <= x_to else (x_to, x_from)
    for x in range(x_from, x_to + 1):
        surf.set_at((x, y), color)


def drawhorzlineclip(surf, color, x_from, y, x_to):
    # check Y inside surf
    if y < surf.clip_rect.y or y >= surf.clip_rect.y + surf.clip_rect.h:
        return

    x_from = max(x_from, surf.clip_rect.x)
    x_to = min(x_to, surf.clip_rect + surf.clip_rect.w - 1)

    # check any x inside surf
    if x_to < surf.clip_rect.x or x_from >= surf.clip_rect.x + surf.clip_rect.w:
        return

    drawhorzline(surf, color, x_from, y, x_to)


def draw_polygon(surface, color, points, _width=0):
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
                print('   i=%s   p%s   x_s = %d' % (i, points[i], x_sect))
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

