#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

import math
import random
import sys

import pygame
from pygame.locals import *

width, height = 640, 480
max_vel = 1.0

random.seed(42)

class Point(object):
    __slots__ = ['pos', 'vel']

    def __init__(self):
        self.pos = [random.randrange(0, width), random.randrange(0, height)]
        self.vel = [(random.random() - 0.5) * max_vel, 
                    (random.random() - 0.5) * max_vel]

class Shape(object):
    def __init__(self, npoints):
        self.points = []
        self.area = None
        for i in range(npoints):
            self.points.append(Point())
        self.color = (random.randint(0, 255),
                      random.randint(0, 255), 
                      random.randint(0, 255))

    def get_points(self):
        return [(int(p.pos[0]), int(p.pos[1])) for p in self.points]

    def update(self, time):
        for point in self.points:
            point.pos[0] += point.vel[0] * time
            point.pos[1] += point.vel[1] * time
            if point.pos[0] >= width or point.pos[0] < 0:
                point.vel[0] *= -1
            if point.pos[1] >= height or point.pos[1] < 0:
                point.vel[1] *= -1

class Rectangle(Shape):
    def __init__(self, width):
        super(Rectangle, self).__init__(2)
        self.width = width

    def draw(self, surface):
        r = Rect(self.get_points())
        r.width -= r.left
        r.height -= r.top
        return pygame.draw.rect(surface, self.color, r, self.width)

class Line(Shape):
    def __init__(self, width):
        super(Line, self).__init__(2)
        self.width = width

    def draw(self, surface):
        return pygame.draw.line(surface, self.color, 
                                self.get_points()[0],
                                self.get_points()[1], self.width)

class AntialiasLine(Shape):
    def __init__(self):
        super(AntialiasLine, self).__init__(2)

    def draw(self, surface):
        return pygame.draw.aaline(surface, self.color, 
                                  self.get_points()[0],
                                  self.get_points()[1])

class Polygon(Shape):
    def __init__(self, width):
        super(Polygon, self).__init__(random.randint(3, 5))
        self.width = width

    def draw(self, surface):
        return pygame.draw.polygon(surface, self.color, 
                                   self.get_points(), self.width)

class Ellipse(Shape):
    def __init__(self, width):
        super(Ellipse, self).__init__(2)
        self.width = width

    def draw(self, surface):
        r = Rect(self.get_points())
        r.width -= r.left
        r.height -= r.top
        r.normalize()
        if self.width * 2 < r.width and self.width * 2 < r.height:
            return pygame.draw.ellipse(surface, self.color, r, self.width)
        return None

class Arc(Shape):
    def __init__(self, width):
        super(Arc, self).__init__(2)
        self.start_angle = random.random() * (math.pi * 2)
        self.stop_angle = random.random() * (math.pi * 2)
        self.d_start_angle = (random.random() - 0.5) * 0.01
        self.d_stop_angle = (random.random() - 0.5) * 0.01
        self.width = width

    def update(self, time):
        super(Arc, self).update(time)
        self.start_angle += time * self.d_start_angle
        self.start_angle = self.start_angle % (math.pi * 2)
        self.stop_angle += time * self.d_stop_angle
        self.stop_angle = self.stop_angle % (math.pi * 2)

    def draw(self, surface):
        r = Rect(self.get_points())
        r.width -= r.left
        r.height -= r.top
        r.normalize()
        if self.width * 2 < r.width and self.width * 2 < r.height:
            return pygame.draw.arc(surface, self.color, r, 
                                   0, math.pi/2, self.width)
        return None

if __name__ == '__main__':
    pygame.init()

    flags = 0
    depth = 0
    screen = pygame.display.set_mode((width, height), flags, depth)

    shapes = []

    clock = pygame.time.Clock()
    show_clips = False
    quit = False
    paused = False
    while not quit:
        for event in pygame.event.get():
            if event.type == QUIT:
                quit = True
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    quit = True
                elif event.key == K_SPACE:
                    paused = not paused
                elif event.unicode == 'r':
                    shapes.append(Rectangle(1))
                elif event.unicode == 'R':
                    shapes.append(Rectangle(0))
                elif event.unicode == 'l':
                    shapes.append(Line(1))
                elif event.unicode == 'L':
                    shapes.append(Line(4))
                elif event.unicode == 'p':
                    shapes.append(Polygon(1))
                elif event.unicode == 'P':
                    shapes.append(Polygon(0))
                elif event.unicode == 'e':
                    shapes.append(Ellipse(1))
                elif event.unicode == 'E':
                    shapes.append(Ellipse(0))
                elif event.unicode == 'a':
                    shapes.append(AntialiasLine())
                elif event.unicode == 'q':
                    shapes.append(Arc(1))
                elif event.unicode == 'Q':
                    shapes.append(Arc(4))
                elif event.unicode == 'c':
                    screen.fill((0, 0, 0))
                    screen.set_clip(screen.get_clip().inflate(-50, -50))
                elif event.unicode == 'C':
                    screen.set_clip(screen.get_clip().inflate(50, 50))
                elif event.unicode == 'i':
                    show_clips = not show_clips

        time = clock.tick()
        if not paused:
            print >> sys.stderr, 'FPS %03.2f, % 3d shapes\r' % \
                (clock.get_fps(), len(shapes)),
            
            screen.fill((0, 0, 0))

            update_rect = Rect(width, height, -width, -height)
            for shape in shapes:
                shape.update(time)
                if shape.area:
                    update_rect.union_ip(shape.area)
                shape.area = shape.draw(screen)
                if shape.area:
                    update_rect.union_ip(shape.area)
                if show_clips and shape.area:
                    pygame.draw.rect(screen, (255, 0, 0), shape.area, 1)

            if show_clips:
                pygame.display.flip()
            else:
                update_rect.normalize()
                pygame.display.update([update_rect])
