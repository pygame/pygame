#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

import random
import sys

import pygame
from pygame.locals import *

width, height = 640, 480
max_vel = 1.0

class Point(object):
    __slots__ = ['pos', 'vel']

    def __init__(self):
        self.pos = [random.randrange(0, width), random.randrange(0, height)]
        self.vel = [(random.random() - 0.5) * max_vel, 
                    (random.random() - 0.5) * max_vel]

class Shape(object):
    def __init__(self, npoints):
        self.points = []
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
        pygame.draw.rect(surface, self.color, r, self.width)

class Line(Shape):
    def __init__(self, width):
        super(Line, self).__init__(2)
        self.width = width

    def draw(self, surface):
        pygame.draw.line(surface, self.color, 
                         self.get_points()[0],
                         self.get_points()[1], self.width)

class AntialiasLine(Shape):
    def __init__(self):
        super(AntialiasLine, self).__init__(2)

    def draw(self, surface):
        pygame.draw.aaline(surface, self.color, 
                           self.get_points()[0],
                           self.get_points()[1])

class Polygon(Shape):
    def __init__(self, width):
        super(Polygon, self).__init__(random.randint(3, 5))
        self.width = width

    def draw(self, surface):
        pygame.draw.polygon(surface, self.color, self.get_points(), self.width)

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
            pygame.draw.ellipse(surface, self.color, r, self.width)

if __name__ == '__main__':
    pygame.init()

    flags = 0
    depth = 0
    screen = pygame.display.set_mode((width, height), flags, depth)

    shapes = [Rectangle(2)]

    clock = pygame.time.Clock()
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
                    shapes.append(Ellipse(4))
                elif event.unicode == 'E':
                    shapes.append(Ellipse(0))
                elif event.unicode == 'a':
                    shapes.append(AntialiasLine())
                elif event.unicode == 'c':
                    screen.fill((0, 0, 0))
                    screen.set_clip(screen.get_clip().inflate(-50, -50))
                elif event.unicode == 'C':
                    screen.set_clip(screen.get_clip().inflate(50, 50))

        time = clock.tick()
        if not paused:
            print >> sys.stderr, 'FPS %03.2f, % 3d shapes\r' % \
                (clock.get_fps(), len(shapes)),

            screen.fill((0, 0, 0))
            for shape in shapes:
                shape.update(time)
                shape.draw(screen)
            pygame.display.flip()
