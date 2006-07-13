#!/usr/bin/env python

'''Test saving OpenGL surface.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

import os
import sys

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def wait():
    while True:
        event = pygame.event.wait()
        if event.type == KEYDOWN:
            break
        elif event.type == QUIT:
            sys.exit(0)

if __name__ == '__main__':
    pygame.init()
    pygame.display.init()

    width, height = 800, 600
    screen = pygame.display.set_mode((width, height), OPENGL)
    pygame.display.set_caption('OpenGL view')

    glViewport(0, 0, width, height)
    gluPerspective(60, width/float(height), 1, 100)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glTranslate(0, 0, -5)
    glLight(GL_LIGHT0, GL_POSITION, (0, 0, 0, 1))
    glTranslate(-2, 2, -3)
    glColor(1, 1, 1)
    glutSolidTeapot(1)
    glTranslate(4, 0, 0)
    glColor(1, 0, 0)
    glutSolidTeapot(1)
    glTranslate(0, -4, 0)
    glColor(0, 1, 0)
    glutSolidTeapot(1)
    glTranslate(-4, 0, 0)
    glColor(0, 0, 1)
    glutSolidTeapot(1)

    pygame.display.flip()
    wait()

    tmp_file = os.tmpfile()
    pygame.image.save(screen, tmp_file)
    tmp_file.seek(0)

    image = pygame.image.load(tmp_file, 'tmp.bmp')
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Saved view')

    screen.blit(image, (0, 0))
    pygame.display.flip()
    wait()
