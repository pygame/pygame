#!/usr/bin/env python

'''Check that scale works.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

import os
import sys

import pygame
from pygame.locals import *
import Image

def show_image(surf, message):
    screen = pygame.display.set_mode(surf.get_size())
    pygame.display.set_caption(message)



if __name__ == '__main__':
    pygame.init()

    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    else:
        image_file = os.path.join(os.path.dirname(sys.argv[0]), 
                                  '../test/sample.bmp')

    image = pygame.image.load(image_file)
    w, h = image.get_size()
    dx, dy = 1, 1

    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))

    while True:
        for event in pygame.event.get():
            if event.type in (KEYDOWN, QUIT):
                sys.exit(0)

        w += dx
        h += dy
        if w >= width or w <= 0:
            dx = -dx
        if h >= height or h <= 0:
            dy = -dy
        w = max(0, w)
        h = max(0, h)

        surf = pygame.transform.scale(image, (w, h))

        screen.blit(surf, (0, 0))
        pygame.display.flip()

