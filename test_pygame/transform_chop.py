#!/usr/bin/env python

'''Check that chop works.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

import math
import os
import sys

import pygame
from pygame.locals import *

if __name__ == '__main__':
    pygame.init()

    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    else:
        image_file = os.path.join(os.path.dirname(sys.argv[0]), 
                                  '../test/sample.bmp')

    image = pygame.image.load(image_file)
    w, h = image.get_size()


    dx = 1
    dy = 1
    x = 0
    y = 0
    chop_w = w / 4
    chop_h = h / 4
    screen = pygame.display.set_mode((w - chop_w, h - chop_h))
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type in (KEYDOWN, QUIT):
                sys.exit(0)

        clock.tick(60)
        x += dx
        y += dy
        if x + chop_w >= w or x < 0:
            dx = -dx
        if y + chop_h >= h or y < 0:
            dy = -dy

        surf = pygame.transform.chop(image, (x, y, chop_w, chop_h))

        screen.fill((255, 0, 0))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

