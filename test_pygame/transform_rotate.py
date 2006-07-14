#!/usr/bin/env python

'''Check that rotate works.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

import math
import os
import sys

import pygame
from pygame.locals import *
import Image

if __name__ == '__main__':
    pygame.init()

    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    else:
        image_file = os.path.join(os.path.dirname(sys.argv[0]), 
                                  '../test/sample.bmp')

    image = pygame.image.load(image_file)
    image.set_colorkey(image.get_at((0, 0)))
    #image.set_colorkey((0, 0, 0))
    w, h = image.get_size()

    width = height = int(math.sqrt(w ** 2 + h ** 2))
    screen = pygame.display.set_mode((width, height))

    angle = 0
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type in (KEYDOWN, QUIT):
                sys.exit(0)

        clock.tick(60)
        angle += 1  

        surf = pygame.transform.rotate(image, angle)

        w, h = surf.get_size()
        screen.fill((255, 0, 0))
        screen.blit(surf, (width / 2 - w / 2, height / 2 - h / 2))
        pygame.display.flip()

