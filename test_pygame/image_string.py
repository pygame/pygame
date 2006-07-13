#!/usr/bin/env python

'''Check that pygame.image tostring and fromstring functions work correctly
with PIL's fromstring and tostring methods.  Press a key to advance each
test.
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

    screen.blit(surf, (0, 0))
    pygame.display.flip()

    while True:
        event = pygame.event.wait()
        if event.type == KEYDOWN:
            break
        elif event.type == QUIT:
            sys.exit(0)

if __name__ == '__main__':
    pygame.init()
    pygame.display.init()

    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    else:
        image_file = os.path.join(os.path.dirname(sys.argv[0]), 
                                  '../test/sample.bmp')

    flip_str = ''
    for flipped in (False, True):
        if flipped:
            flip_str = ' flipped'
        else:
            flip_str = ''
        # Load image with PIL, use pygame.image.fromstring with various formats
        pil_image = Image.open(image_file)
        for format in ('P', 'RGB', 'RGBA'):
            pil_converted = pil_image.convert(format)
            pil_string = pil_converted.tostring()
            pygame_image = pygame.image.fromstring(pil_string, 
                                                   pil_converted.size, 
                                                   format, flipped)
            if pil_converted.mode == 'P':
                # Gaa, no way to retrieve palette directly
                pal_image = pil_converted.resize((256, 1))
                pal_image.putdata(range(256))
                palette = list(pal_image.convert('RGB').getdata())
                pygame_image.set_palette(palette)
            show_image(pygame_image, 'frombuffer %s %s' % (format, flip_str))

        # Load image with pygame, use pygame.image.tostring with various formats,
        # save image with PIL and then check result
        pygame_image = pygame.image.load(image_file)
        for format in ('P', 'RGB', 'RGBA'):
            if format == 'P' and pygame_image.get_bitsize() != 8:
                continue
            pygame_string = pygame.image.tostring(pygame_image, format, flipped)
            pil_image = Image.fromstring(format, 
                                         pygame_image.get_size(), 
                                         pygame_string)
            if format == 'P':
                palette = pygame_image.get_palette()
                palette = reduce(lambda a,b: a + list(b), palette, [])
                pil_image.putpalette(palette)
            temp_file = os.tmpfile()
            pil_image.save(temp_file, 'png')
            temp_file.seek(0)
            displayed_image = pygame.image.load(temp_file, 'temp.png')
            show_image(displayed_image, 'tostring %s %s' % (format, flip_str))
