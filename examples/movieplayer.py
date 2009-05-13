#!/usr/bin/env python

import pygame
from pygame.locals import *

import sys
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
from pygame.compat import unicode_

QUIT_CHAR = unicode_('q')

usage = """\
python movieplayer.py <movie file>

A simple movie player that plays an MPEG movie in a Pygame window. It showcases
the pygame.movie module. The window adjusts to the size of the movie image. It
is given a boarder to demonstrate that a movie can play autonomously in a sub-
window. Also, the file is copied to a file like object to show that not just
Python files can be used as a movie source.

"""

def main(filepath):
    pygame.init()
    pygame.mixer.quit()

    f = BytesIO(open(filepath, 'rb').read())
    movie = pygame.movie.Movie(f)
    w, h = movie.get_size()
    w = int(w * 1.3 + 0.5)
    h = int(h * 1.3 + 0.5)
    wsize = (w+10, h+10)
    msize = (w, h)
    screen = pygame.display.set_mode(wsize)
    movie.set_display(screen, Rect((5, 5), msize))

    pygame.event.set_allowed((QUIT, KEYDOWN))
    pygame.time.set_timer(USEREVENT, 1000)
    movie.play()
    while movie.get_busy():
        evt = pygame.event.wait()
        if evt.type == QUIT:
            break
        if evt.type == KEYDOWN and evt.unicode == QUIT_CHAR:
            break
    if movie.get_busy():
        movie.stop()
    pygame.time.set_timer(USEREVENT, 0)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print (usage)
    else:
        main(sys.argv[1])
