#!/usr/bin/env python

import sys
import os
import time

if sys.platform == 'win32' and sys.getwindowsversion()[0] >= 5: # condi. and
    # On NT like Windows versions smpeg video needs windb.
    os.environ['SDL_VIDEODRIVER'] = 'windib'
    
import pygame
import pygame._movie
from pygame.locals import *

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
    pygame.display.init()

    #f = BytesIO(open(filepath, 'rb').read())
    #movie = pygame.movie.Movie(f)
    #w, h = movie.get_size()
    info = pygame._movie.MovieInfo(filepath)
    w, h = info.width, info.height
    msize = (w, h)


    print ("new screen...")
    screen = pygame.display.set_mode(msize)


    pygame.display.set_caption(os.path.split(info.filename)[-1])


    print ("before movie = pygame._movie.Movie(filepath, screen)")
    #surf = screen.copy().convert()
    #movie = pygame._movie.Movie(filepath, screen)
    movie = pygame._movie.Movie(filepath)
    #movie.surface = surf
    print ("after movie = pygame._movie.Movie(filepath, screen)")
    #movie.set_display(screen, Rect((5, 5), msize))

    print (dir(movie))
    print (movie.surface)
    #movie.xleft = 300
    print ("before movie.play()")
    movie.play(0)
    print ("after movie.play()")

    while movie.playing:

        events = pygame.event.get()
        for e in events:
            print (e)
            if e.type == QUIT or e.type == KEYDOWN and e.key == K_ESCAPE:
                movie.stop()


        if 1:
            if(not screen.get_locked()):
                try:
                    #pygame.display.update()
                    #pygame.display.flip()
                    pass
                except pygame.error:
                    break
        else:
            if(not surf.get_locked() and not screen.get_locked()):
                try:
                    screen.blit(surf, (0,0))
                except pygame.error:
                    pass
            
        time.sleep(0.1) # release the GIL.


    if movie.playing:
        movie.stop()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print (usage)
    else:
        main(sys.argv[1])
