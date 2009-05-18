#################################### IMPORTS ###################################

from __future__ import generators

if __name__ == '__main__':
    import sys
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

if is_pygame_pkg:
    from pygame.tests.test_utils import test_not_implemented, unittest
else:
    from test.test_utils import test_not_implemented, unittest


import pygame
import pygame.gmovie
from pygame.locals import *

import os
import sys
import time

################################### CONSTANTS ##################################
filename = "dhs.avi"

"""
	(1) init(filename), init(file-like object), init(filename, surface), init(file-like object, surface)	
	(1) play(int loops) if loops<0, infinite play.
	(1) stop
	(1) pause
	(1) rewind(), (2) rewind(time_pos)
    (1) surface member, which refers to which surface to place an overlay on. 
    (1) playing member, true or false
    (1) paused member, true or false
    (1) _dealloc
    streams member list, with Stream objects
		(1) Stream
		(1) play(int loops) if loops<0, infinite play.
		(1) pause
		(1) stop
		(1) rewind(), (2) rewind(time_pos) <- rewinds to time-pos, accepts a float, does not guarantee exact positioning.
        (1) type()->audio, video, or subtitle
"""

class MovieTypeTest( unittest.TestCase ): 
    def test_init(self):     
        pygame.display.init()    
        movie_file = test_utils.trunk_relative_path('examples/data/blue.mpg')
        movie = pygame.gmovie.Video(movie_file)
        self.assertEqual(movie, True)
        
        screen = pygame.display.get_surface()
        movie = pygame.gmovie.Video(movie_file, screen)
        self.assertEqual(movie, True)
        
        del movie
        
    def test_play_pause(self):
        pygame.display.init()
        screen = pygame.display.get_surface()
        movie_file = test_utils.trunk_relative_path('examples/data/blue.mpg')
        movie = pygame.gmovie.Video(movie_file, screen)
        
        self.assertEqual(movie.playing, False)

        movie.play(-1)

        self.assertEqual(movie.playing, True)
        self.assertEqual(movie.paused, False)

        movie.pause()

        self.assertEqual(movie.playing, False)
        self.assertEqual(movie.paused, True)
        
        del movie
        
    def test_stop(self):
        pygame.display.init()
        screen = pygame.display.get_surface()
        movie_file = test_utils.trunk_relative_path('examples/data/blue.mpg')
        movie = pygame.gmovie.Video(movie_file, screen)
        
        self.assertEqual(movie.playing, False)
        movie.play(-1)
        self.assertEqual(movie.playing, True)
        self.assertEqual(movie.paused, False)
        movie.stop()
        self.assertEqual(movie.playing, False)
        self.assertEqual(movie.paused, False)
        
        del movie
        
    def test_rewind(self):
        pygame.display.init()
        screen = pygame.display.get_surface()
        movie_file = test_utils.trunk_relative_path('examples/data/blue.mpg')
        movie = pygame.gmovie.Video(movie_file, screen)
        
        movie.play(-1)
        time.sleep(2)
        #equivalent to stop without a time-argument
        movie.rewind()
        self.assertEqual(movie.playing, False)
        self.assertEqual(movie.paused, False)
        
        
    def test_surface(self):
        pygame.display.init()
        screen = pygame.display.get_surface()
        movie_file = test_utils.trunk_relative_path('examples/data/blue.mpg')
        movie = pygame.gmovie.Video(movie_file, screen)
        
        surf = pygame.Surface((600, 400))
        movie.surface = surf
        
        movie.play(-1)
        self.assertEqual(movie.playing, True)
        self.assertEqual(movie.paused, False)
        self.assertEqual(surf.get_locked(), True) #surface should be locked while being used
        
        movie.pause()
        self.assertEqual(movie.playing, False)
        self.assertEqual(movie.paused, True)
        self.assertEqual(surf.get_locked(), False) #surface is unlocked when paused/stopped.
        

        del surf
        del movie
        
    def test_streams(self):
        pygame.display.init()
        screen = pygame.display.get_surface()
        movie_file = test_utils.trunk_relative_path('examples/data/blue.mpg')
        movie = pygame.gmovie.Video(movie_file, screen)
        
        self.assertEqual(movie.streams, True)
        self.assertTrue(len(movie.streams)>0)
        
    

        
        
        
        
        