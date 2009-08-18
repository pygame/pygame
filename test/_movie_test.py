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
    from pygame.tests.test_utils import test_not_implemented, unittest, trunk_relative_path
else:
    from test.test_utils import test_not_implemented, unittest


import pygame
try:
    import pygame._movie as gmovie
except:
    gmovie = None



from pygame.locals import *

import os
import sys
import time

################################### CONSTANTS ##################################
filename = "War3.avi"


class MovieTypeTest( unittest.TestCase ): 
    def test_init(self):     
        pygame.display.init()    
        pygame.mixer.quit()
        movie_file = trunk_relative_path('examples/data/blue.mpg')
        movie = gmovie.Movie(movie_file)
        self.assertEqual(movie, True)

        #screen = pygame.display.get_surface()
        #movie = pygame.gmovie.Movie(filename, screen)
        #self.assertEqual(movie, True)
        
        del movie
        
    def test_play_pause(self):
        pygame.display.init()
        pygame.mixer.quit()
        movie_file = trunk_relative_path('examples/data/blue.mpg')
        movie = gmovie.Movie(movie_file)
        
        self.assertEqual(movie.playing, False)

        movie.play(-1)

        self.assertEqual(movie.playing, True)
        self.assertEqual(movie.paused, False)

        movie.pause()

        self.assertEqual(movie.playing, False)
        self.assertEqual(movie.paused, True)
        
        movie.pause()
    
        self.assertEqual(movie.playing, True)
        self.assertEqual(movie.paused, False)
        
        del movie
        
    def test_stop(self):
        pygame.display.init()
        pygame.mixer.quit()
        movie_file = trunk_relative_path('examples/data/blue.mpg')
        movie = gmovie.Movie(movie_file)
        
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
        pygame.mixer.quit()
        movie_file = trunk_relative_path('examples/data/blue.mpg')
        movie = gmovie.Movie(movie_file)
        
        movie.play(-1)
        time.sleep(2)
        #equivalent to stop without a time-argument
        movie.rewind()
        self.assertEqual(movie.playing, False)
        self.assertEqual(movie.paused, False)
        
        del movie

    def test_width(self):
        pygame.display.init()
        pygame.mixer.quit()
        movie_file = trunk_relative_path('examples/data/blue.mpg')
        movie = gmovie.Movie(movie_file)
        print movie.width
        self.assertEqual(movie.width, 200)
        
        del movie
        
    def test_height(self):
        pygame.display.init()
        pygame.mixer.quit()
        movie_file = trunk_relative_path('examples/data/blue.mpg')
        movie = gmovie.Movie(movie_file)
        print movie.height
        self.assertEqual(movie.height, 200)
        
        del movie

        
    def test_resize(self):
        
        pygame.display.init()
        pygame.mixer.quit()
        movie_file = trunk_relative_path('examples/data/blue.mpg')
        movie = gmovie.Movie(movie_file)
        
        movie.play(-1)
        movie.resize(movie.width/2, movie.height/2)
        #equivalent to stop without a time-argument
        
        self.assertEqual(movie.height, 100)
        self.assertEqual(movie.width, 100)
        
        del movie
        
        
