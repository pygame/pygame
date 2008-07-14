import unittest

import test_utils
from test_utils import test_not_implemented

import pygame, pygame.movie

class MovieTypeTest( unittest.TestCase ):    
    def test_Movie(self):
        # __doc__ (as of 2008-06-25) for pygame.movie.Movie:
    
          # pygame.movie.Movie(filename): return Movie
          # pygame.movie.Movie(object): return Movie
          # load an mpeg movie file
    
        self.assert_(test_not_implemented()) 

if __name__ == '__main__':
    unittest.main()