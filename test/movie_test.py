import test_utils
import test.unittest as unittest
import os, sys

from test_utils import test_not_implemented

import pygame, pygame.movie, time

from pygame.locals import *

# TODO fix bugs: checking to avoid segfaults



def within(a,b, error_range):
    return abs(a - b) < error_range

def within_seq(a,b,error_range):
    for x,y in zip(a,b):
	#print x,y
	if not within(x,y,error_range):
	    return 0
    return 1



class MovieTypeTest( unittest.TestCase ):            
    def test_render_frame__off_screen(self):
        # __doc__ (as of 2008-06-25) for pygame.movie.Movie:
    
          # pygame.movie.Movie(filename): return Movie
          # pygame.movie.Movie(object): return Movie
          # load an mpeg movie file

        # pygame accepts only MPEG program stream containers, 
        # with MPEG1 video and MPEG2 audio. I found
        # that the command 
        
        # mencoder -of mpeg -ovc lavc -oac lavc -lavcopts \
        # acodec=mp2:vcodec=mpeg1video:vbitrate=1000 -o new.mpg old.avi 
        
        # os.environ.update({"SDL_VIDEODRIVER":'windib'})
        
        movie_file = test_utils.trunk_relative_path('examples/data/blue.mpg')
        
        # Need to init display before using it.
        self.assertRaises(Exception, (pygame.movie.Movie, movie_file))

    
        pygame.display.init() # Needs to be init
        
        
        movie = pygame.movie.Movie(movie_file)
        movie_dimensions = movie.get_size()
        screen = pygame.display.set_mode(movie_dimensions)

        self.assertEqual(movie_dimensions, (320, 240))

        off_screen = pygame.Surface(movie_dimensions).convert()

        movie.set_display(off_screen)
        frame_number = movie.render_frame(5)

        #self.assertEqual(off_screen.get_at((10,10)), (16, 16, 255, 255))
        #self.assert_(off_screen.get_at((10,10)) in [(16, 16, 255, 255), (18, 13, 238, 255)])
        self.assert_(within_seq( off_screen.get_at((10,10)), (16, 16, 255, 255), 20 ))

        pygame.display.quit()

    def dont_test_render_frame__on_screen(self):

        pygame.display.init() # Needs to be init or will segfault
        
        movie_file = test_utils.trunk_relative_path('examples/data/blue.mpg')
        movie = pygame.movie.Movie(movie_file)
        movie_dimensions = movie.get_size()

        self.assertEqual(movie_dimensions, (320, 240))

        screen = pygame.display.set_mode(movie_dimensions)
        movie.set_display(screen)
        movie.render_frame(5)
        
        #self.assertEqual(screen.get_at((10,10)), (16, 16, 255, 255))
        #self.assert_(screen.get_at((10,10)) in [(16, 16, 255, 255), (18, 13, 238, 255)])
        self.assert_(within_seq( screen.get_at((10,10)), (16, 16, 255, 255), 20 ))

        pygame.display.quit()

#    def test_get_busy(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.get_busy:
#
#          # Movie.get_busy(): return bool
#          # check if the movie is currently playing
#
#        self.assert_(test_not_implemented()) 
#
#    def test_get_frame(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.get_frame:
#
#          # Movie.get_frame(): return frame_number
#          # get the current video frame
#
#        self.assert_(test_not_implemented()) 
#
#    def test_get_length(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.get_length:
#
#          # Movie.get_length(): return seconds
#          # the total length of the movie in seconds
#
#        self.assert_(test_not_implemented()) 
#
#    def test_get_size(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.get_size:
#
#          # Movie.get_size(): return (width, height)
#          # get the resolution of the video
#
#        self.assert_(test_not_implemented()) 
#
#    def test_get_time(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.get_time:
#
#          # Movie.get_time(): return seconds
#          # get the current vide playback time
#
#        self.assert_(test_not_implemented()) 
#
#    def test_has_audio(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.has_audio:
#
#          # Movie.get_audio(): return bool
#          # check if the movie file contains audio
#
#        self.assert_(test_not_implemented()) 
#
#    def test_has_video(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.has_video:
#
#          # Movie.get_video(): return bool
#          # check if the movie file contains video
#
#        self.assert_(test_not_implemented()) 
#
#    def test_pause(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.pause:
#
#          # Movie.pause(): return None
#          # temporarily stop and resume playback
#
#        self.assert_(test_not_implemented()) 
#
#    def test_play(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.play:
#
#          # Movie.play(loops=0): return None
#          # start playback of a movie
#
#        self.assert_(test_not_implemented()) 
#
#    def test_render_frame(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.render_frame:
#
#          # Movie.render_frame(frame_number): return frame_number
#          # set the current video frame
#
#        self.assert_(test_not_implemented()) 
#
#    def test_rewind(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.rewind:
#
#          # Movie.rewind(): return None
#          # restart the movie playback
#
#        self.assert_(test_not_implemented()) 
#
#    def test_set_display(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.set_display:
#
#          # Movie.set_display(Surface, rect=None): return None
#          # set the video target Surface
#
#        self.assert_(test_not_implemented()) 
#
#    def test_set_volume(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.set_volume:
#
#          # Movie.set_volume(value): return None
#          # set the audio playback volume
#
#        self.assert_(test_not_implemented()) 
#
#    def test_skip(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.skip:
#
#          # Movie.skip(seconds): return None
#          # advance the movie playback position
#
#        self.assert_(test_not_implemented()) 
#
#    def test_stop(self):
#
#        # __doc__ (as of 2008-07-18) for pygame.movie.Movie.stop:
#
#          # Movie.stop(): return None
#          # stop movie playback
#
#        self.assert_(test_not_implemented()) 

if __name__ == '__main__':
    unittest.main()
