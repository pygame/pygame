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
    from pygame.tests import test_utils
    from pygame.tests.test_utils import test_not_implemented, unittest
else:
    from test import test_utils
    from test.test_utils import test_not_implemented, unittest
import pygame
import pygame.movie
from pygame.locals import *

import os
import sys
import time

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
    

    def todo_test_get_busy(self):

        # __doc__ (as of 2008-08-02) for pygame.movie.Movie.get_busy:

          # Movie.get_busy(): return bool
          # check if the movie is currently playing
          # 
          # Returns true if the movie is currently being played. 

        self.fail() 

    def todo_test_get_frame(self):

        # __doc__ (as of 2008-08-02) for pygame.movie.Movie.get_frame:

          # Movie.get_frame(): return frame_number
          # get the current video frame
          # 
          # Returns the integer frame number of the current video frame. 

        self.fail() 

    def todo_test_get_length(self):

        # __doc__ (as of 2008-08-02) for pygame.movie.Movie.get_length:

          # Movie.get_length(): return seconds
          # the total length of the movie in seconds
          # 
          # Returns the length of the movie in seconds as a floating point value. 

        self.fail() 

    def todo_test_get_size(self):

        # __doc__ (as of 2008-08-02) for pygame.movie.Movie.get_size:

          # Movie.get_size(): return (width, height)
          # get the resolution of the video
          # 
          # Gets the resolution of the movie video. The movie will be stretched
          # to the size of any Surface, but this will report the natural video
          # size.

        self.fail() 

    def todo_test_get_time(self):

        # __doc__ (as of 2008-08-02) for pygame.movie.Movie.get_time:

          # Movie.get_time(): return seconds
          # get the current vide playback time
          # 
          # Return the current playback time as a floating point value in
          # seconds. This method currently seems broken and always returns 0.0.

        self.fail() 

    def todo_test_has_audio(self):

        # __doc__ (as of 2008-08-02) for pygame.movie.Movie.has_audio:

          # Movie.get_audio(): return bool
          # check if the movie file contains audio
          # 
          # True when the opened movie file contains an audio stream. 

        self.fail() 

    def todo_test_has_video(self):

        # __doc__ (as of 2008-08-02) for pygame.movie.Movie.has_video:

          # Movie.get_video(): return bool
          # check if the movie file contains video
          # 
          # True when the opened movie file contains a video stream. 

        self.fail() 

    def todo_test_pause(self):

        # __doc__ (as of 2008-08-02) for pygame.movie.Movie.pause:

          # Movie.pause(): return None
          # temporarily stop and resume playback
          # 
          # This will temporarily stop or restart movie playback. 

        self.fail() 

    def todo_test_play(self):

        # __doc__ (as of 2008-08-02) for pygame.movie.Movie.play:

          # Movie.play(loops=0): return None
          # start playback of a movie
          # 
          # Starts playback of the movie. Sound and video will begin playing if
          # they are not disabled. The optional loops argument controls how many
          # times the movie will be repeated. A loop value of -1 means the movie
          # will repeat forever.

        self.fail() 

    def todo_test_rewind(self):

        # __doc__ (as of 2008-08-02) for pygame.movie.Movie.rewind:

          # Movie.rewind(): return None
          # restart the movie playback
          # 
          # Sets the movie playback position to the start of the movie. The
          # movie will automatically begin playing even if it stopped.
          # 
          # The can raise a ValueError if the movie cannot be rewound. If the
          # rewind fails the movie object is considered invalid.

        self.fail() 

    def todo_test_set_display(self):

        # __doc__ (as of 2008-08-02) for pygame.movie.Movie.set_display:

          # Movie.set_display(Surface, rect=None): return None
          # set the video target Surface
          # 
          # Set the output target Surface for the movie video. You may also pass
          # a rectangle argument for the position, which will move and stretch
          # the video into the given area.
          # 
          # If None is passed as the target Surface, the video decoding will be disabled. 

        self.fail() 

    def todo_test_set_volume(self):

        # __doc__ (as of 2008-08-02) for pygame.movie.Movie.set_volume:

          # Movie.set_volume(value): return None
          # set the audio playback volume
          # 
          # Set the playback volume for this movie. The argument is a value
          # between 0.0 and 1.0. If the volume is set to 0 the movie audio will
          # not be decoded.

        self.fail() 

    def todo_test_skip(self):

        # __doc__ (as of 2008-08-02) for pygame.movie.Movie.skip:

          # Movie.skip(seconds): return None
          # advance the movie playback position
          # 
          # Advance the movie playback time in seconds. This can be called
          # before the movie is played to set the starting playback time. This
          # can only skip the movie forward, not backwards. The argument is a
          # floating point number.

        self.fail() 

    def todo_test_stop(self):

        # __doc__ (as of 2008-08-02) for pygame.movie.Movie.stop:

          # Movie.stop(): return None
          # stop movie playback
          # 
          # Stops the playback of a movie. The video and audio playback will be
          # stopped at their current position.

        self.fail() 
    
if __name__ == '__main__':
    unittest.main()
