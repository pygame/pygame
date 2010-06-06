import unittest
import pygame2
import pygame2.sdl.audio as audio
import pygame2.sdl.constants as constants

class SDLAudioTest (unittest.TestCase):
    __tags__ = [ "sdl" ]

    def test_pygame2_sdl_audio_init(self):

        # __doc__ (as of 2009-05-13) for pygame2.sdl.audio.init:

        # init () -> None
        # 
        # Initializes the audio subsystem of the SDL library.
        self.assertEqual (audio.init (), None)
        audio.quit ()

    def test_pygame2_sdl_audio_quit(self):

        # __doc__ (as of 2009-05-13) for pygame2.sdl.audio.quit:

        # quit () -> None
        # 
        # Shuts down the audio subsystem of the SDL library.
        # 
        # After calling this function, you should not invoke any class, method
        # or function related to the audio subsystem as they are likely to
        # fail or might give unpredictable results.
        self.assertEqual (audio.quit (), None)

    def test_pygame2_sdl_audio_was_init(self):

        # __doc__ (as of 2009-05-13) for pygame2.sdl.audio.was_init:

        # was_init () -> bool
        # 
        # Returns whether the audio subsystem of the SDL library is
        # initialized.
        audio.quit ()
        self.assertEqual (audio.was_init (), False)
        audio.init ()
        self.assertEqual (audio.was_init (), True)
        audio.quit ()
        self.assertEqual (audio.was_init (), False)

if __name__ == "__main__":
    unittest.main ()
