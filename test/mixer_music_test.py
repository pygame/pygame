import test_utils
import test.unittest as unittest
import os, pygame
from test_utils import test_not_implemented

class MixerMusicModuleTest(unittest.TestCase):
    def test_load(self):
        # __doc__ (as of 2008-07-13) for pygame.mixer_music.load:
        
          # pygame.mixer.music.load(filename): return None
          # Load a music file for playback


        data_fname = os.path.join('examples', 'data')
        pygame.mixer.init()

        formats = ['mp3', 'ogg', 'wav']

        for f in formats:
            musfn = os.path.join(data_fname, 'house_lo.%s' % f)
    
            pygame.mixer.music.load(musfn)

            #NOTE: TODO: loading from filelikes are disabled...
            # because as of writing it only works in SDL_mixer svn.
            #pygame.mixer.music.load(open(musfn))
            #musf = open(musfn)
            #pygame.mixer.music.load(musf)
        pygame.mixer.quit()
        
    def test_queue(self):

        # __doc__ (as of 2008-07-13) for pygame.mixer_music.queue:

          # pygame.mixer.music.queue(filename): return None
          # queue a music file to follow the current

        self.assert_(test_not_implemented())

    def test_stop(self):

        # __doc__ (as of 2008-07-13) for pygame.mixer_music.stop:

          # pygame.mixer.music.stop(): return None
          # stop the music playback

        self.assert_(test_not_implemented())

    def test_rewind(self):

        # __doc__ (as of 2008-07-13) for pygame.mixer_music.rewind:

          # pygame.mixer.music.rewind(): return None
          # restart music

        self.assert_(test_not_implemented())

    def test_get_pos(self):

        # __doc__ (as of 2008-07-13) for pygame.mixer_music.get_pos:

          # pygame.mixer.music.get_pos(): return time
          # get the music play time

        self.assert_(test_not_implemented())

    def test_fadeout(self):

        # __doc__ (as of 2008-07-13) for pygame.mixer_music.fadeout:

          # pygame.mixer.music.fadeout(time): return None
          # stop music playback after fading out

        self.assert_(test_not_implemented())

    def test_play(self):

        # __doc__ (as of 2008-07-13) for pygame.mixer_music.play:

          # pygame.mixer.music.play(loops=0, start=0.0): return None
          # Start the playback of the music stream

        self.assert_(test_not_implemented())

    def test_get_volume(self):

        # __doc__ (as of 2008-07-13) for pygame.mixer_music.get_volume:

          # pygame.mixer.music.get_volume(): return value
          # get the music volume

        self.assert_(test_not_implemented())

    def test_set_endevent(self):

        # __doc__ (as of 2008-07-13) for pygame.mixer_music.set_endevent:

          # pygame.mixer.music.set_endevent(): return None
          # pygame.mixer.music.set_endevent(type): return None
          # have the music send an event when playback stops

        self.assert_(test_not_implemented())

    def test_pause(self):

        # __doc__ (as of 2008-07-13) for pygame.mixer_music.pause:

          # pygame.mixer.music.pause(): return None
          # temporarily stop music playback

        self.assert_(test_not_implemented())

    def test_get_busy(self):

        # __doc__ (as of 2008-07-13) for pygame.mixer_music.get_busy:

          # pygame.mixer.music.get_busy(): return bool
          # check if the music stream is playing

        self.assert_(test_not_implemented())

    def test_get_endevent(self):

        # __doc__ (as of 2008-07-13) for pygame.mixer_music.get_endevent:

          # pygame.mixer.music.get_endevent(): return type
          # get the event a channel sends when playback stops

        self.assert_(test_not_implemented())

    def test_unpause(self):

        # __doc__ (as of 2008-07-13) for pygame.mixer_music.unpause:

          # pygame.mixer.music.unpause(): return None
          # resume paused music

        self.assert_(test_not_implemented())

    def test_set_volume(self):

        # __doc__ (as of 2008-07-13) for pygame.mixer_music.set_volume:

          # pygame.mixer.music.set_volume(value): return None
          # set the music volume

        self.assert_(test_not_implemented())
        
if __name__ == '__main__':
    unittest.main()
