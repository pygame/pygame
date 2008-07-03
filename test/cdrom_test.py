#################################### IMPORTS ###################################

import test_utils, unittest
from test_utils import test_not_implemented

################################################################################

class CdromModuleTest(unittest.TestCase):
    def test_CD(self):

        # __doc__ (as of 2008-06-25) for pygame.cdrom.CD:

          # pygame.cdrom.CD(id): return CD
          # class to manage a cdrom drive

        self.assert_(test_not_implemented()) 

    def test_get_count(self):

        # __doc__ (as of 2008-06-25) for pygame.cdrom.get_count:

          # pygame.cdrom.get_count(): return count
          # number of cd drives on the system

        self.assert_(test_not_implemented()) 

    def test_get_init(self):

        # __doc__ (as of 2008-06-25) for pygame.cdrom.get_init:

          # pygame.cdrom.get_init(): return bool
          # true if the cdrom module is initialized

        self.assert_(test_not_implemented()) 

    def test_init(self):

        # __doc__ (as of 2008-06-25) for pygame.cdrom.init:

          # pygame.cdrom.init(): return None
          # initialize the cdrom module

        self.assert_(test_not_implemented()) 

    def test_quit(self):

        # __doc__ (as of 2008-06-25) for pygame.cdrom.quit:

          # pygame.cdrom.quit(): return None
          # uninitialize the cdrom module

        self.assert_(test_not_implemented()) 

class CDTypeTest(unittest.TestCase):
    def test_eject(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.eject:

          # CD.eject(): return None
          # eject or open the cdrom drive

        self.assert_(test_not_implemented()) 

    def test_get_all(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.get_all:

          # CD.get_all(): return [(audio, start, end, lenth), ...]
          # get all track information

        self.assert_(test_not_implemented()) 

    def test_get_busy(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.get_busy:

          # CD.get_busy(): return bool
          # true if the drive is playing audio

        self.assert_(test_not_implemented()) 

    def test_get_current(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.get_current:

          # CD.get_current(): return track, seconds
          # the current audio playback position

        self.assert_(test_not_implemented()) 

    def test_get_empty(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.get_empty:

          # CD.get_empty(): return bool
          # False if a cdrom is in the drive

        self.assert_(test_not_implemented()) 

    def test_get_id(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.get_id:

          # CD.get_init(): return bool
          # true if this cd device initialized

        self.assert_(test_not_implemented()) 

    def test_get_init(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.get_init:

          # CD.get_init(): return bool
          # true if this cd device initialized

        self.assert_(test_not_implemented()) 

    def test_get_name(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.get_name:

          # CD.get_name(): return name
          # the system name of the cdrom drive

        self.assert_(test_not_implemented()) 

    def test_get_numtracks(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.get_numtracks:

          # CD.get_numtracks(): return count
          # the number of tracks on the cdrom

        self.assert_(test_not_implemented()) 

    def test_get_paused(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.get_paused:

          # CD.get_paused(): return bool
          # true if the drive is paused

        self.assert_(test_not_implemented()) 

    def test_get_track_audio(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.get_track_audio:

          # CD.get_track_audio(track): return bool
          # true if the cdrom track has audio data

        self.assert_(test_not_implemented()) 

    def test_get_track_length(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.get_track_length:

          # CD.get_track_length(track): return seconds
          # length of a cdrom track

        self.assert_(test_not_implemented()) 

    def test_get_track_start(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.get_track_start:

          # CD.get_track_start(track): return seconds
          # start time of a cdrom track

        self.assert_(test_not_implemented()) 

    def test_init(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.init:

          # CD.init(): return None
          # initialize a cdrom drive for use

        self.assert_(test_not_implemented()) 

    def test_pause(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.pause:

          # CD.pause(): return None
          # temporarily stop audio playback

        self.assert_(test_not_implemented()) 

    def test_play(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.play:

          # CD.init(): return None
          # initialize a cdrom drive for use

        self.assert_(test_not_implemented()) 

    def test_quit(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.quit:

          # CD.quit(): return None
          # uninitialize a cdrom drive for use

        self.assert_(test_not_implemented()) 

    def test_resume(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.resume:

          # CD.resume(): return None
          # unpause audio playback

        self.assert_(test_not_implemented()) 

    def test_stop(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.stop:

          # CD.stop(): return None
          # stop audio playback

        self.assert_(test_not_implemented())

################################################################################

if __name__ == '__main__':
    test_utils.get_fail_incomplete_tests_option()
    unittest.main()