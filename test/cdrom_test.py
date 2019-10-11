import unittest
from pygame.tests.test_utils import question, prompt

import pygame


pygame.cdrom.init()
# The number of CD drives available for testing.
CD_DRIVE_COUNT = pygame.cdrom.get_count()
pygame.cdrom.quit()


class CDROMModuleTest(unittest.TestCase):
    def setUp(self):
        pygame.cdrom.init()

    def tearDown(self):
        pygame.cdrom.quit()

    def todo_test_CD(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD:

        # pygame.cdrom.CD(id): return CD
        # class to manage a cdrom drive
        #
        # You can create a CD object for each cdrom on the system. Use
        # pygame.cdrom.get_count() to determine how many drives actually
        # exist. The id argument is an integer of the drive, starting at zero.
        #
        # The CD object is not initialized, you can only call CD.get_id() and
        # CD.get_name() on an uninitialized drive.
        #
        # It is safe to create multiple CD objects for the same drive, they
        # will all cooperate normally.
        #

        self.fail()

    def test_get_count(self):
        """Ensure the correct number of CD drives can be detected."""
        count = pygame.cdrom.get_count()
        response = question(
            "Is the correct number of CD drives on this " "system [{}]?".format(count)
        )

        self.assertTrue(response)

    def test_get_init(self):
        """Ensure the initialization state can be retrieved."""
        self.assertTrue(pygame.cdrom.get_init())

    def test_init(self):
        """Ensure module still initialized after multiple init() calls."""
        pygame.cdrom.init()
        pygame.cdrom.init()

        self.assertTrue(pygame.cdrom.get_init())

    def test_quit(self):
        """Ensure module not initialized after quit() called."""
        pygame.cdrom.quit()

        self.assertFalse(pygame.cdrom.get_init())

    def test_quit__multiple(self):
        """Ensure module still not initialized after multiple quit() calls."""
        pygame.cdrom.quit()
        pygame.cdrom.quit()

        self.assertFalse(pygame.cdrom.get_init())


@unittest.skipIf(0 == CD_DRIVE_COUNT, "No CD drives detected")
class CDTypeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.cdrom.init()

        cls._cd_id = 0  # Only testing drive 0 for now. Expand in the future.
        cls._cd = pygame.cdrom.CD(cls._cd_id)

    @classmethod
    def tearDownClass(cls):
        pygame.cdrom.quit()

    def setUp(self):
        self._cd.init()

    def tearDown(self):
        self._cd.quit()

    def test_eject(self):
        """Ensure CD drive opens/ejects."""
        self._cd.eject()
        response = question("Did the CD eject?")

        self.assertTrue(response)

        prompt("Please close the CD drive")

    def test_get_name(self):
        """Ensure correct name for CD drive."""
        cd_name = self._cd.get_name()
        response = question(
            "Is the correct name for the CD drive [{}]?" "".format(cd_name)
        )

        self.assertTrue(response)

    def todo_test_get_all(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_all:

        # CD.get_all(): return [(audio, start, end, lenth), ...]
        # get all track information
        #
        # Return a list with information for every track on the cdrom. The
        # information consists of a tuple with four values. The audio value is
        # True if the track contains audio data. The start, end, and length
        # values are floating point numbers in seconds. Start and end
        # represent absolute times on the entire disc.
        #

        self.fail()

    def todo_test_get_busy(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_busy:

        # CD.get_busy(): return bool
        # true if the drive is playing audio
        #
        # Returns True if the drive busy playing back audio.

        self.fail()

    def todo_test_get_current(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_current:

        # CD.get_current(): return track, seconds
        # the current audio playback position
        #
        # Returns both the current track and time of that track. This method
        # works when the drive is either playing or paused.
        #
        # Note, track 0 is the first track on the CD.  Track numbers start at zero.

        self.fail()

    def test_get_empty(self):
        """Ensure correct name for CD drive."""
        prompt("Please ensure the CD drive is closed")
        is_empty = self._cd.get_empty()
        response = question("Is the CD drive empty?")

        self.assertEqual(is_empty, response)

    def test_get_id(self):
        """Ensure the drive id/index is correct."""
        cd_id = self._cd.get_id()

        self.assertEqual(self._cd_id, cd_id)

    def test_get_init(self):
        """Ensure the initialization state can be retrieved."""
        self.assertTrue(self._cd.get_init())

    def todo_test_get_numtracks(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_numtracks:

        # CD.get_numtracks(): return count
        # the number of tracks on the cdrom
        #
        # Return the number of tracks on the cdrom in the drive. This will
        # return zero of the drive is empty or has no tracks.
        #

        self.fail()

    def todo_test_get_paused(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_paused:

        # CD.get_paused(): return bool
        # true if the drive is paused
        #
        # Returns True if the drive is currently paused.

        self.fail()

    def todo_test_get_track_audio(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_track_audio:

        # CD.get_track_audio(track): return bool
        # true if the cdrom track has audio data
        #
        # Determine if a track on a cdrom contains audio data. You can also
        # call CD.num_tracks() and CD.get_all() to determine more information
        # about the cdrom.
        #
        # Note, track 0 is the first track on the CD.  Track numbers start at zero.

        self.fail()

    def todo_test_get_track_length(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_track_length:

        # CD.get_track_length(track): return seconds
        # length of a cdrom track
        #
        # Return a floating point value in seconds of the length of the cdrom track.
        # Note, track 0 is the first track on the CD.  Track numbers start at zero.

        self.fail()

    def todo_test_get_track_start(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_track_start:

        # CD.get_track_start(track): return seconds
        # start time of a cdrom track
        #
        # Return the absolute time in seconds where at start of the cdrom track.
        # Note, track 0 is the first track on the CD.  Track numbers start at zero.

        self.fail()

    def test_init(self):
        """Ensure CD drive still initialized after multiple init() calls."""
        self._cd.init()
        self._cd.init()

        self.assertTrue(self._cd.get_init())

    def todo_test_pause(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.pause:

        # CD.pause(): return None
        # temporarily stop audio playback
        #
        # Temporarily stop audio playback on the CD. The playback can be
        # resumed at the same point with the CD.resume() method. If the CD is
        # not playing this method does nothing.
        #
        # Note, track 0 is the first track on the CD.  Track numbers start at zero.

        self.fail()

    def todo_test_play(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.play:

        # CD.init(): return None
        # initialize a cdrom drive for use
        #
        # Playback audio from an audio cdrom in the drive. Besides the track
        # number argument, you can also pass a starting and ending time for
        # playback. The start and end time are in seconds, and can limit the
        # section of an audio track played.
        #
        # If you pass a start time but no end, the audio will play to the end
        # of the track. If you pass a start time and 'None' for the end time,
        # the audio will play to the end of the entire disc.
        #
        # See the CD.get_numtracks() and CD.get_track_audio() to find tracks to playback.
        # Note, track 0 is the first track on the CD.  Track numbers start at zero.

        self.fail()

    def test_quit(self):
        """Ensure CD drive not initialized after quit() called."""
        self._cd.quit()

        self.assertFalse(self._cd.get_init())

    def test_quit__multiple(self):
        """Ensure CD drive still not initialized after multiple quit() calls.
        """
        self._cd.quit()
        self._cd.quit()

        self.assertFalse(self._cd.get_init())

    def todo_test_resume(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.resume:

        # CD.resume(): return None
        # unpause audio playback
        #
        # Unpause a paused CD. If the CD is not paused or already playing,
        # this method does nothing.
        #

        self.fail()

    def todo_test_stop(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.stop:

        # CD.stop(): return None
        # stop audio playback
        #
        # Stops playback of audio from the cdrom. This will also lose the
        # current playback position. This method does nothing if the drive
        # isn't already playing audio.
        #

        self.fail()


################################################################################

if __name__ == "__main__":
    unittest.main()
