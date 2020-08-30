# -*- coding: utf-8 -*-

import os
import sys
import platform
import unittest

from pygame.tests.test_utils import example_path
import pygame
from pygame.compat import as_unicode, unicode_, filesystem_encode


class MixerMusicModuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initializing the mixer is slow, so minimize the times it is called.
        pygame.mixer.init()

    @classmethod
    def tearDownClass(cls):
        pygame.mixer.quit()

    def setUp(cls):
        # This makes sure the mixer is always initialized before each test (in
        # case a test calls pygame.mixer.quit()).
        if pygame.mixer.get_init() is None:
            pygame.mixer.init()

    @unittest.skipIf(
        "Darwin" in platform.system(), "SDL2_mixer not loading mp3 on travisci"
    )
    def test_load_mp3(self):
        "|tags:music|"
        self.music_load("mp3")

    def test_load_ogg(self):
        "|tags:music|"
        self.music_load("ogg")

    def test_load_wav(self):
        "|tags:music|"
        self.music_load("wav")

    def music_load(self, format):
        data_fname = example_path("data")

        path = os.path.join(data_fname, "house_lo.%s" % format)
        if os.sep == "\\":
            path = path.replace("\\", "\\\\")
        umusfn = as_unicode(path)
        bmusfn = filesystem_encode(umusfn)

        pygame.mixer.music.load(umusfn)
        pygame.mixer.music.load(bmusfn)

    def test_load_object(self):
        """test loading music from file-like objects."""
        formats = ["ogg", "wav"]
        data_fname = example_path("data")
        for f in formats:
            path = os.path.join(data_fname, "house_lo.%s" % f)
            if os.sep == "\\":
                path = path.replace("\\", "\\\\")
            bmusfn = filesystem_encode(path)

            with open(bmusfn, "rb") as musf:
                pygame.mixer.music.load(musf)

    def test_load_unicode(self):
        """test non-ASCII unicode path"""
        import shutil

        ep = unicode_(example_path("data"))
        temp_file = os.path.join(ep, u"你好.wav")
        org_file = os.path.join(ep, u"house_lo.wav")
        try:
            with open(temp_file, "w") as f:
                pass
            os.remove(temp_file)
        except IOError:
            raise unittest.SkipTest("the path cannot be opened")
        shutil.copy(org_file, temp_file)
        try:
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.load(org_file)  # unload
        finally:
            os.remove(temp_file)

    def test_unload(self):
        import shutil
        import tempfile

        ep = unicode_(example_path("data"))
        org_file = os.path.join(ep, u"house_lo.wav")
        tmpfd, tmppath = tempfile.mkstemp(".wav")
        os.close(tmpfd)
        shutil.copy(org_file, tmppath)
        try:
            pygame.mixer.music.load(tmppath)
            pygame.mixer.music.unload()
        finally:
            os.remove(tmppath)

    @unittest.skipIf(
        "Darwin" in platform.system(), "SDL2_mixer issue with mp3 files on Travis CI"
    )
    def test_queue_mp3(self):
        """Ensures queue() accepts mp3 files.

        |tags:music|
        """
        filename = example_path(os.path.join("data", "house_lo.mp3"))
        pygame.mixer.music.queue(filename)

    def test_queue_ogg(self):
        """Ensures queue() accepts ogg files.

        |tags:music|
        """
        filename = example_path(os.path.join("data", "house_lo.ogg"))
        pygame.mixer.music.queue(filename)

    def test_queue_wav(self):
        """Ensures queue() accepts wav files.

        |tags:music|
        """
        filename = example_path(os.path.join("data", "house_lo.wav"))
        pygame.mixer.music.queue(filename)

    def test_queue__multiple_calls(self):
        """Ensures queue() can be called multiple times."""
        ogg_file = example_path(os.path.join("data", "house_lo.ogg"))
        wav_file = example_path(os.path.join("data", "house_lo.wav"))

        pygame.mixer.music.queue(ogg_file)
        pygame.mixer.music.queue(wav_file)

    def test_queue__no_file(self):
        """Ensures queue() correctly handles missing the file argument."""
        with self.assertRaises(TypeError):
            pygame.mixer.music.queue()

    def test_queue__invalid_sound_type(self):
        """Ensures queue() correctly handles invalid file types."""
        not_a_sound_file = example_path(os.path.join("data", "city.png"))

        with self.assertRaises(pygame.error):
            pygame.mixer.music.queue(not_a_sound_file)

    def test_queue__invalid_filename(self):
        """Ensures queue() correctly handles invalid filenames."""
        with self.assertRaises(pygame.error):
            pygame.mixer.music.queue("")

    def todo_test_stop(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer_music.stop:

        # Stops the music playback if it is currently playing.

        self.fail()

    def todo_test_rewind(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer_music.rewind:

        # Resets playback of the current music to the beginning.

        self.fail()

    def todo_test_get_pos(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer_music.get_pos:

        # This gets the number of milliseconds that the music has been playing
        # for. The returned time only represents how long the music has been
        # playing; it does not take into account any starting position
        # offsets.
        #

        self.fail()

    def todo_test_fadeout(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer_music.fadeout:

        # This will stop the music playback after it has been faded out over
        # the specified time (measured in milliseconds).
        #
        # Note, that this function blocks until the music has faded out.

        self.fail()

    @unittest.skipIf(os.environ.get("SDL_AUDIODRIVER")  == 'disk',
                     'disk audio driver "playback" writing to disk is slow')
    def test_play__start_time(self):

        pygame.display.init()

        # music file is 7 seconds long
        filename = example_path(os.path.join("data", "house_lo.ogg"))
        pygame.mixer.music.load(filename)
        start_time_in_seconds = 6.0  # 6 seconds

        music_finished = False
        clock = pygame.time.Clock()
        start_time_in_ms = clock.tick()
        # should play the last 1 second
        pygame.mixer.music.play(0, start=start_time_in_seconds)
        running = True
        while running:
            pygame.event.pump()

            if not (pygame.mixer.music.get_busy() or music_finished):
                music_finished = True
                time_to_finish = (clock.tick() - start_time_in_ms) // 1000
                self.assertEqual(time_to_finish, 1)
                running = False

    def todo_test_play(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer_music.play:

        # This will play the loaded music stream. If the music is already
        # playing it will be restarted.
        #
        # The loops argument controls the number of repeats a music will play.
        # play(5) will cause the music to played once, then repeated five
        # times, for a total of six. If the loops is -1 then the music will
        # repeat indefinitely.
        #
        # The starting position argument controls where in the music the song
        # starts playing. The starting position is dependent on the format of
        # music playing. MP3 and OGG use the position as time (in seconds).
        # MOD music it is the pattern order number. Passing a startpos will
        # raise a NotImplementedError if it cannot set the start position
        #

        self.fail()

    def todo_test_load(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer_music.load:

        # This will load a music file and prepare it for playback. If a music
        # stream is already playing it will be stopped. This does not start
        # the music playing.
        #
        # Music can only be loaded from filenames, not python file objects
        # like the other pygame loading functions.
        #

        self.fail()

    def todo_test_get_volume(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer_music.get_volume:

        # Returns the current volume for the mixer. The value will be between
        # 0.0 and 1.0.
        #

        self.fail()

    def todo_test_set_endevent(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer_music.set_endevent:

        # This causes Pygame to signal (by means of the event queue) when the
        # music is done playing. The argument determines the type of event
        # that will be queued.
        #
        # The event will be queued every time the music finishes, not just the
        # first time. To stop the event from being queued, call this method
        # with no argument.
        #

        self.fail()

    def todo_test_pause(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer_music.pause:

        # Temporarily stop playback of the music stream. It can be resumed
        # with the pygame.mixer.music.unpause() function.
        #

        self.fail()

    def todo_test_get_busy(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer_music.get_busy:

        # Returns True when the music stream is actively playing. When the
        # music is idle this returns False.
        #

        self.fail()

    def todo_test_get_endevent(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer_music.get_endevent:

        # Returns the event type to be sent every time the music finishes
        # playback. If there is no endevent the function returns
        # pygame.NOEVENT.
        #

        self.fail()

    def todo_test_unpause(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer_music.unpause:

        # This will resume the playback of a music stream after it has been paused.

        self.fail()

    def todo_test_set_volume(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer_music.set_volume:

        # Set the volume of the music playback. The value argument is between
        # 0.0 and 1.0. When new music is loaded the volume is reset.
        #

        self.fail()

    def todo_test_set_pos(self):

        # __doc__ (as of 2010-24-05) for pygame.mixer_music.set_pos:

        # This sets the position in the music file where playback will start. The
        # meaning of "pos", a float (or a number that can be converted to a float),
        # depends on the music format. Newer versions of SDL_mixer have better
        # positioning support than earlier. An SDLError is raised if a particular
        # format does not support positioning.
        #

        self.fail()

    def test_init(self):
        """issue #955. unload music whenever mixer.quit() is called"""
        import tempfile
        import shutil

        testfile = example_path(os.path.join("data", "house_lo.wav"))
        tempcopy = os.path.join(tempfile.gettempdir(), "tempfile.wav")

        for i in range(10):
            pygame.mixer.init()
            try:
                shutil.copy2(testfile, tempcopy)
                pygame.mixer.music.load(tempcopy)
                pygame.mixer.quit()
            finally:
                os.remove(tempcopy)


if __name__ == "__main__":
    unittest.main()
