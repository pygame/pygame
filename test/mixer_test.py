import sys
import os
import unittest
import pathlib
import platform

from pygame.tests.test_utils import example_path

import pygame
from pygame import mixer

IS_PYPY = "PyPy" == platform.python_implementation()

################################### CONSTANTS ##################################

FREQUENCIES = [11025, 22050, 44100, 48000]
SIZES = [-16, -8, 8, 16]  # fixme
# size 32 failed in test_get_init__returns_exact_values_used_for_init
CHANNELS = [1, 2]
BUFFERS = [3024]

CONFIGS = [
    {"frequency": f, "size": s, "channels": c}
    for f in FREQUENCIES
    for s in SIZES
    for c in CHANNELS
]
# Using all CONFIGS fails on a Mac; probably older SDL_mixer; we could do:
# if platform.system() == 'Darwin':
# But using all CONFIGS is very slow (> 10 sec for example)
# And probably, we don't need to be so exhaustive, hence:

CONFIG = {"frequency": 44100, "size": 32, "channels": 2, "allowedchanges": 0}


class InvalidBool:
    """To help test invalid bool values."""

    __bool__ = None


############################## MODULE LEVEL TESTS #############################


class MixerModuleTest(unittest.TestCase):
    def tearDown(self):
        mixer.quit()
        mixer.pre_init(0, 0, 0, 0)

    def test_init__keyword_args(self):
        # note: this test used to loop over all CONFIGS, but it's very slow..
        mixer.init(**CONFIG)
        mixer_conf = mixer.get_init()

        self.assertEqual(mixer_conf[0], CONFIG["frequency"])
        # Not all "sizes" are supported on all systems,  hence "abs".
        self.assertEqual(abs(mixer_conf[1]), abs(CONFIG["size"]))
        self.assertGreaterEqual(mixer_conf[2], CONFIG["channels"])

    def test_pre_init__keyword_args(self):
        # note: this test used to loop over all CONFIGS, but it's very slow..
        mixer.pre_init(**CONFIG)
        mixer.init()

        mixer_conf = mixer.get_init()

        self.assertEqual(mixer_conf[0], CONFIG["frequency"])
        # Not all "sizes" are supported on all systems,  hence "abs".
        self.assertEqual(abs(mixer_conf[1]), abs(CONFIG["size"]))
        self.assertGreaterEqual(mixer_conf[2], CONFIG["channels"])

    def test_pre_init__zero_values(self):
        # Ensure that argument values of 0 are replaced with
        # default values. No way to check buffer size though.
        mixer.pre_init(22050, -8, 1)  # Non default values
        mixer.pre_init(0, 0, 0)  # Should reset to default values
        mixer.init(allowedchanges=0)
        self.assertEqual(mixer.get_init()[0], 44100)
        self.assertEqual(mixer.get_init()[1], -16)
        self.assertGreaterEqual(mixer.get_init()[2], 2)

    def test_init__zero_values(self):
        # Ensure that argument values of 0 are replaced with
        # preset values. No way to check buffer size though.
        mixer.pre_init(44100, 8, 1, allowedchanges=0)  # None default values
        mixer.init(0, 0, 0)
        self.assertEqual(mixer.get_init(), (44100, 8, 1))

    def test_get_init__returns_exact_values_used_for_init(self):
        # TODO: size 32 fails in this test (maybe SDL_mixer bug)

        for init_conf in CONFIGS:
            frequency, size, channels = init_conf.values()
            if (frequency, size) == (22050, 16):
                continue
            mixer.init(frequency, size, channels)

            mixer_conf = mixer.get_init()

            self.assertEqual(tuple(init_conf.values()), mixer_conf)
            mixer.quit()

    def test_get_init__returns_None_if_mixer_not_initialized(self):
        self.assertIsNone(mixer.get_init())

    def test_get_num_channels__defaults_eight_after_init(self):
        mixer.init()
        self.assertEqual(mixer.get_num_channels(), 8)

    def test_set_num_channels(self):
        mixer.init()

        default_num_channels = mixer.get_num_channels()
        for i in range(1, default_num_channels + 1):
            mixer.set_num_channels(i)
            self.assertEqual(mixer.get_num_channels(), i)

    def test_quit(self):
        """get_num_channels() Should throw pygame.error if uninitialized
        after mixer.quit()"""
        mixer.init()
        mixer.quit()
        self.assertRaises(pygame.error, mixer.get_num_channels)

    # TODO: FIXME: appveyor and pypy (on linux) fails here sometimes.
    @unittest.skipIf(sys.platform.startswith("win"), "See github issue 892.")
    @unittest.skipIf(IS_PYPY, "random errors here with pypy")
    def test_sound_args(self):
        def get_bytes(snd):
            return snd.get_raw()

        mixer.init()

        sample = b"\x00\xff" * 24
        wave_path = example_path(os.path.join("data", "house_lo.wav"))
        uwave_path = str(wave_path)
        bwave_path = uwave_path.encode(sys.getfilesystemencoding())
        snd = mixer.Sound(file=wave_path)
        self.assertTrue(snd.get_length() > 0.5)
        snd_bytes = get_bytes(snd)
        self.assertTrue(len(snd_bytes) > 1000)

        self.assertEqual(get_bytes(mixer.Sound(wave_path)), snd_bytes)

        self.assertEqual(get_bytes(mixer.Sound(file=uwave_path)), snd_bytes)
        self.assertEqual(get_bytes(mixer.Sound(uwave_path)), snd_bytes)
        arg_emsg = "Sound takes either 1 positional or 1 keyword argument"

        with self.assertRaises(TypeError) as cm:
            mixer.Sound()
        self.assertEqual(str(cm.exception), arg_emsg)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(wave_path, buffer=sample)
        self.assertEqual(str(cm.exception), arg_emsg)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(sample, file=wave_path)
        self.assertEqual(str(cm.exception), arg_emsg)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(buffer=sample, file=wave_path)
        self.assertEqual(str(cm.exception), arg_emsg)

        with self.assertRaises(TypeError) as cm:
            mixer.Sound(foobar=sample)
        self.assertEqual(str(cm.exception), "Unrecognized keyword argument 'foobar'")

        snd = mixer.Sound(wave_path, **{})
        self.assertEqual(get_bytes(snd), snd_bytes)
        snd = mixer.Sound(*[], **{"file": wave_path})

        with self.assertRaises(TypeError) as cm:
            mixer.Sound([])
        self.assertEqual(str(cm.exception), "Unrecognized argument (type list)")

        with self.assertRaises(TypeError) as cm:
            snd = mixer.Sound(buffer=[])
        emsg = "Expected object with buffer interface: got a list"
        self.assertEqual(str(cm.exception), emsg)

        ufake_path = "12345678"
        self.assertRaises(IOError, mixer.Sound, ufake_path)
        self.assertRaises(IOError, mixer.Sound, "12345678")

        with self.assertRaises(TypeError) as cm:
            mixer.Sound(buffer="something")
        emsg = "Unicode object not allowed as buffer object"
        self.assertEqual(str(cm.exception), emsg)
        self.assertEqual(get_bytes(mixer.Sound(buffer=sample)), sample)
        if type(sample) != str:
            somebytes = get_bytes(mixer.Sound(sample))
            # on python 2 we do not allow using string except as file name.
            self.assertEqual(somebytes, sample)
        self.assertEqual(get_bytes(mixer.Sound(file=bwave_path)), snd_bytes)
        self.assertEqual(get_bytes(mixer.Sound(bwave_path)), snd_bytes)

        snd = mixer.Sound(wave_path)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(wave_path, array=snd)
        self.assertEqual(str(cm.exception), arg_emsg)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(buffer=sample, array=snd)
        self.assertEqual(str(cm.exception), arg_emsg)
        snd2 = mixer.Sound(array=snd)
        self.assertEqual(snd.get_raw(), snd2.get_raw())

    def test_sound_unicode(self):
        """test non-ASCII unicode path"""
        mixer.init()
        import shutil

        ep = example_path("data")
        temp_file = os.path.join(ep, "你好.wav")
        org_file = os.path.join(ep, "house_lo.wav")
        shutil.copy(org_file, temp_file)
        try:
            with open(temp_file, "rb") as f:
                pass
        except OSError:
            raise unittest.SkipTest("the path cannot be opened")

        try:
            sound = mixer.Sound(temp_file)
            del sound
        finally:
            os.remove(temp_file)

    @unittest.skipIf(
        os.environ.get("SDL_AUDIODRIVER") == "disk",
        "this test fails without real sound card",
    )
    def test_array_keyword(self):
        try:
            from numpy import (
                array,
                arange,
                zeros,
                int8,
                uint8,
                int16,
                uint16,
                int32,
                uint32,
            )
        except ImportError:
            self.skipTest("requires numpy")

        freq = 22050
        format_list = [-8, 8, -16, 16]
        channels_list = [1, 2]

        a_lists = {f: [] for f in format_list}
        a32u_mono = arange(0, 256, 1, uint32)
        a16u_mono = a32u_mono.astype(uint16)
        a8u_mono = a32u_mono.astype(uint8)
        au_list_mono = [(1, a) for a in [a8u_mono, a16u_mono, a32u_mono]]
        for format in format_list:
            if format > 0:
                a_lists[format].extend(au_list_mono)
        a32s_mono = arange(-128, 128, 1, int32)
        a16s_mono = a32s_mono.astype(int16)
        a8s_mono = a32s_mono.astype(int8)
        as_list_mono = [(1, a) for a in [a8s_mono, a16s_mono, a32s_mono]]
        for format in format_list:
            if format < 0:
                a_lists[format].extend(as_list_mono)
        a32u_stereo = zeros([a32u_mono.shape[0], 2], uint32)
        a32u_stereo[:, 0] = a32u_mono
        a32u_stereo[:, 1] = 255 - a32u_mono
        a16u_stereo = a32u_stereo.astype(uint16)
        a8u_stereo = a32u_stereo.astype(uint8)
        au_list_stereo = [(2, a) for a in [a8u_stereo, a16u_stereo, a32u_stereo]]
        for format in format_list:
            if format > 0:
                a_lists[format].extend(au_list_stereo)
        a32s_stereo = zeros([a32s_mono.shape[0], 2], int32)
        a32s_stereo[:, 0] = a32s_mono
        a32s_stereo[:, 1] = -1 - a32s_mono
        a16s_stereo = a32s_stereo.astype(int16)
        a8s_stereo = a32s_stereo.astype(int8)
        as_list_stereo = [(2, a) for a in [a8s_stereo, a16s_stereo, a32s_stereo]]
        for format in format_list:
            if format < 0:
                a_lists[format].extend(as_list_stereo)

        for format in format_list:
            for channels in channels_list:
                try:
                    mixer.init(freq, format, channels)
                except pygame.error:
                    # Some formats (e.g. 16) may not be supported.
                    continue
                try:
                    __, f, c = mixer.get_init()
                    if f != format or c != channels:
                        # Some formats (e.g. -8) may not be supported.
                        continue
                    for c, a in a_lists[format]:
                        self._test_array_argument(format, a, c == channels)
                finally:
                    mixer.quit()

    def _test_array_argument(self, format, a, test_pass):
        from numpy import array, all as all_

        try:
            snd = mixer.Sound(array=a)
        except ValueError:
            if not test_pass:
                return
            self.fail("Raised ValueError: Format %i, dtype %s" % (format, a.dtype))
        if not test_pass:
            self.fail(
                "Did not raise ValueError: Format %i, dtype %s" % (format, a.dtype)
            )
        a2 = array(snd)
        a3 = a.astype(a2.dtype)
        lshift = abs(format) - 8 * a.itemsize
        if lshift >= 0:
            # This is asymmetric with respect to downcasting.
            a3 <<= lshift
        self.assertTrue(all_(a2 == a3), "Format %i, dtype %s" % (format, a.dtype))

    def _test_array_interface_fail(self, a):
        self.assertRaises(ValueError, mixer.Sound, array=a)

    def test_array_interface(self):
        mixer.init(22050, -16, 1, allowedchanges=0)
        snd = mixer.Sound(buffer=b"\x00\x7f" * 20)
        d = snd.__array_interface__
        self.assertTrue(isinstance(d, dict))
        if pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN:
            typestr = "<i2"
        else:
            typestr = ">i2"
        self.assertEqual(d["typestr"], typestr)
        self.assertEqual(d["shape"], (20,))
        self.assertEqual(d["strides"], (2,))
        self.assertEqual(d["data"], (snd._samples_address, False))

    @unittest.skipIf(not pygame.HAVE_NEWBUF, "newbuf not implemented")
    @unittest.skipIf(IS_PYPY, "pypy no likey")
    def test_newbuf__one_channel(self):
        mixer.init(22050, -16, 1)
        self._NEWBUF_export_check()

    @unittest.skipIf(not pygame.HAVE_NEWBUF, "newbuf not implemented")
    @unittest.skipIf(IS_PYPY, "pypy no likey")
    def test_newbuf__twho_channel(self):
        mixer.init(22050, -16, 2)
        self._NEWBUF_export_check()

    def _NEWBUF_export_check(self):
        freq, fmt, channels = mixer.get_init()
        ndim = 1 if (channels == 1) else 2
        itemsize = abs(fmt) // 8
        formats = {
            8: "B",
            -8: "b",
            16: "=H",
            -16: "=h",
            32: "=I",
            -32: "=i",  # 32 and 64 for future consideration
            64: "=Q",
            -64: "=q",
        }
        format = formats[fmt]
        from pygame.tests.test_utils import buftools

        Exporter = buftools.Exporter
        Importer = buftools.Importer
        is_lil_endian = pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN
        fsys, frev = ("<", ">") if is_lil_endian else (">", "<")
        shape = (10, channels)[:ndim]
        strides = (channels * itemsize, itemsize)[2 - ndim :]
        exp = Exporter(shape, format=frev + "i")
        snd = mixer.Sound(array=exp)
        buflen = len(exp) * itemsize * channels
        imp = Importer(snd, buftools.PyBUF_SIMPLE)
        self.assertEqual(imp.ndim, 0)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertTrue(imp.shape is None)
        self.assertTrue(imp.strides is None)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_WRITABLE)
        self.assertEqual(imp.ndim, 0)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertTrue(imp.shape is None)
        self.assertTrue(imp.strides is None)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_FORMAT)
        self.assertEqual(imp.ndim, 0)
        self.assertEqual(imp.format, format)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertTrue(imp.shape is None)
        self.assertTrue(imp.strides is None)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_ND)
        self.assertEqual(imp.ndim, ndim)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertEqual(imp.shape, shape)
        self.assertTrue(imp.strides is None)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_STRIDES)
        self.assertEqual(imp.ndim, ndim)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertEqual(imp.shape, shape)
        self.assertEqual(imp.strides, strides)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_FULL_RO)
        self.assertEqual(imp.ndim, ndim)
        self.assertEqual(imp.format, format)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, 2)
        self.assertEqual(imp.shape, shape)
        self.assertEqual(imp.strides, strides)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_FULL_RO)
        self.assertEqual(imp.ndim, ndim)
        self.assertEqual(imp.format, format)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertEqual(imp.shape, exp.shape)
        self.assertEqual(imp.strides, strides)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_C_CONTIGUOUS)
        self.assertEqual(imp.ndim, ndim)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.strides, strides)
        imp = Importer(snd, buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertEqual(imp.ndim, ndim)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.strides, strides)
        if ndim == 1:
            imp = Importer(snd, buftools.PyBUF_F_CONTIGUOUS)
            self.assertEqual(imp.ndim, 1)
            self.assertTrue(imp.format is None)
            self.assertEqual(imp.strides, strides)
        else:
            self.assertRaises(BufferError, Importer, snd, buftools.PyBUF_F_CONTIGUOUS)

    def todo_test_fadeout(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.fadeout:

        # pygame.mixer.fadeout(time): return None
        # fade out the volume on all sounds before stopping
        #
        # This will fade out the volume on all active channels over the time
        # argument in milliseconds. After the sound is muted the playback will
        # stop.
        #

        self.fail()

    def test_find_channel(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.find_channel:

        # pygame.mixer.find_channel(force=False): return Channel
        # find an unused channel
        mixer.init()

        filename = example_path(os.path.join("data", "house_lo.wav"))
        sound = mixer.Sound(file=filename)

        num_channels = mixer.get_num_channels()

        if num_channels > 0:
            found_channel = mixer.find_channel()
            self.assertIsNotNone(found_channel)

            # try playing on all channels
            channels = []
            for channel_id in range(0, num_channels):
                channel = mixer.Channel(channel_id)
                channel.play(sound)
                channels.append(channel)

            # should fail without being forceful
            found_channel = mixer.find_channel()
            self.assertIsNone(found_channel)

            # try forcing without keyword
            found_channel = mixer.find_channel(True)
            self.assertIsNotNone(found_channel)

            # try forcing with keyword
            found_channel = mixer.find_channel(force=True)
            self.assertIsNotNone(found_channel)

            for channel in channels:
                channel.stop()
            found_channel = mixer.find_channel()
            self.assertIsNotNone(found_channel)

    def todo_test_get_busy(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.get_busy:

        # pygame.mixer.get_busy(): return bool
        # test if any sound is being mixed
        #
        # Returns True if the mixer is busy mixing any channels. If the mixer
        # is idle then this return False.
        #

        self.fail()

    def todo_test_pause(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.pause:

        # pygame.mixer.pause(): return None
        # temporarily stop playback of all sound channels
        #
        # This will temporarily stop all playback on the active mixer
        # channels. The playback can later be resumed with
        # pygame.mixer.unpause()
        #

        self.fail()

    def test_set_reserved(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.set_reserved:

        # pygame.mixer.set_reserved(count): return count
        mixer.init()
        default_num_channels = mixer.get_num_channels()

        # try reserving all the channels
        result = mixer.set_reserved(default_num_channels)
        self.assertEqual(result, default_num_channels)

        # try reserving all the channels + 1
        result = mixer.set_reserved(default_num_channels + 1)
        # should still be default
        self.assertEqual(result, default_num_channels)

        # try unreserving all
        result = mixer.set_reserved(0)
        # should still be default
        self.assertEqual(result, 0)

        # try reserving half
        result = mixer.set_reserved(int(default_num_channels / 2))
        # should still be default
        self.assertEqual(result, int(default_num_channels / 2))

    def todo_test_stop(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.stop:

        # pygame.mixer.stop(): return None
        # stop playback of all sound channels
        #
        # This will stop all playback of all active mixer channels.

        self.fail()

    def todo_test_unpause(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.unpause:

        # pygame.mixer.unpause(): return None
        # resume paused playback of sound channels
        #
        # This will resume all active sound channels after they have been paused.

        self.fail()

    def test_get_sdl_mixer_version(self):
        """Ensures get_sdl_mixer_version works correctly with no args."""
        expected_length = 3
        expected_type = tuple
        expected_item_type = int

        version = pygame.mixer.get_sdl_mixer_version()

        self.assertIsInstance(version, expected_type)
        self.assertEqual(len(version), expected_length)

        for item in version:
            self.assertIsInstance(item, expected_item_type)

    def test_get_sdl_mixer_version__args(self):
        """Ensures get_sdl_mixer_version works correctly using args."""
        expected_length = 3
        expected_type = tuple
        expected_item_type = int

        for value in (True, False):
            version = pygame.mixer.get_sdl_mixer_version(value)

            self.assertIsInstance(version, expected_type)
            self.assertEqual(len(version), expected_length)

            for item in version:
                self.assertIsInstance(item, expected_item_type)

    def test_get_sdl_mixer_version__kwargs(self):
        """Ensures get_sdl_mixer_version works correctly using kwargs."""
        expected_length = 3
        expected_type = tuple
        expected_item_type = int

        for value in (True, False):
            version = pygame.mixer.get_sdl_mixer_version(linked=value)

            self.assertIsInstance(version, expected_type)
            self.assertEqual(len(version), expected_length)

            for item in version:
                self.assertIsInstance(item, expected_item_type)

    def test_get_sdl_mixer_version__invalid_args_kwargs(self):
        """Ensures get_sdl_mixer_version handles invalid args and kwargs."""
        invalid_bool = InvalidBool()

        with self.assertRaises(TypeError):
            version = pygame.mixer.get_sdl_mixer_version(invalid_bool)

        with self.assertRaises(TypeError):
            version = pygame.mixer.get_sdl_mixer_version(linked=invalid_bool)

    def test_get_sdl_mixer_version__linked_equals_compiled(self):
        """Ensures get_sdl_mixer_version's linked/compiled versions are equal."""
        linked_version = pygame.mixer.get_sdl_mixer_version(linked=True)
        complied_version = pygame.mixer.get_sdl_mixer_version(linked=False)

        self.assertTupleEqual(linked_version, complied_version)


############################## CHANNEL CLASS TESTS #############################


class ChannelTypeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initializing the mixer is slow, so minimize the times it is called.
        mixer.init()

    @classmethod
    def tearDownClass(cls):
        mixer.quit()

    def setUp(cls):
        # This makes sure the mixer is always initialized before each test (in
        # case a test calls pygame.mixer.quit()).
        if mixer.get_init() is None:
            mixer.init()

    def test_channel(self):
        """Ensure Channel() creation works."""
        channel = mixer.Channel(0)

        self.assertIsInstance(channel, mixer.ChannelType)
        self.assertEqual(channel.__class__.__name__, "Channel")

    def test_channel__without_arg(self):
        """Ensure exception for Channel() creation with no argument."""
        with self.assertRaises(TypeError):
            mixer.Channel()

    def test_channel__invalid_id(self):
        """Ensure exception for Channel() creation with an invalid id."""
        with self.assertRaises(IndexError):
            mixer.Channel(-1)

    def test_channel__before_init(self):
        """Ensure exception for Channel() creation with non-init mixer."""
        mixer.quit()

        with self.assertRaisesRegex(pygame.error, "mixer not initialized"):
            mixer.Channel(0)

    def todo_test_fadeout(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.fadeout:

        # Channel.fadeout(time): return None
        # stop playback after fading channel out
        #
        # Stop playback of a channel after fading out the sound over the given
        # time argument in milliseconds.
        #

        self.fail()

    def test_get_busy(self):
        """Ensure an idle channel's busy state is correct."""
        expected_busy = False
        channel = mixer.Channel(0)

        busy = channel.get_busy()

        self.assertEqual(busy, expected_busy)

    def todo_test_get_busy__active(self):
        """Ensure an active channel's busy state is correct."""
        self.fail()

    def todo_test_get_endevent(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.get_endevent:

        # Channel.get_endevent(): return type
        # get the event a channel sends when playback stops
        #
        # Returns the event type to be sent every time the Channel finishes
        # playback of a Sound. If there is no endevent the function returns
        # pygame.NOEVENT.
        #

        self.fail()

    def todo_test_get_queue(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.get_queue:

        # Channel.get_queue(): return Sound
        # return any Sound that is queued
        #
        # If a Sound is already queued on this channel it will be returned.
        # Once the queued sound begins playback it will no longer be on the
        # queue.
        #

        self.fail()

    def todo_test_get_sound(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.get_sound:

        # Channel.get_sound(): return Sound
        # get the currently playing Sound
        #
        # Return the actual Sound object currently playing on this channel. If
        # the channel is idle None is returned.
        #

        self.fail()

    def test_get_volume(self):
        """Ensure a channel's volume can be retrieved."""
        expected_volume = 1.0  # default
        channel = mixer.Channel(0)

        volume = channel.get_volume()

        self.assertAlmostEqual(volume, expected_volume)

    def todo_test_get_volume__while_playing(self):
        """Ensure a channel's volume can be retrieved while playing."""
        self.fail()

    def todo_test_pause(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.pause:

        # Channel.pause(): return None
        # temporarily stop playback of a channel
        #
        # Temporarily stop the playback of sound on a channel. It can be
        # resumed at a later time with Channel.unpause()
        #

        self.fail()

    def todo_test_play(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.play:

        # Channel.play(Sound, loops=0, maxtime=0, fade_ms=0): return None
        # play a Sound on a specific Channel
        #
        # This will begin playback of a Sound on a specific Channel. If the
        # Channel is currently playing any other Sound it will be stopped.
        #
        # The loops argument has the same meaning as in Sound.play(): it is
        # the number of times to repeat the sound after the first time. If it
        # is 3, the sound will be played 4 times (the first time, then three
        # more). If loops is -1 then the playback will repeat indefinitely.
        #
        # As in Sound.play(), the maxtime argument can be used to stop
        # playback of the Sound after a given number of milliseconds.
        #
        # As in Sound.play(), the fade_ms argument can be used fade in the sound.

        self.fail()

    def todo_test_queue(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.queue:

        # Channel.queue(Sound): return None
        # queue a Sound object to follow the current
        #
        # When a Sound is queued on a Channel, it will begin playing
        # immediately after the current Sound is finished. Each channel can
        # only have a single Sound queued at a time. The queued Sound will
        # only play if the current playback finished automatically. It is
        # cleared on any other call to Channel.stop() or Channel.play().
        #
        # If there is no sound actively playing on the Channel then the Sound
        # will begin playing immediately.
        #

        self.fail()

    def todo_test_set_endevent(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.set_endevent:

        # Channel.set_endevent(): return None
        # Channel.set_endevent(type): return None
        # have the channel send an event when playback stops
        #
        # When an endevent is set for a channel, it will send an event to the
        # pygame queue every time a sound finishes playing on that channel
        # (not just the first time). Use pygame.event.get() to retrieve the
        # endevent once it's sent.
        #
        # Note that if you called Sound.play(n) or Channel.play(sound,n), the
        # end event is sent only once: after the sound has been played "n+1"
        # times (see the documentation of Sound.play).
        #
        # If Channel.stop() or Channel.play() is called while the sound was
        # still playing, the event will be posted immediately.
        #
        # The type argument will be the event id sent to the queue. This can
        # be any valid event type, but a good choice would be a value between
        # pygame.locals.USEREVENT and pygame.locals.NUMEVENTS. If no type
        # argument is given then the Channel will stop sending endevents.
        #

        self.fail()

    def todo_test_set_volume(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.set_volume:

        # Channel.set_volume(value): return None
        # Channel.set_volume(left, right): return None
        # set the volume of a playing channel
        #
        # Set the volume (loudness) of a playing sound. When a channel starts
        # to play its volume value is reset. This only affects the current
        # sound. The value argument is between 0.0 and 1.0.
        #
        # If one argument is passed, it will be the volume of both speakers.
        # If two arguments are passed and the mixer is in stereo mode, the
        # first argument will be the volume of the left speaker and the second
        # will be the volume of the right speaker. (If the second argument is
        # None, the first argument will be the volume of both speakers.)
        #
        # If the channel is playing a Sound on which set_volume() has also
        # been called, both calls are taken into account. For example:
        #
        #     sound = pygame.mixer.Sound("s.wav")
        #     channel = s.play()      # Sound plays at full volume by default
        #     sound.set_volume(0.9)   # Now plays at 90% of full volume.
        #     sound.set_volume(0.6)   # Now plays at 60% (previous value replaced).
        #     channel.set_volume(0.5) # Now plays at 30% (0.6 * 0.5).

        self.fail()

    def todo_test_stop(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.stop:

        # Channel.stop(): return None
        # stop playback on a Channel
        #
        # Stop sound playback on a channel. After playback is stopped the
        # channel becomes available for new Sounds to play on it.
        #

        self.fail()

    def todo_test_unpause(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.unpause:

        # Channel.unpause(): return None
        # resume pause playback of a channel
        #
        # Resume the playback on a paused channel.

        self.fail()


############################### SOUND CLASS TESTS ##############################


class SoundTypeTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        mixer.quit()

    def setUp(cls):
        # This makes sure the mixer is always initialized before each test (in
        # case a test calls pygame.mixer.quit()).
        if mixer.get_init() is None:
            mixer.init()

    # See MixerModuleTest's methods test_sound_args(), test_sound_unicode(),
    # and test_array_keyword() for additional testing of Sound() creation.
    def test_sound(self):
        """Ensure Sound() creation with a filename works."""
        filename = example_path(os.path.join("data", "house_lo.wav"))
        sound1 = mixer.Sound(filename)
        sound2 = mixer.Sound(file=filename)

        self.assertIsInstance(sound1, mixer.Sound)
        self.assertIsInstance(sound2, mixer.Sound)

    def test_sound__from_file_object(self):
        """Ensure Sound() creation with a file object works."""
        filename = example_path(os.path.join("data", "house_lo.wav"))

        # Using 'with' ensures the file is closed even if test fails.
        with open(filename, "rb") as file_obj:
            sound = mixer.Sound(file_obj)

            self.assertIsInstance(sound, mixer.Sound)

    def test_sound__from_sound_object(self):
        """Ensure Sound() creation with a Sound() object works."""
        filename = example_path(os.path.join("data", "house_lo.wav"))
        sound_obj = mixer.Sound(file=filename)

        sound = mixer.Sound(sound_obj)

        self.assertIsInstance(sound, mixer.Sound)

    def test_sound__from_pathlib(self):
        """Ensure Sound() creation with a pathlib.Path object works."""
        path = pathlib.Path(example_path(os.path.join("data", "house_lo.wav")))
        sound1 = mixer.Sound(path)
        sound2 = mixer.Sound(file=path)
        self.assertIsInstance(sound1, mixer.Sound)
        self.assertIsInstance(sound2, mixer.Sound)

    def todo_test_sound__from_buffer(self):
        """Ensure Sound() creation with a buffer works."""
        self.fail()

    def todo_test_sound__from_array(self):
        """Ensure Sound() creation with an array works."""
        self.fail()

    def test_sound__without_arg(self):
        """Ensure exception raised for Sound() creation with no argument."""
        with self.assertRaises(TypeError):
            mixer.Sound()

    def test_sound__before_init(self):
        """Ensure exception raised for Sound() creation with non-init mixer."""
        mixer.quit()
        filename = example_path(os.path.join("data", "house_lo.wav"))

        with self.assertRaisesRegex(pygame.error, "mixer not initialized"):
            mixer.Sound(file=filename)

    @unittest.skipIf(IS_PYPY, "pypy skip")
    def test_samples_address(self):
        """Test the _samples_address getter."""
        try:
            from ctypes import pythonapi, c_void_p, py_object

            Bytes_FromString = pythonapi.PyBytes_FromString

            Bytes_FromString.restype = c_void_p
            Bytes_FromString.argtypes = [py_object]
            samples = b"abcdefgh"  # keep byte size a multiple of 4
            sample_bytes = Bytes_FromString(samples)

            snd = mixer.Sound(buffer=samples)

            self.assertNotEqual(snd._samples_address, sample_bytes)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, "mixer not initialized"):
                snd._samples_address

    def todo_test_fadeout(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.Sound.fadeout:

        # Sound.fadeout(time): return None
        # stop sound playback after fading out
        #
        # This will stop playback of the sound after fading it out over the
        # time argument in milliseconds. The Sound will fade and stop on all
        # actively playing channels.
        #

        self.fail()

    def test_get_length(self):
        """Tests if get_length returns a correct length."""
        try:
            for size in SIZES:
                pygame.mixer.quit()
                pygame.mixer.init(size=size)
                filename = example_path(os.path.join("data", "punch.wav"))
                sound = mixer.Sound(file=filename)
                # The sound data is in the mixer output format. So dividing the
                # length of the raw sound data by the mixer settings gives
                # the expected length of the sound.
                sound_bytes = sound.get_raw()
                mix_freq, mix_bits, mix_channels = pygame.mixer.get_init()
                mix_bytes = abs(mix_bits) / 8
                expected_length = (
                    float(len(sound_bytes)) / mix_freq / mix_bytes / mix_channels
                )
                self.assertAlmostEqual(expected_length, sound.get_length())
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, "mixer not initialized"):
                sound.get_length()

    def test_get_num_channels(self):
        """
        Tests if Sound.get_num_channels returns the correct number
        of channels playing a specific sound.
        """
        try:
            filename = example_path(os.path.join("data", "house_lo.wav"))
            sound = mixer.Sound(file=filename)

            self.assertEqual(sound.get_num_channels(), 0)
            sound.play()
            self.assertEqual(sound.get_num_channels(), 1)
            sound.play()
            self.assertEqual(sound.get_num_channels(), 2)
            sound.stop()
            self.assertEqual(sound.get_num_channels(), 0)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, "mixer not initialized"):
                sound.get_num_channels()

    def test_get_volume(self):
        """Ensure a sound's volume can be retrieved."""
        try:
            expected_volume = 1.0  # default
            filename = example_path(os.path.join("data", "house_lo.wav"))
            sound = mixer.Sound(file=filename)

            volume = sound.get_volume()

            self.assertAlmostEqual(volume, expected_volume)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, "mixer not initialized"):
                sound.get_volume()

    def todo_test_get_volume__while_playing(self):
        """Ensure a sound's volume can be retrieved while playing."""
        self.fail()

    def todo_test_play(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.Sound.play:

        # Sound.play(loops=0, maxtime=0, fade_ms=0): return Channel
        # begin sound playback
        #
        # Begin playback of the Sound (i.e., on the computer's speakers) on an
        # available Channel. This will forcibly select a Channel, so playback
        # may cut off a currently playing sound if necessary.
        #
        # The loops argument controls how many times the sample will be
        # repeated after being played the first time. A value of 5 means that
        # the sound will be played once, then repeated five times, and so is
        # played a total of six times. The default value (zero) means the
        # Sound is not repeated, and so is only played once. If loops is set
        # to -1 the Sound will loop indefinitely (though you can still call
        # stop() to stop it).
        #
        # The maxtime argument can be used to stop playback after a given
        # number of milliseconds.
        #
        # The fade_ms argument will make the sound start playing at 0 volume
        # and fade up to full volume over the time given. The sample may end
        # before the fade-in is complete.
        #
        # This returns the Channel object for the channel that was selected.

        self.fail()

    def test_set_volume(self):
        """Ensure a sound's volume can be set."""
        try:
            float_delta = 1.0 / 128  # SDL volume range is 0 to 128
            filename = example_path(os.path.join("data", "house_lo.wav"))
            sound = mixer.Sound(file=filename)
            current_volume = sound.get_volume()

            # (volume_set_value : expected_volume)
            volumes = (
                (-1, current_volume),  # value < 0 won't change volume
                (0, 0.0),
                (0.01, 0.01),
                (0.1, 0.1),
                (0.5, 0.5),
                (0.9, 0.9),
                (0.99, 0.99),
                (1, 1.0),
                (1.1, 1.0),
                (2.0, 1.0),
            )

            for volume_set_value, expected_volume in volumes:
                sound.set_volume(volume_set_value)

                self.assertAlmostEqual(
                    sound.get_volume(), expected_volume, delta=float_delta
                )
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, "mixer not initialized"):
                sound.set_volume(1)

    def todo_test_set_volume__while_playing(self):
        """Ensure a sound's volume can be set while playing."""
        self.fail()

    def test_stop(self):
        """Ensure stop can be called while not playing a sound."""
        try:
            expected_channels = 0
            filename = example_path(os.path.join("data", "house_lo.wav"))
            sound = mixer.Sound(file=filename)

            sound.stop()

            self.assertEqual(sound.get_num_channels(), expected_channels)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, "mixer not initialized"):
                sound.stop()

    def todo_test_stop__while_playing(self):
        """Ensure stop stops a playing sound."""
        self.fail()

    def test_get_raw(self):
        """Ensure get_raw returns the correct bytestring."""
        try:
            samples = b"abcdefgh"  # keep byte size a multiple of 4
            snd = mixer.Sound(buffer=samples)

            raw = snd.get_raw()

            self.assertIsInstance(raw, bytes)
            self.assertEqual(raw, samples)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, "mixer not initialized"):
                snd.get_raw()

    def test_correct_subclassing(self):
        class CorrectSublass(mixer.Sound):
            def __init__(self, file):
                super().__init__(file=file)

        filename = example_path(os.path.join("data", "house_lo.wav"))
        correct = CorrectSublass(filename)

        try:
            correct.get_volume()
        except Exception:
            self.fail("This should not raise an exception.")

    def test_incorrect_subclassing(self):
        class IncorrectSuclass(mixer.Sound):
            def __init__(self):
                pass

        incorrect = IncorrectSuclass()

        self.assertRaises(RuntimeError, incorrect.get_volume)


##################################### MAIN #####################################

if __name__ == "__main__":
    unittest.main()
