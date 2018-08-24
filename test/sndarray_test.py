import unittest

from numpy import int8, int16, uint8, uint16, array, alltrue

import pygame
from pygame.compat import as_bytes
import pygame.sndarray


class SndarrayTest (unittest.TestCase):
    array_dtypes = {8: uint8, -8: int8, 16: uint16, -16: int16}

    def _assert_compatible(self, arr, size):
        dtype = self.array_dtypes[size]
        self.assertEqual(arr.dtype, dtype)

    def test_array(self):

        def check_array(size, channels, test_data):
            try:
                pygame.mixer.init(22050, size, channels)
            except pygame.error:
                # Not all sizes are supported on all systems.
                return
            try:
                __, sz, __ = pygame.mixer.get_init()
                if sz == size:
                    srcarr = array(test_data, self.array_dtypes[size])
                    snd = pygame.sndarray.make_sound(srcarr)
                    arr = pygame.sndarray.array(snd)
                    self._assert_compatible(arr, size)
                    self.failUnless(alltrue(arr == srcarr),
                                    "size: %i\n%s\n%s" %
                                    (size, arr, test_data))
            finally:
                pygame.mixer.quit()

        check_array(8, 1, [0, 0x0f, 0xf0, 0xff])
        check_array(8, 2,
                    [[0, 0x80], [0x2D, 0x41], [0x64, 0xA1], [0xff, 0x40]])
        check_array(16, 1, [0, 0x00ff, 0xff00, 0xffff])
        check_array(16, 2, [[0, 0xffff], [0xffff, 0],
                            [0x00ff, 0xff00], [0x0f0f, 0xf0f0]])
        check_array(-8, 1, [0, -0x80, 0x7f, 0x64])
        check_array(-8, 2,
                    [[0, -0x80], [-0x64, 0x64], [0x25, -0x50], [0xff, 0]])
        check_array(-16, 1, [0, 0x7fff, -0x7fff, -1])
        check_array(-16, 2, [[0, -0x7fff], [-0x7fff, 0],
                             [0x7fff, 0], [0, 0x7fff]])

    def test_get_arraytype(self):
        self.failUnless((pygame.sndarray.get_arraytype() in
                         ['numpy']),
                        ("unknown array type %s" %
                         pygame.sndarray.get_arraytype()))

    def test_get_arraytypes(self):
        arraytypes = pygame.sndarray.get_arraytypes()
        self.failUnless('numpy' in arraytypes)

        for atype in arraytypes:
            self.failUnless(atype in ['numpy'],
                            "unknown array type %s" % atype)

    def test_make_sound(self):

        def check_sound(size, channels, test_data):
            try:
                pygame.mixer.init(22050, size, channels)
            except pygame.error:
                # Not all sizes are supported on all systems.
                return
            try:
                __, sz, __ = pygame.mixer.get_init()
                if sz == size:
                    srcarr = array(test_data, self.array_dtypes[size])
                    snd = pygame.sndarray.make_sound(srcarr)
                    arr = pygame.sndarray.samples(snd)
                    self.failUnless(alltrue(arr == srcarr),
                                    "size: %i\n%s\n%s" %
                                    (size, arr, test_data))
            finally:
                pygame.mixer.quit()

        check_sound(8, 1, [0, 0x0f, 0xf0, 0xff])
        check_sound(8, 2,
                    [[0, 0x80], [0x2D, 0x41], [0x64, 0xA1], [0xff, 0x40]])
        check_sound(16, 1, [0, 0x00ff, 0xff00, 0xffff])
        check_sound(16, 2, [[0, 0xffff], [0xffff, 0],
                            [0x00ff, 0xff00], [0x0f0f, 0xf0f0]])
        check_sound(-8, 1, [0, -0x80, 0x7f, 0x64])
        check_sound(-8, 2,
                    [[0, -0x80], [-0x64, 0x64], [0x25, -0x50], [0xff, 0]])
        check_sound(-16, 1, [0, 0x7fff, -0x7fff, -1])
        check_sound(-16, 2, [[0, -0x7fff], [-0x7fff, 0],
                             [0x7fff, 0], [0, 0x7fff]])

    def test_samples(self):

        null_byte = as_bytes('\x00')
        def check_sample(size, channels, test_data):
            try:
                pygame.mixer.init(22050, size, channels)
            except pygame.error:
                # Not all sizes are supported on all systems.
                return
            try:
                __, sz, __ = pygame.mixer.get_init()
                if sz == size:
                    zeroed = null_byte * ((abs(size) // 8) *
                                          len(test_data) *
                                          channels)
                    snd = pygame.mixer.Sound(buffer=zeroed)
                    samples = pygame.sndarray.samples(snd)
                    self._assert_compatible(samples, size)
                    ##print ('X %s' % (samples.shape,))
                    ##print ('Y %s' % (test_data,))
                    samples[...] = test_data
                    arr = pygame.sndarray.array(snd)
                    self.failUnless(alltrue(samples == arr),
                                    "size: %i\n%s\n%s" %
                                    (size, arr, test_data))
            finally:
                pygame.mixer.quit()

        check_sample(8, 1, [0, 0x0f, 0xf0, 0xff])
        check_sample(8, 2,
                    [[0, 0x80], [0x2D, 0x41], [0x64, 0xA1], [0xff, 0x40]])
        check_sample(16, 1, [0, 0x00ff, 0xff00, 0xffff])
        check_sample(16, 2, [[0, 0xffff], [0xffff, 0],
                            [0x00ff, 0xff00], [0x0f0f, 0xf0f0]])
        check_sample(-8, 1, [0, -0x80, 0x7f, 0x64])
        check_sample(-8, 2,
                    [[0, -0x80], [-0x64, 0x64], [0x25, -0x50], [0xff, 0]])
        check_sample(-16, 1, [0, 0x7fff, -0x7fff, -1])
        check_sample(-16, 2, [[0, -0x7fff], [-0x7fff, 0],
                             [0x7fff, 0], [0, 0x7fff]])

    def test_use_arraytype(self):

        def do_use_arraytype(atype):
            pygame.sndarray.use_arraytype(atype)

        pygame.sndarray.use_arraytype('numpy')
        self.assertEqual(pygame.sndarray.get_arraytype(), 'numpy')

        self.assertRaises(ValueError, do_use_arraytype, 'not an option')


if __name__ == '__main__':
    unittest.main()
