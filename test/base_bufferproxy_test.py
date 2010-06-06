import sys
import unittest

from pygame2.base import BufferProxy, Color
from pygame2.colorpalettes import VGAPALETTE
import pygame2.sdl.video as video

class BufferProxyTest (unittest.TestCase):
    __tags__ = [ "sdl" ]

    def setUp (self):
        video.init ()

    def tearDown (self):
        video.quit ()
    
    def test_pygame2_base_BufferProxy_length(self):

        # __doc__ (as of 2010-01-13) for pygame2.base.BufferProxy.length:

        # Gets the size of the buffer data in bytes.
        for bpp in (32, 24, 16, 8):
            surface = video.Surface (10, 10, bpp)
            # 10 * 10 * bpp/8 byte
            buf = surface.pixels
            self.assertEqual (buf.length, surface.h * surface.pitch)
            del buf

    def test_pygame2_base_BufferProxy_raw(self):

        # __doc__ (as of 2010-01-13) for pygame2.base.BufferProxy.raw:

        # Gets the raw buffer data as string. The string may contain
        # NUL bytes.
        for bpp in (32, 24, 16, 8):
            surface = video.Surface (10, 10, bpp)
            buf = surface.pixels
            if sys.version_info < (3, 0):
                for b in buf.raw:
                    self.assertEqual (b, '\x00')
            else:
                for b in buf.raw:
                    self.assertEqual (b, 0)
            del buf

    def test_pygame2_base_BufferProxy_write(self):

        # __doc__ (as of 2010-01-13) for pygame2.base.BufferProxy.write:

        # write (buffer, offset) -> None
        # 
        # Writes raw data to the BufferProxy.
        # 
        # Writes the raw data from *buffer* to the BufferProxy object,
        # starting at the specified *offset* within the BufferProxy. If
        # the length of the passed *buffer* exceeds the length of the
        # BufferProxy (reduced by *offset*), an IndexError will
        # be raised.
        for bpp in (32, 16, 8):
            surface = video.Surface (10, 10, bpp)
            buf = surface.pixels
            for y in range (surface.height):
                for x in range (surface.width):
                    buf.write ('\xff', x + y * surface.pitch)
            del buf

            # getat = surface.get_at
            # color = Color (255, 255, 255, 255)
            # for x in range (surface.width):
            #     for y in range (surface.height):
            #         self.failUnlessEqual (getat (x, y), color,
            #             "%s != %s at (%d, %d)" % (getat (x, y), color, x, y))

if __name__ == '__main__':
    unittest.main ()

