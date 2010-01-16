import sys
try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

from pygame2.base import BufferProxy, Color
from pygame2.colorpalettes import VGAPALETTE
import pygame2.sdl.video as video

class BufferProxyTest (unittest.TestCase):
    
    def test_pygame2_base_BufferProxy_length(self):

        # __doc__ (as of 2010-01-13) for pygame2.base.BufferProxy.length:

        # Gets the size of the buffer data in bytes.
        video.init ()
        for bpp in (32, 24, 16, 8):
            surface = video.Surface (10, 10, bpp)
            # 10 * 10 * bpp/8 byte
            buf = surface.pixels
            self.assertEqual (buf.length, surface.h * surface.pitch)
            del buf
        video.quit ()

    def test_pygame2_base_BufferProxy_raw(self):

        # __doc__ (as of 2010-01-13) for pygame2.base.BufferProxy.raw:

        # Gets the raw buffer data as string. The string may contain
        # NUL bytes.
        video.init ()
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
        video.quit ()

    def todo_test_pygame2_base_BufferProxy_write(self):

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
        self.fail ()

if __name__ == '__main__':
    unittest.main ()

