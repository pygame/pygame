try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

import pygame2
import pygame2.sdl.cursors as cursors
import pygame2.sdl.constants as constants

class SDLCursorsTest (unittest.TestCase):

    def todo_test_pygame2_sdl_cursors_compile(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cursors.compile:

        # compile(strings, black, white,xor) -> data, mask
        # 
        # Compile cursor strings into cursor data
        # 
        # This takes a set of strings with equal length and computes
        # the binary data for that cursor. The string widths must be
        # divisible by 8.
        # 
        # The black and white arguments are single letter strings that
        # tells which characters will represent black pixels, and which
        # characters represent white pixels. All other characters are
        # considered clear.
        # 
        # This returns a tuple containing the cursor data and cursor mask
        # data. Both these arguments are used when setting a cursor with
        # pygame2.sdl.mouse.set_cursor().

        self.fail() 

    def todo_test_pygame2_sdl_cursors_load_xbm(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cursors.load_xbm:

        # load_xbm(cursorfile, maskfile) -> cursor_args
        # 
        # Reads a pair of XBM files into set_cursor arguments
        # 
        # Arguments can either be filenames or filelike objects
        # with the readlines method. Not largely tested, but
        # should work with typical XBM files.

        self.fail() 

if __name__ == "__main__":
    unittest.main ()
