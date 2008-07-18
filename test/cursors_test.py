#################################### IMPORTS ###################################

import test_utils
import test.unittest as unittest
from test_utils import test_not_implemented

################################################################################

class CursorsModuleTest(unittest.TestCase):
    def test_compile(self):

        # __doc__ (as of 2008-06-25) for pygame.cursors.compile:

          # pygame.cursors.compile(strings, black, white,xor) -> data, mask
          # compile cursor strings into cursor data
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
          # pygame.mouse.set_cursor().

        self.assert_(test_not_implemented()) 

    def test_load_xbm(self):

        # __doc__ (as of 2008-06-25) for pygame.cursors.load_xbm:

          # pygame.cursors.load_xbm(cursorfile, maskfile) -> cursor_args
          # reads a pair of XBM files into set_cursor arguments
          # 
          # Arguments can either be filenames or filelike objects
          # with the readlines method. Not largely tested, but
          # should work with typical XBM files.

        self.assert_(test_not_implemented()) 


################################################################################

if __name__ == '__main__':
    unittest.main()
