#################################### IMPORTS ###################################

if __name__ == '__main__':
    import sys
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

import unittest
if is_pygame_pkg:
    from pygame.tests.test_utils import fixture_path
else:
    from test.test_utils import fixture_path
import pygame

################################################################################

class CursorsModuleTest(unittest.TestCase):
    def todo_test_compile(self):

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
        
        self.fail()

    def test_load_xbm(self):
        # __doc__ (as of 2008-06-25) for pygame.cursors.load_xbm:

          # pygame.cursors.load_xbm(cursorfile, maskfile) -> cursor_args
          # reads a pair of XBM files into set_cursor arguments
          #
          # Arguments can either be filenames or filelike objects
          # with the readlines method. Not largely tested, but
          # should work with typical XBM files.

        # Test that load_xbm will take filenames as arguments
        cursorfile = fixture_path(r"xbm_cursors/white_sizing.xbm")
        maskfile   = fixture_path(r"xbm_cursors/white_sizing_mask.xbm")
        cursor = pygame.cursors.load_xbm(cursorfile, maskfile)
        
        # Test that load_xbm will take file objects as arguments
        cursorfile, maskfile = [open(pth) for pth in (cursorfile, maskfile)]
        cursor = pygame.cursors.load_xbm(cursorfile, maskfile)

        # Is it in a format that mouse.set_cursor won't blow up on?
        pygame.display.init()
        pygame.mouse.set_cursor(*cursor)
        pygame.display.quit()

################################################################################

if __name__ == '__main__':
    unittest.main()
    
################################################################################
