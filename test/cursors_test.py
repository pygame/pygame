import unittest
from pygame.tests.test_utils import fixture_path
import pygame


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

        # Test taht load_xbm can handle xbm cursor images that do not have a hotspot specified
        cursorfile_no_hotspot = fixture_path(r"xbm_cursors/white_no_hotspot.xbm")
        maskfile_no_hotspot = fixture_path(r"xbm_cursors/white_no_hotspot_mask.xbm")
        cursor_no_hotspot = pygame.cursors.load_xbm(cursorfile_no_hotspot, maskfile_no_hotspot)

        # Assert that image size was extracted correctly
        self.assertEqual(cursor_no_hotspot[0],(16, 16))

        # Assert that the hotspot coordinates was set correctly
        self.assertEqual(cursor_no_hotspot[1],(0,0))

        # Assert that the pixels from the cursor file were extracted correctly
        self.assertEqual(cursor_no_hotspot[2],(0, 63, 0, 63, 15, 255, 7, 255, 35, 255, 49, 255, 56, 255, 60, 127, 62, 63, 63, 31, 255, 143, 255, 199, 255, 227, 255, 241, 255, 248, 255, 252))

        # Assert that the pixels from the mask file were extraccted correctly
        self.assertEqual(cursor_no_hotspot[3],(255, 192, 255, 192, 240, 0, 248, 0, 220, 0, 206, 0, 199, 0, 195, 128, 193, 192, 192, 224, 0, 112, 0, 56, 0, 28, 0, 14, 0, 7, 0, 3))


################################################################################

if __name__ == '__main__':
    unittest.main()

################################################################################
