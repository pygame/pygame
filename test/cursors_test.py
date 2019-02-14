import unittest
from pygame.tests.test_utils import fixture_path
import pygame


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

        thickarrow_strings = (               #sized 24x24
          "XX                      ",
          "XXX                     ",
          "XXXX                    ",
          "XX.XX                   ",
          "XX..XX                  ",
          "XX...XX                 ",
          "XX....XX                ",
          "XX.....XX               ",
          "XX......XX              ",
          "XX.......XX             ",
          "XX........XX            ",
          "XX........XXX           ",
          "XX......XXXXX           ",
          "XX.XXX..XX              ",
          "XXXX XX..XX             ",
          "XX   XX..XX             ",
          "     XX..XX             ",
          "      XX..XX            ",
          "      XX..XX            ",
          "       XXXX             ",
          "       XX               ",
          "                        ",
          "                        ",
          "                        ",
        )

        #Compile the thickarrow_strings cursor
        compiled_cursor = pygame.cursors.compile(thickarrow_strings, black='X', white='.')

        #Assert that the returned cursor data is correct
        self.assertEqual(compiled_cursor[0], (192, 0, 0, 224, 0, 0, 240, 0, 0, 216, 0, 0, 204, 0, 0, 198, 0, 0, 195, 0, 0, 193, 128, 0, 192, 192, 0, 192, 96, 0, 192, 48,
                                              0, 192, 56, 0, 192, 248, 0, 220, 192, 0, 246, 96, 0, 198, 96, 0, 6, 96, 0, 3, 48, 0, 3, 48, 0, 1, 224, 0, 1, 128, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0))

        #Assert that the returned cursor mask data is correct
        self.assertEqual(compiled_cursor[1], (192, 0, 0, 224, 0, 0, 240, 0, 0, 248, 0, 0, 252, 0, 0, 254, 0, 0, 255, 0, 0, 255, 128, 0, 255, 192, 0, 255, 224, 0, 255, 240,
                                              0, 255, 248, 0, 255, 248, 0, 255, 192, 0, 247, 224, 0, 199, 224, 0, 7, 224, 0, 3, 240, 0, 3, 240, 0, 1, 224, 0, 1, 128, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0))


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
