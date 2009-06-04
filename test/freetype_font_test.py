import os
try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

import pygame2
import pygame2.freetype as ft
import pygame2.freetype.constants as ft_const

ft.init()

FONTDIR = os.path.dirname (os.path.abspath (__file__))

class FreeTypeFontTest(unittest.TestCase):
    _TEST_FONTS = {
            # Inconsolata is an open-source font designed by Raph Levien
            # Licensed under the Open Font License
            # http://www.levien.com/type/myfonts/inconsolata.html
            'fixed' : ft.Font(os.path.join (FONTDIR, 'test_fixed.otf')),

            # Liberation Sans is an open-source font designed by Steve Matteson
            # Licensed under the GNU GPL
            # https://fedorahosted.org/liberation-fonts/
            'sans'  : ft.Font(os.path.join (FONTDIR, 'test_sans.ttf')),
    }

    def test_pygame2_freetype_Font_init(self):
        self.assertRaises(RuntimeError, ft.Font, os.path.join (FONTDIR, 'nonexistant.ttf'))

        f = self._TEST_FONTS['sans']
        self.assertTrue(isinstance(f, pygame2.base.Font))
        self.assertTrue(isinstance(f, ft.Font))

        f = self._TEST_FONTS['fixed']
        self.assertTrue(isinstance(f, pygame2.base.Font))
        self.assertTrue(isinstance(f, ft.Font))


    def test_pygame2_freetype_Font_fixed_width(self):

        # __doc__ (as of 2009-05-25) for pygame2.freetype.Font.fixed_width:

        # Returns whether this font is a fixed-width (bitmap) font

        f = self._TEST_FONTS['sans']
        self.assertFalse(f.fixed_width)

        f = self._TEST_FONTS['fixed']
        self.assertFalse(f.fixed_width)

        # TODO: Find a real fixed width font to test with

    def test_pygame2_freetype_Font_get_metrics(self):

        # __doc__ (as of 2009-05-25) for pygame2.freetype.Font.get_metrics:

        # get_metrics(pt, text) -> [(int, int, int ...), ...]
        # 
        # Returns the glyph metrics for each character in 'text'.

        TEST_VALUES_FIXED = {
            12 : [
                (0, 6, 0, 8, 6), (0, 6, 0, 8, 6), (0, 6, 0, 8, 6),
                (0, 6, 0, 8, 6), (0, 6, 0, 6, 6), (0, 6, 0, 8, 6),
                (0, 6, 0, 6, 6), (0, 6, 0, 8, 6)],
            18 : [
                (0, 9, 0, 12, 9), (0, 9, 0, 12, 9), (0, 9, 0, 12, 9),
                (0, 9, 0, 12, 9), (0, 8, 0, 9, 9), (1, 9, 0, 12, 9),
                (0, 9, 0, 9, 9), (0, 8, 0, 12, 9)],
            24 : [
                (0, 12, 0, 16, 12), (1, 11, 0, 15, 12), (0, 12, 0, 15, 12),
                (1, 12, 0, 15, 12), (1, 11, 0, 11, 12), (1, 11, 0, 16, 12),
                (1, 11, 0, 11, 12), (1, 11, 0, 16, 12)],
            32 : [
                (0, 16, 0, 21, 16), (1, 15, 0, 20, 16), (1, 15, 0, 20, 16),
                (1, 15, 0, 20, 16), (1, 14, 0, 15, 16), (1, 15, 0, 22, 16),
                (1, 15, 0, 15, 16), (1, 15, 0, 22, 16)],
            48 : [
                (0, 24, -1, 31, 24), (2, 22, 0, 30, 24), (1, 23, -1, 31, 24),
                (2, 23, -1, 30, 24), (2, 21, -1, 23, 24), (2, 22, -1, 32, 24),
                (2, 22, -1, 23, 24), (2, 22, -1, 32, 24)],
        }

        TEST_VALUES_SANS = {
            12 : [
                (0, 7, 0, 9, 7), (1, 7, 0, 9, 8), (1, 8, 0, 9, 9),
                (1, 8, 0, 9, 9), (1, 7, 0, 7, 7), (0, 6, 0, 9, 7),
                (1, 5, 0, 7, 6), (1, 7, 0, 9, 7)],
            18 : [
                (0, 11, 0, 12, 11), (1, 11, 0, 12, 12), (1, 12, 0, 12, 13),
                (1, 12, 0, 12, 13), (1, 10, 0, 10, 10), (0, 9, 0, 13, 10),
                (1, 8, 0, 10, 9), (1, 10, 0, 13, 10)],
            24 : [
                (0, 15, 0, 17, 15), (2, 15, 0, 17, 16), (1, 16, 0, 17, 17),
                (2, 16, 0, 17, 17), (1, 13, 0, 13, 13), (1, 13, 0, 17, 14),
                (1, 11, 0, 13, 12), (1, 13, 0, 17, 14)],
            32 : [
                (0, 21, 0, 22, 21), (3, 19, 0, 22, 21), (2, 22, 0, 22, 23),
                (3, 21, 0, 22, 23), (1, 18, 0, 17, 17), (1, 16, 0, 23, 17),
                (1, 15, 0, 17, 16), (1, 16, 0, 23, 17)],
            48 : [
                (0, 32, 0, 33, 32), (4, 29, 0, 33, 32), (2, 33, 0, 33, 35),
                (4, 32, 0, 33, 35), (2, 27, 0, 26, 27), (2, 25, 0, 35, 27),
                (2, 22, 0, 26, 24), (2, 25, 0, 35, 27)],
        }

        f = self._TEST_FONTS['fixed']
        for (ptsize, test_data) in TEST_VALUES_FIXED.items():
            self.assertEqual(f.get_metrics('ABCDabcd', ptsize=ptsize), test_data)

        f = self._TEST_FONTS['sans']
        for (ptsize, test_data) in TEST_VALUES_SANS.items():
            self.assertEqual(f.get_metrics('ABCDabcd', ptsize=ptsize), test_data)

        metrics = f.get_metrics('ABCD', ptsize=24, bbmode=ft_const.BBOX_EXACT)
        self.assertEqual(len(metrics), len('ABCD'))
        self.assertTrue(isinstance(metrics[0], tuple))

        for m in metrics[0]:
            self.assertTrue(isinstance(m, float))


    def todo_test_pygame2_freetype_Font_get_size(self):

        # __doc__ (as of 2009-05-25) for pygame2.freetype.Font.get_size:

        # get_size(pt, text) -> int, int
        # 
        # Gets the size which 'text' will occupy when rendered using
        # this Font.

        self.fail() 

    def test_pygame2_freetype_Font_height(self):

        # __doc__ (as of 2009-05-25) for pygame2.freetype.Font.height:

        # Gets the height of the Font.

        f = self._TEST_FONTS['sans']
        self.assertEqual(f.height, 2355)

        f = self._TEST_FONTS['fixed']
        self.assertEqual(f.height, 1100)
        

    def test_pygame2_freetype_Font_name(self):

        # __doc__ (as of 2009-05-25) for pygame2.freetype.Font.name:

        # Gets the name of the font face.
        
        f = self._TEST_FONTS['sans']
        self.assertEqual(f.name, 'Liberation Sans')

        f = self._TEST_FONTS['fixed']
        self.assertEqual(f.name, 'Inconsolata')


    def todo_test_pygame2_freetype_Font_render(self):

        # __doc__ (as of 2009-05-25) for pygame2.freetype.Font.render:

        # render(text, fgcolor[, bgcolor, renderflag]) -> Surface
        # 
        # Renders a text to a Surface.
        # 
        # *TODO*

        self.fail() 

    def todo_test_pygame2_freetype_Font_style(self):

        # __doc__ (as of 2009-05-25) for pygame2.freetype.Font.style:

        # Gets or sets the style of the font. *TODO*

        self.fail()

if __name__ == '__main__':
    unittest.main()
