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

        f = self._TEST_FONTS['sans']
        self.assertFalse(f.fixed_width)

        f = self._TEST_FONTS['fixed']
        self.assertFalse(f.fixed_width)

        # TODO: Find a real fixed width font to test with

    def test_pygame2_freetype_Font_get_metrics(self):

        font = self._TEST_FONTS['sans']
        
        # test for floating point values (BBOX_EXACT)
        metrics = font.get_metrics('ABCD', ptsize=24, bbmode=ft_const.BBOX_EXACT)
        self.assertEqual(len(metrics), len('ABCD'))
        self.assertTrue(isinstance(metrics, list))

        for metrics_tuple in metrics:
            self.assertTrue(isinstance(metrics_tuple, tuple))
            self.assertEqual(len(metrics_tuple), 5)
            for m in metrics_tuple:
                self.assertTrue(isinstance(m, float))

        # test for integer values (BBOX_PIXEL)
        metrics = font.get_metrics('foobar', ptsize=24, bbmode=ft_const.BBOX_PIXEL)
        self.assertEqual(len(metrics), len('foobar'))
        self.assertTrue(isinstance(metrics, list))

        for metrics_tuple in metrics:
            self.assertTrue(isinstance(metrics_tuple, tuple))
            self.assertEqual(len(metrics_tuple), 5)
            for m in metrics_tuple:
                self.assertTrue(isinstance(m, int))

        # test for empty string
        metrics = font.get_metrics('', ptsize=24)
        self.assertEqual(metrics, [])
        metrics = font.get_metrics('', ptsize=24, bbmode=ft_const.BBOX_EXACT)
        self.assertEqual(metrics, [])
        metrics = font.get_metrics('', ptsize=24, bbmode=ft_const.BBOX_PIXEL)
        self.assertEqual(metrics, [])

        # test for invalid string
        self.assertRaises(TypeError, font.get_metrics, 24, 24)

    def test_pygame2_freetype_Font_get_size(self):

        font = self._TEST_FONTS['sans']

        def test_size(s):
            self.assertTrue(isinstance(s, tuple))
            self.assertEqual(len(s), 2)
            self.assertTrue(isinstance(s[0], int))
            self.assertTrue(isinstance(s[1], int))

        size_default = font.get_size("ABCDabcd", ptsize=24)
        test_size(size_default)
        self.assertTrue(size_default > (0, 0), size_default)
        self.assertTrue(size_default[0] > size_default[1])

        size_bigger = font.get_size("ABCDabcd", ptsize=32)
        test_size(size_bigger)
        self.assertTrue(size_bigger > size_default)

        size_bolden = font.get_size("ABCDabcd", ptsize=24, style=ft_const.STYLE_BOLD)
        test_size(size_bolden)
        self.assertTrue(size_bolden > size_default)

        font.vertical = True
        size_vert = font.get_size("ABCDabcd", ptsize=24)
        test_size(size_vert)
        self.assertTrue(size_vert[0] < size_vert[1])
        font.vertical = False

        # TODO: Slanted text is slightly wider!
        size_italic = font.get_size("ABCDabcd", ptsize=24, style=ft_const.STYLE_ITALIC)
        test_size(size_italic)
        self.assertTrue(size_italic[0] > size_default[0])
        self.assertTrue(size_italic[1] == size_default[1])

        # TODO: Text size must consider the underline!
        size_under = font.get_size("ABCDabcd", ptsize=24, style=ft_const.STYLE_UNDERLINE)
        test_size(size_under)
        self.assertTrue(size_under[0] == size_default[0])
        self.assertTrue(size_under[1] > size_default[1])

        # Empty strings.
        size_null = font.get_size("", ptsize=24, style=ft_const.STYLE_UNDERLINE)
        test_size (size_null)
        self.assertTrue (size_null[0] == size_null[1] == 0)
        size_null = font.get_size("", ptsize=24, style=ft_const.STYLE_BOLD)
        test_size (size_null)
        self.assertTrue (size_null[0] == size_null[1] == 0)
        size_null = font.get_size("", ptsize=24, style=ft_const.STYLE_ITALIC)
        test_size (size_null)
        self.assertTrue (size_null[0] == size_null[1] == 0)
        size_null = font.get_size("", ptsize=24)
        test_size (size_null)
        self.assertTrue (size_null[0] == size_null[1] == 0)
        
    def test_pygame2_freetype_Font_height(self):

        f = self._TEST_FONTS['sans']
        self.assertEqual(f.height, 2355)

        f = self._TEST_FONTS['fixed']
        self.assertEqual(f.height, 1100)
        

    def test_pygame2_freetype_Font_name(self):

        f = self._TEST_FONTS['sans']
        self.assertEqual(f.name, 'Liberation Sans')

        f = self._TEST_FONTS['fixed']
        self.assertEqual(f.name, 'Inconsolata')


    def test_pygame2_freetype_Font_render(self):

        font = self._TEST_FONTS['sans']

        pygame2.sdl.video.init()
        surf = pygame2.sdl.video.Surface(800, 600)
        color = pygame2.base.Color(0, 0, 0)

        # make sure we always have a valid fg color
        self.assertRaises(TypeError, font.render, None, 'FoobarBaz')
        self.assertRaises(TypeError, font.render, None, 'FoobarBaz', None)

        # render to new surface
        rend = font.render(None, 'FoobarBaz', pygame2.base.Color(0, 0, 0), None, ptsize=24)
        self.assertTrue(isinstance(rend, tuple))
        self.assertTrue(isinstance(rend[0], int))
        self.assertTrue(isinstance(rend[1], int))
        self.assertTrue(isinstance(rend[2], pygame2.base.Surface))

        # render to existing surface
        rend = font.render((surf, 32, 32), 'FoobarBaz', color, None, ptsize=24)
        self.assertTrue(isinstance(rend, tuple))
        self.assertTrue(isinstance(rend[0], int))
        self.assertTrue(isinstance(rend[1], int))

        # render empty to new surface
        rend = font.render(None, '', pygame2.base.Color(0, 0, 0), None, ptsize=24)
        self.assertTrue(isinstance(rend, tuple))
        self.assertTrue(isinstance(rend[0], int) and rend[0] == 0)
        self.assertTrue(isinstance(rend[1], int) and rend[1] == 0)
        self.assertTrue(isinstance(rend[2], pygame2.base.Surface))
        self.assertTrue(rend[2].size == (0, 0))
        
        # render empty to existing surface
        rend = font.render((surf, 32, 32), '', color, None, ptsize=24)
        self.assertTrue(isinstance(rend, tuple))
        self.assertTrue(isinstance(rend[0], int) and rend[0] == 0)
        self.assertTrue(isinstance(rend[1], int) and rend[1] == 0)
        
        # misc parameter test
        self.assertRaises(ValueError, font.render, None, 'foobar', color)
        self.assertRaises(TypeError, font.render, None, 'foobar', color, "",
                ptsize=24)
        self.assertRaises(ValueError, font.render, None, 'foobar', color, None,
                style=42, ptsize=24)
        self.assertRaises(TypeError, font.render, None, 'foobar', color, None,
                style=None, ptsize=24)
        self.assertRaises(ValueError, font.render, None, 'foobar', color, None,
                style=97, ptsize=24)

        pygame2.sdl.video.quit()



    def test_pygame2_freetype_Font_style(self):

        font = self._TEST_FONTS['sans']

        # make sure STYLE_NORMAL is the default value
        self.assertEqual(ft_const.STYLE_NORMAL, font.style)

        # make sure we check for style type
        try:    font.style = "None"
        except TypeError: pass
        else:   self.fail("Failed style assignement")

        try:    font.style = None
        except TypeError: pass
        else:   self.fail("Failed style assignement")

        # make sure we only accept valid constants
        try:    font.style = 112
        except ValueError: pass
        else:   self.fail("Failed style assignement")

        # make assure no assignements happened
        self.assertEqual(ft_const.STYLE_NORMAL, font.style)

        # test assignement
        font.style = ft_const.STYLE_UNDERLINE
        self.assertEqual(ft_const.STYLE_UNDERLINE, font.style)

        # test complex styles
        st = (  ft_const.STYLE_BOLD | ft_const.STYLE_UNDERLINE |
                ft_const.STYLE_ITALIC )

        font.style = st
        self.assertEqual(st, font.style)

        # revert changes
        font.style = ft_const.STYLE_NORMAL
        self.assertEqual(ft_const.STYLE_NORMAL, font.style)


if __name__ == '__main__':
    unittest.main()
