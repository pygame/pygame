import sys
import os
if __name__ == '__main__':
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

if is_pygame_pkg:
    from pygame.tests.test_utils import test_not_implemented, unittest, \
                                        geterror
else:
    from test.test_utils import test_not_implemented, unittest, geterror

import pygame
try:
    import pygame.freetype as ft
except ImportError:
    ft = None
from pygame.compat import as_unicode


FONTDIR = os.path.join(os.path.dirname (os.path.abspath (__file__)),
                       'fixtures', 'fonts')

def nullfont():
    """return an uninitialized font instance"""
    return ft.Font.__new__(ft.Font)

class FreeTypeFontTest(unittest.TestCase):

    _fixed_path = os.path.join(FONTDIR, 'test_fixed.otf')
    _sans_path = os.path.join(FONTDIR, 'test_sans.ttf')
    _TEST_FONTS = {}

    def setUp(self):
        ft.init()

        if 'fixed' not in self._TEST_FONTS:
            # Inconsolata is an open-source font designed by Raph Levien
            # Licensed under the Open Font License
            # http://www.levien.com/type/myfonts/inconsolata.html
            self._TEST_FONTS['fixed'] = ft.Font(self._fixed_path)

        if 'sans' not in self._TEST_FONTS:
            # Liberation Sans is an open-source font designed by Steve Matteson
            # Licensed under the GNU GPL
            # https://fedorahosted.org/liberation-fonts/
            self._TEST_FONTS['sans'] = ft.Font(self._sans_path)

    def test_freetype_defaultfont(self):
        font = ft.Font(None)
        self.assertEqual(font.name, "FreeSans")

    def test_freetype_Font_init(self):

        self.assertRaises(RuntimeError, ft.Font, os.path.join (FONTDIR, 'nonexistant.ttf'))

        f = self._TEST_FONTS['sans']
        self.assertTrue(isinstance(f, ft.Font))

        f = self._TEST_FONTS['fixed']
        self.assertTrue(isinstance(f, ft.Font))

        f = ft.Font(None, ptsize=24)
        self.assert_(f.height > 0)
        self.assertRaises(RuntimeError, f.__init__,
                          os.path.join(FONTDIR, 'nonexistant.ttf'))
        self.assertRaises(RuntimeError, f.get_size, 'a', ptsize=24)
        
        # Test attribute preservation during reinitalization
        f = ft.Font(self._sans_path, ptsize=24)
        self.assertEqual(f.name, 'Liberation Sans')
        self.assertFalse(f.fixed_width)
        self.assertTrue(f.antialiased)
        self.assertFalse(f.italic)
        f.antialiased = False
        f.italic = True
        f.__init__(self._fixed_path)
        self.assertEqual(f.name, 'Inconsolata')
        ##self.assertTrue(f.fixed_width)
        self.assertFalse(f.fixed_width)  # need a properly marked Mono font
        self.assertFalse(f.antialiased)
        self.assertTrue(f.italic)
        
    def test_freetype_Font_fixed_width(self):

        f = self._TEST_FONTS['sans']
        self.assertFalse(f.fixed_width)

        f = self._TEST_FONTS['fixed']
        ##self.assertTrue(f.fixed_width)
        self.assertFalse(f.fixed_width)  # need a properly marked Mone font

        self.assertRaises(RuntimeError, lambda : nullfont().fixed_width)

    def test_freetype_Font_get_metrics(self):

        font = self._TEST_FONTS['sans']

        # test for floating point values (BBOX_EXACT)
        metrics = font.get_metrics('ABCD', ptsize=24, bbmode=ft.BBOX_EXACT)
        self.assertEqual(len(metrics), len('ABCD'))
        self.assertTrue(isinstance(metrics, list))

        for metrics_tuple in metrics:
            self.assertTrue(isinstance(metrics_tuple, tuple))
            self.assertEqual(len(metrics_tuple), 5)
            for m in metrics_tuple:
                self.assertTrue(isinstance(m, float))

        # test for integer values (BBOX_PIXEL)
        metrics = font.get_metrics('foobar', ptsize=24, bbmode=ft.BBOX_PIXEL)
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

        # test for invalid string
        self.assertRaises(TypeError, font.get_metrics, 24, 24)

        # raises exception when uninitalized
        self.assertRaises(RuntimeError, nullfont().get_metrics,
                          'a', ptsize=24)

    def test_freetype_Font_get_size(self):

        font = self._TEST_FONTS['sans']

        def test_size(s):
            self.assertTrue(isinstance(s, tuple))
            self.assertEqual(len(s), 2)
            self.assertTrue(isinstance(s[0], int))
            self.assertTrue(isinstance(s[1], int))

        size_default = font.get_size("ABCDabcd", ptsize=24)
        test_size(size_default)
        self.assertTrue(size_default > (0, 0))
        self.assertTrue(size_default[0] > size_default[1])

        size_bigger = font.get_size("ABCDabcd", ptsize=32)
        test_size(size_bigger)
        self.assertTrue(size_bigger > size_default)

        size_bolden = font.get_size("ABCDabcd", ptsize=24, style=ft.STYLE_BOLD)
        test_size(size_bolden)
        self.assertTrue(size_bolden > size_default)

        font.vertical = True
        size_vert = font.get_size("ABCDabcd", ptsize=24)
        test_size(size_vert)
        self.assertTrue(size_vert[0] < size_vert[1])
        font.vertical = False

        size_italic = font.get_size("ABCDabcd", ptsize=24, style=ft.STYLE_ITALIC)
        test_size(size_italic)
        self.assertTrue(size_italic[0] > size_default[0])
        self.assertTrue(size_italic[1] == size_default[1])

        size_under = font.get_size("ABCDabcd", ptsize=24, style=ft.STYLE_UNDERLINE)
        test_size(size_under)
        self.assertTrue(size_under[0] == size_default[0])
        self.assertTrue(size_under[1] > size_default[1])

        size_utf32 = font.get_size(as_unicode(r'\U000130A7'), ptsize=24)
        size_utf16 = font.get_size(as_unicode(r'\uD80C\uDCA7'),
                                   ptsize=24, surrogates=True)
        self.assertEqual(size_utf16[0], size_utf32[0])
        size_utf16 = font.get_size(as_unicode(r'\uD80C\uDCA7'), ptsize=24)
        self.assertNotEqual(size_utf16[0], size_utf32[0]);
        
        self.assertRaises(RuntimeError,
                          nullfont().get_size, 'a', ptsize=24)

    def test_freetype_Font_height(self):

        f = self._TEST_FONTS['sans']
        self.assertEqual(f.height, 2355)

        f = self._TEST_FONTS['fixed']
        self.assertEqual(f.height, 1100)

        self.assertRaises(RuntimeError, lambda : nullfont().height)
        

    def test_freetype_Font_name(self):

        f = self._TEST_FONTS['sans']
        self.assertEqual(f.name, 'Liberation Sans')

        f = self._TEST_FONTS['fixed']
        self.assertEqual(f.name, 'Inconsolata')

        nf = nullfont()
        self.assertEqual(nf.name, repr(nf))

    def test_freetype_Font_render(self):

        font = self._TEST_FONTS['sans']

        surf = pygame.Surface((800, 600))
        color = pygame.Color(0, 0, 0)

        # make sure we always have a valid fg color
        self.assertRaises(TypeError, font.render, None, 'FoobarBaz')
        self.assertRaises(TypeError, font.render, None, 'FoobarBaz', None)

        # render to new surface
        rend = font.render(None, 'FoobarBaz', pygame.Color(0, 0, 0), None, ptsize=24)
        self.assertTrue(isinstance(rend, tuple))
        self.assertEqual(len(rend), 3)
        self.assertTrue(isinstance(rend[0], pygame.Surface))
        self.assertTrue(isinstance(rend[1], int))
        self.assertTrue(isinstance(rend[2], int))

        # render to existing surface
        refcount = sys.getrefcount(surf);
        rend = font.render((surf, 32, 32), 'FoobarBaz', color, None, ptsize=24)
        self.assertTrue(isinstance(rend, tuple))
        self.assertEqual(len(rend), 3)
        self.assertTrue(rend[0] is surf)
        self.assertEqual(sys.getrefcount(surf), refcount + 1)
        self.assertTrue(isinstance(rend[1], int))
        self.assertTrue(isinstance(rend[2], int))

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

        # unicode text (incomplete)
        self.assertRaises(UnicodeEncodeError, font.render,
                          None, as_unicode(r'\uD80C'), color, ptsize=24,
                          surrogates=True)
        self.assertRaises(UnicodeEncodeError, font.render,
                          None, as_unicode(r'\uDCA7'), color, ptsize=24,
                          surrogates=True)
        self.assertRaises(UnicodeEncodeError, font.render,
                          None, as_unicode(r'\uFEFF'), color, ptsize=24,
                          surrogates=True)
        self.assertRaises(UnicodeEncodeError, font.render,
                          None, as_unicode(r'\uFFFE'), color, ptsize=24,
                          surrogates=True)
        rend1 = font.render(None, as_unicode(r'\uD80C\uDCA7'),
                            color, ptsize=24, surrogates=True)
        rend2 = font.render(None, as_unicode(r'\U000130A7'),
                            color, ptsize=24)
        self.assertEqual(rend1[1], rend2[1])
        rend1 = font.render(None, as_unicode(r'\uD80C\uDCA7'),
                            color, ptsize=24)
        self.assertNotEqual(rend1[1], rend2[1])

        # raises exception when uninitalized
        self.assertRaises(RuntimeError, nullfont().render,
                          None, 'a', (0, 0, 0), ptsize=24)

    def test_freetype_Font_style(self):

        font = self._TEST_FONTS['sans']

        # make sure STYLE_NORMAL is the default value
        self.assertEqual(ft.STYLE_NORMAL, font.style)

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
        self.assertEqual(ft.STYLE_NORMAL, font.style)

        # test assignement
        font.style = ft.STYLE_UNDERLINE
        self.assertEqual(ft.STYLE_UNDERLINE, font.style)

        # test complex styles
        st = (  ft.STYLE_BOLD | ft.STYLE_UNDERLINE |
                ft.STYLE_ITALIC )

        font.style = st
        self.assertEqual(st, font.style)

        # revert changes
        font.style = ft.STYLE_NORMAL
        self.assertEqual(ft.STYLE_NORMAL, font.style)


if __name__ == '__main__':
    unittest.main()
