import sys
import os
import unittest

import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
from pygame.compat import as_unicode, as_bytes, xrange_, filesystem_errors
from pygame.compat import PY_MAJOR_VERSION

UCS_4 = sys.maxunicode > 0xFFFF

def equal_images(s1, s2):
    size = s1.get_size()
    if s2.get_size() != size:
        return False
    w, h = size
    for x in xrange_(w):
        for y in xrange_(h):
            if s1.get_at((x, y)) != s2.get_at((x, y)):
                return False
    return True


import platform
IS_PYPY = 'PyPy' == platform.python_implementation()

@unittest.skipIf(IS_PYPY, 'pypy skip known failure') # TODO
class FontModuleTest( unittest.TestCase ):
    def setUp(self):
        pygame_font.init()

    def tearDown(self):
        pygame_font.quit()

    def test_SysFont(self):
        # Can only check that a font object is returned.
        fonts = pygame_font.get_fonts()
        o = pygame_font.SysFont(fonts[0], 20)
        self.failUnless(isinstance(o, pygame_font.FontType))
        o = pygame_font.SysFont(fonts[0], 20, italic=True)
        self.failUnless(isinstance(o, pygame_font.FontType))
        o = pygame_font.SysFont(fonts[0], 20, bold=True)
        self.failUnless(isinstance(o, pygame_font.FontType))
        o = pygame_font.SysFont('thisisnotafont', 20)
        self.failUnless(isinstance(o, pygame_font.FontType))

    def test_get_default_font(self):
        self.assertEqual(pygame_font.get_default_font(), 'freesansbold.ttf')

    def test_get_fonts_returns_something(self):
        fnts = pygame_font.get_fonts()
        self.failUnless(fnts)


    # to test if some files exist...
    #def XXtest_has_file_osx_10_5_sdk(self):
    #    import os
    #    f = "/Developer/SDKs/MacOSX10.5.sdk/usr/X11/include/ft2build.h"
    #    self.assertEqual(os.path.exists(f), True)

    #def XXtest_has_file_osx_10_4_sdk(self):
    #    import os
    #    f = "/Developer/SDKs/MacOSX10.4u.sdk/usr/X11R6/include/ft2build.h"
    #    self.assertEqual(os.path.exists(f), True)





    def test_get_fonts(self):
        fnts = pygame_font.get_fonts()

        if not fnts:
            raise Exception(repr(fnts))

        self.failUnless(fnts)

        if (PY_MAJOR_VERSION >= 3):
            # For Python 3.x, names will always be unicode strings.
            name_types = (str,)
        else:
            # For Python 2.x, names may be either unicode or ascii strings.
            name_types = (str, unicode)

        for name in fnts:
            # note, on ubuntu 2.6 they are all unicode strings.

            self.failUnless(isinstance(name, name_types), name)
            self.failUnless(name.islower(), name)
            self.failUnless(name.isalnum(), name)

    def test_get_init(self):
        self.failUnless(pygame_font.get_init())
        pygame_font.quit()
        self.assertFalse(pygame_font.get_init())

    def test_init(self):
        pygame_font.init()

    def test_match_font_all_exist(self):
        fonts = pygame_font.get_fonts()

        # Ensure all listed fonts are in fact available, and the returned file
        # name is a full path.
        for font in fonts:
            path = pygame_font.match_font(font)
            self.assertFalse(path is None)
            self.failUnless(os.path.isabs(path))

    def test_match_font_bold(self):

        fonts = pygame_font.get_fonts()

        # Look for a bold font.
        for font in fonts:
            if pygame_font.match_font(font, bold=True) is not None:
                break
        else:
            self.fail()

    def test_match_font_italic(self):

        fonts = pygame_font.get_fonts()

        # Look for an italic font.
        for font in fonts:
            if pygame_font.match_font(font, italic=True) is not None:
                break
        else:
            self.fail()


    def test_match_font_comma_separated(self):

        fonts = pygame_font.get_fonts()

        # Check for not found.
        self.failUnless(pygame_font.match_font('thisisnotafont') is None)

        # Check comma separated list.
        names = ','.join(['thisisnotafont', fonts[-1], 'anothernonfont'])
        self.assertFalse(pygame_font.match_font(names) is None)
        names = ','.join(['thisisnotafont1', 'thisisnotafont2', 'thisisnotafont3'])
        self.failUnless(pygame_font.match_font(names) is None)



    def test_quit(self):
        pygame_font.quit()





@unittest.skipIf(IS_PYPY, 'pypy skip known failure') # TODO
class FontTest(unittest.TestCase):
    def setUp(self):
        pygame_font.init()

    def tearDown(self):
        pygame_font.quit()


    def test_render_args(self):
        screen = pygame.display.set_mode((600, 400))
        rect = screen.get_rect()
        f = pygame_font.Font(None, 20)
        screen.fill((10, 10, 10))
        font_surface = f.render("   bar", True, (0, 0, 0), (255, 255, 255))
        font_rect = font_surface.get_rect()
        font_rect.topleft = rect.topleft
        self.assertTrue(font_surface)
        screen.blit(font_surface, font_rect, font_rect)
        pygame.display.update()
        self.assertEqual(tuple(screen.get_at((0,0)))[:3], (255, 255, 255))
        self.assertEqual(tuple(screen.get_at(font_rect.topleft))[:3], (255, 255, 255))

        # If we don't have a real display, don't do this test.
        # Transparent background doesn't seem to work without a read video card.
        if os.environ.get('SDL_VIDEODRIVER') != 'dummy':
            screen.fill((10, 10, 10))
            font_surface = f.render("   bar", True, (0, 0, 0), None)
            font_rect = font_surface.get_rect()
            font_rect.topleft = rect.topleft
            self.assertTrue(font_surface)
            screen.blit(font_surface, font_rect, font_rect)
            pygame.display.update()
            self.assertEqual(tuple(screen.get_at((0,0)))[:3], (10, 10, 10))
            self.assertEqual(tuple(screen.get_at(font_rect.topleft))[:3], (10, 10, 10))

            screen.fill((10, 10, 10))
            font_surface = f.render("   bar", True, (0, 0, 0))
            font_rect = font_surface.get_rect()
            font_rect.topleft = rect.topleft
            self.assertTrue(font_surface)
            screen.blit(font_surface, font_rect, font_rect)
            pygame.display.update(rect)
            self.assertEqual(tuple(screen.get_at((0,0)))[:3], (10, 10, 10))
            self.assertEqual(tuple(screen.get_at(font_rect.topleft))[:3], (10, 10, 10))







@unittest.skipIf(IS_PYPY, 'pypy skip known failure') # TODO
class FontTypeTest( unittest.TestCase ):
    def setUp(self):
        pygame_font.init()

    def tearDown(self):
        pygame_font.quit()

    def test_get_ascent(self):
        # Ckecking ascent would need a custom test font to do properly.
        f = pygame_font.Font(None, 20)
        ascent = f.get_ascent()
        self.failUnless(isinstance(ascent, int))
        self.failUnless(ascent > 0)
        s = f.render("X", False, (255, 255, 255))
        self.failUnless(s.get_size()[1] > ascent)

    def test_get_descent(self):
        # Ckecking descent would need a custom test font to do properly.
        f = pygame_font.Font(None, 20)
        descent = f.get_descent()
        self.failUnless(isinstance(descent, int))
        self.failUnless(descent < 0)

    def test_get_height(self):
        # Ckecking height would need a custom test font to do properly.
        f = pygame_font.Font(None, 20)
        height = f.get_height()
        self.failUnless(isinstance(height, int))
        self.failUnless(height > 0)
        s = f.render("X", False, (255, 255, 255))
        self.failUnless(s.get_size()[1] == height)

    def test_get_linesize(self):
        # Ckecking linesize would need a custom test font to do properly.
        # Questions: How do linesize, height and descent relate?
        f = pygame_font.Font(None, 20)
        linesize = f.get_linesize()
        self.failUnless(isinstance(linesize, int))
        self.failUnless(linesize > 0)

    def test_metrics(self):
        # Ensure bytes decoding works correctly. Can only compare results
        # with unicode for now.
        f = pygame_font.Font(None, 20)
        um = f.metrics(as_unicode("."))
        bm = f.metrics(as_bytes("."))
        self.assert_(len(um) == 1)
        self.assert_(len(bm) == 1)
        self.assert_(um[0] is not None)
        self.assert_(um == bm)
        u = as_unicode(r"\u212A")
        b = u.encode("UTF-16")[2:] # Keep byte order consistent. [2:] skips BOM
        bm = f.metrics(b)
        self.assert_(len(bm) == 2)
        try:
            um = f.metrics(u)
        except pygame.error:
            pass
        else:
            self.assert_(len(um) == 1)
            self.assert_(bm[0] != um[0])
            self.assert_(bm[1] != um[0])

        if UCS_4:
            u = as_unicode(r"\U00013000")
            bm = f.metrics(u)
            self.assert_(len(bm) == 1 and bm[0] is None)

        return # unfinished
        # The documentation is useless here. How large a list?
        # How do list positions relate to character codes?
        # What about unicode characters?

        # __doc__ (as of 2008-08-02) for pygame_font.Font.metrics:

          # Font.metrics(text): return list
          # Gets the metrics for each character in the pased string.
          #
          # The list contains tuples for each character, which contain the
          # minimum X offset, the maximum X offset, the minimum Y offset, the
          # maximum Y offset and the advance offset (bearing plus width) of the
          # character. [(minx, maxx, miny, maxy, advance), (minx, maxx, miny,
          # maxy, advance), ...]

        self.fail()

    def test_render(self):
        """
        """

        f = pygame_font.Font(None, 20)
        s = f.render("foo", True, [0, 0, 0], [255, 255, 255])
        s = f.render("xxx", True, [0, 0, 0], [255, 255, 255])
        s = f.render("", True, [0, 0, 0], [255, 255, 255])
        s = f.render("foo", False, [0, 0, 0], [255, 255, 255])
        s = f.render("xxx", False, [0, 0, 0], [255, 255, 255])
        s = f.render("xxx", False, [0, 0, 0])
        s = f.render("   ", False, [0, 0, 0])
        s = f.render("   ", False, [0, 0, 0], [255, 255, 255])
        # null text should be 1 pixel wide.
        s = f.render("", False, [0, 0, 0], [255, 255, 255])
        self.assertEqual(s.get_size()[0], 1)
        # None text should be 1 pixel wide.
        s = f.render(None, False, [0, 0, 0], [255, 255, 255])
        self.assertEqual(s.get_size()[0], 1)
        # Non-text should raise a TypeError.
        self.assertRaises(TypeError, f.render,
                          [], False, [0, 0, 0], [255, 255, 255])
        self.assertRaises(TypeError, f.render,
                          1, False, [0, 0, 0], [255, 255, 255])
        # is background transparent for antialiasing?
        s = f.render(".", True, [255, 255, 255])
        self.assertEqual(s.get_at((0, 0))[3], 0)
        # is Unicode and bytes encoding correct?
        # Cannot really test if the correct characters are rendered, but
        # at least can assert the encodings differ.
        su = f.render(as_unicode("."), False, [0, 0, 0], [255, 255, 255])
        sb = f.render(as_bytes("."), False, [0, 0, 0], [255, 255, 255])
        self.assert_(equal_images(su, sb))
        u = as_unicode(r"\u212A")
        b = u.encode("UTF-16")[2:] # Keep byte order consistent. [2:] skips BOM
        sb = f.render(b, False, [0, 0, 0], [255, 255, 255])
        try:
            su = f.render(u, False, [0, 0, 0], [255, 255, 255])
        except pygame.error:
            pass
        else:
            self.assert_(not equal_images(su, sb))

        # If the font module is SDL_ttf based, then it can only supports  UCS-2;
        # it will raise an exception for an out-of-range UCS-4 code point.
        if UCS_4 and not hasattr(f, 'ucs4'):
            ucs_2 = as_unicode(r"\uFFEE")
            s = f.render(ucs_2, False, [0, 0, 0], [255, 255, 255])
            ucs_4 = as_unicode(r"\U00010000")
            self.assertRaises(UnicodeError, f.render,
                              ucs_4, False, [0, 0, 0], [255, 255, 255])

        b = as_bytes("ab\x00cd")
        self.assertRaises(ValueError, f.render, b, 0, [0, 0, 0])
        u = as_unicode("ab\x00cd")
        self.assertRaises(ValueError, f.render, b, 0, [0, 0, 0])

        # __doc__ (as of 2008-08-02) for pygame_font.Font.render:

          # Font.render(text, antialias, color, background=None): return Surface
          # draw text on a new Surface
          #
          # This creates a new Surface with the specified text rendered on it.
          # Pygame provides no way to directly draw text on an existing Surface:
          # instead you must use Font.render() to create an image (Surface) of
          # the text, then blit this image onto another Surface.
          #
          # The text can only be a single line: newline characters are not
          # rendered. The antialias argument is a boolean: if true the
          # characters will have smooth edges. The color argument is the color
          # of the text [e.g.: (0,0,255) for blue]. The optional background
          # argument is a color to use for the text background. If no background
          # is passed the area outside the text will be transparent.
          #
          # The Surface returned will be of the dimensions required to hold the
          # text. (the same as those returned by Font.size()). If an empty
          # string is passed for the text, a blank surface will be returned that
          # is one pixel wide and the height of the font.
          #
          # Depending on the type of background and antialiasing used, this
          # returns different types of Surfaces. For performance reasons, it is
          # good to know what type of image will be used. If antialiasing is not
          # used, the return image will always be an 8bit image with a two color
          # palette. If the background is transparent a colorkey will be set.
          # Antialiased images are rendered to 24-bit RGB images. If the
          # background is transparent a pixel alpha will be included.
          #
          # Optimization: if you know that the final destination for the text
          # (on the screen) will always have a solid background, and the text is
          # antialiased, you can improve performance by specifying the
          # background color. This will cause the resulting image to maintain
          # transparency information by colorkey rather than (much less
          # efficient) alpha values.
          #
          # If you render '\n' a unknown char will be rendered.  Usually a
          # rectangle. Instead you need to handle new lines yourself.
          #
          # Font rendering is not thread safe: only a single thread can render
          # text any time.


    def test_set_bold(self):
        f = pygame_font.Font(None, 20)
        self.assertFalse(f.get_bold())
        f.set_bold(True)
        self.failUnless(f.get_bold())
        f.set_bold(False)
        self.assertFalse(f.get_bold())

    def test_set_italic(self):
        f = pygame_font.Font(None, 20)
        self.assertFalse(f.get_italic())
        f.set_italic(True)
        self.failUnless(f.get_italic())
        f.set_italic(False)
        self.assertFalse(f.get_italic())

    def test_set_underline(self):
        f = pygame_font.Font(None, 20)
        self.assertFalse(f.get_underline())
        f.set_underline(True)
        self.failUnless(f.get_underline())
        f.set_underline(False)
        self.assertFalse(f.get_underline())

    def test_size(self):
        f = pygame_font.Font(None, 20)
        text = as_unicode("Xg")
        size = f.size(text)
        w, h = size
        self.assert_(isinstance(w, int) and isinstance(h, int))
        s = f.render(text, False, (255, 255, 255))
        self.assert_(size == s.get_size())
        btext = text.encode("ascii")
        self.assert_(f.size(btext) == size)
        text = as_unicode(r"\u212A")
        btext = text.encode("UTF-16")[2:] # Keep the byte order consistent.
        bsize = f.size(btext)
        try:
            size = f.size(text)
        except pygame.error:
            pass
        else:
            self.assert_(size != bsize)

    def test_font_file_not_found(self):
        # A per BUG reported by Bo Jangeborg on pygame-user mailing list,
        # http://www.mail-archive.com/pygame-users@seul.org/msg11675.html

        pygame_font.init()
        self.assertRaises(IOError,
                          pygame_font.Font,
                          'some-fictional-font.ttf', 20)

    def test_load_from_file(self):
        font_name = pygame_font.get_default_font()
        font_path = os.path.join(os.path.split(pygame.__file__)[0],
                                 pygame_font.get_default_font())
        f = pygame_font.Font(font_path, 20)

    def test_load_from_file_obj(self):
        font_name = pygame_font.get_default_font()
        font_path = os.path.join(os.path.split(pygame.__file__)[0],
                                 pygame_font.get_default_font())
        f = open(font_path, "rb")
        font = pygame_font.Font(f, 20)

    def test_load_default_font_filename(self):
        # In font_init, a special case is when the filename argument is
        # identical to the default font file name.
        f = pygame_font.Font(pygame_font.get_default_font(), 20)

    def test_load_from_file_unicode(self):
        base_dir = os.path.dirname(pygame.__file__)
        font_path = os.path.join(base_dir, pygame_font.get_default_font())
        if os.path.sep == '\\':
            font_path = font_path.replace('\\', '\\\\')
        ufont_path = as_unicode(font_path)
        f = pygame_font.Font(ufont_path, 20)

    def test_load_from_file_bytes(self):
        font_path = os.path.join(os.path.split(pygame.__file__)[0],
                                 pygame_font.get_default_font())
        filesystem_encoding = sys.getfilesystemencoding()
        try:
            font_path = font_path.decode(filesystem_encoding,
                                         filesystem_errors)
        except AttributeError:
            pass
        bfont_path = font_path.encode(filesystem_encoding,
                                      filesystem_errors)
        f = pygame_font.Font(bfont_path, 20)


@unittest.skipIf(IS_PYPY, 'pypy skip known failure') # TODO
class VisualTests( unittest.TestCase ):
    __tags__ = ['interactive']

    screen = None
    aborted = False

    def setUp(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 200))
            self.screen.fill((255, 255, 255))
            pygame.display.flip()
            self.f = pygame_font.Font(None, 32)

    def abort(self):
        if self.screen is not None:
            pygame.quit()
        self.aborted = True

    def query(self,
              bold=False, italic=False, underline=False, antialiase=False):
        if self.aborted:
            return False
        spacing = 10
        offset = 20
        y = spacing
        f = self.f
        screen = self.screen
        screen.fill((255, 255, 255))
        pygame.display.flip()
        if not (bold or italic or underline or antialiase):
            text = "normal"
        else:
            modes = []
            if bold:
                modes.append("bold")
            if italic:
                modes.append("italic")
            if underline:
                modes.append("underlined")
            if antialiase:
                modes.append("antialiased")
            text = "%s (y/n):" % ('-'.join(modes),)
        f.set_bold(bold)
        f.set_italic(italic)
        f.set_underline(underline)
        s = f.render(text, antialiase, (0, 0, 0))
        screen.blit(s, (offset, y))
        y += s.get_size()[1] + spacing
        f.set_bold(False)
        f.set_italic(False)
        f.set_underline(False)
        s = f.render("(some comparison text)", False, (0, 0, 0))
        screen.blit(s, (offset, y))
        pygame.display.flip()
        while 1:
            for evt in pygame.event.get():
                if evt.type == pygame.KEYDOWN:
                    if evt.key == pygame.K_ESCAPE:
                        self.abort()
                        return False
                    if evt.key == pygame.K_y:
                        return True
                    if evt.key == pygame.K_n:
                        return False
                if evt.type == pygame.QUIT:
                    self.abort()
                    return False

    def test_bold(self):
        self.failUnless(self.query(bold=True))

    def test_italic(self):
        self.failUnless(self.query(italic=True))

    def test_underline(self):
        self.failUnless(self.query(underline=True))

    def test_antialiase(self):
        self.failUnless(self.query(antialiase=True))

    def test_bold_antialiase(self):
        self.failUnless(self.query(bold=True, antialiase=True))

    def test_italic_underline(self):
        self.failUnless(self.query(italic=True, underline=True))


if __name__ == '__main__':
    unittest.main()
