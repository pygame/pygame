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


class FontModuleTest( unittest.TestCase ):
    def setUp(self):
        pygame.font.init()

    def tearDown(self):
        pygame.font.quit()

    def test_SysFont(self):
        # Can only check that a font object is returned.
        fonts = pygame.font.get_fonts()
        o = pygame.font.SysFont(fonts[0], 20)
        self.failUnless(isinstance(o, pygame.font.FontType))
        o = pygame.font.SysFont(fonts[0], 20, italic=True)
        self.failUnless(isinstance(o, pygame.font.FontType))
        o = pygame.font.SysFont(fonts[0], 20, bold=True)
        self.failUnless(isinstance(o, pygame.font.FontType))
        o = pygame.font.SysFont('thisisnotafont', 20)
        self.failUnless(isinstance(o, pygame.font.FontType))

    def test_get_default_font(self):
        self.failUnlessEqual(pygame.font.get_default_font(), 'freesansbold.ttf')

    def test_get_fonts(self):
        fnts = pygame.font.get_fonts()
        self.failUnless(fnts)
        for name in fnts:
            self.failUnless(isinstance(name, str))
            self.failUnless(name.islower(), name)
            self.failUnless(name.isalnum(), name)

    def test_get_init(self):
        self.failUnless(pygame.font.get_init())
        pygame.font.quit()
        self.failIf(pygame.font.get_init())

    def test_init(self):
        pygame.font.init()

    def test_match_font(self):
        fonts = pygame.font.get_fonts()

        # Ensure all listed fonts are in fact available, and the returned file
        # name is a full path.
        for font in fonts:
            path = pygame.font.match_font(font)
            self.failIf(path is None)
            self.failUnless(os.path.isabs(path))

        # Look for a bold font.
        for font in fonts:
            if pygame.font.match_font(font, bold=True) is not None:
                break
        else:
            self.fail()

        # Look for an italic font.
        for font in fonts:
            if pygame.font.match_font(font, italic=True) is not None:
                break
        else:
            self.fail()

        # Check for not found.
        self.failUnless(pygame.font.match_font('thisisnotafont') is None)

        # Check comma separated list.
        names = ','.join(['thisisnotafont', fonts[-1], 'anothernonfont'])
        self.failIf(pygame.font.match_font(names) is None)
        names = ','.join(['thisisnotafont1', 'thisisnotafont2', 'thisisnotafont3'])
        self.failUnless(pygame.font.match_font(names) is None)

    def test_quit(self):
        pygame.font.quit()


class FontTypeTest( unittest.TestCase ):
    def todo_test_get_ascent(self):

        # __doc__ (as of 2008-08-02) for pygame.font.Font.get_ascent:

          # Font.get_ascent(): return int
          # get the ascent of the font
          # 
          # Return the height in pixels for the font ascent. The ascent is the
          # number of pixels from the font baseline to the top of the font.
          # 

        self.fail() 

    def todo_test_get_descent(self):

        # __doc__ (as of 2008-08-02) for pygame.font.Font.get_descent:

          # Font.get_descent(): return int
          # get the descent of the font
          # 
          # Return the height in pixels for the font descent. The descent is the
          # number of pixels from the font baseline to the bottom of the font.

        self.fail() 

    def todo_test_get_height(self):

        # __doc__ (as of 2008-08-02) for pygame.font.Font.get_height:

          # Font.get_height(): return int
          # get the height of the font
          # 
          # Return the height in pixels of the actual rendered text. This is the
          # average size for each glyph in the font.

        self.fail() 

    def todo_test_get_linesize(self):

        # __doc__ (as of 2008-08-02) for pygame.font.Font.get_linesize:

          # Font.get_linesize(): return int
          # get the line space of the font text
          # 
          # Return the height in pixels for a line of text with the font. When
          # rendering multiple lines of text this is the recommended amount of
          # space between lines.

        self.fail() 

    def todo_test_metrics(self):

        # __doc__ (as of 2008-08-02) for pygame.font.Font.metrics:

          # Font.metrics(text): return list
          # Gets the metrics for each character in the pased string.
          # 
          # The list contains tuples for each character, which contain the
          # minimum X offset, the maximum X offset, the minimum Y offset, the
          # maximum Y offset and the advance offset (bearing plus width) of the
          # character. [(minx, maxx, miny, maxy, advance), (minx, maxx, miny,
          # maxy, advance), ...]

        self.fail() 

    def todo_test_render(self):
        """ 
        """

        f = pygame.font.Font(None, 20)
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

        # __doc__ (as of 2008-08-02) for pygame.font.Font.render:

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
        f = pygame.font.Font(None, 20)
        self.failIf(f.get_bold())
        f.set_bold(True)
        self.failUnless(f.get_bold())
        f.set_bold(False)
        self.failIf(f.get_bold())

    def test_set_italic(self):
        f = pygame.font.Font(None, 20)
        self.failIf(f.get_italic())
        f.set_italic(True)
        self.failUnless(f.get_italic())
        f.set_italic(False)
        self.failIf(f.get_bold())

    def test_set_underline(self):
        f = pygame.font.Font(None, 20)
        self.failIf(f.get_underline())
        f.set_underline(True)
        self.failUnless(f.get_underline())
        f.set_underline(False)
        self.failIf(f.get_underline())

    def todo_test_size(self):

        # __doc__ (as of 2008-08-02) for pygame.font.Font.size:

          # Font.size(text): return (width, height)
          # determine the amount of space needed to render text
          # 
          # Returns the dimensions needed to render the text. This can be used
          # to help determine the positioning needed for text before it is
          # rendered. It can also be used for wordwrapping and other layout
          # effects.
          # 
          # Be aware that most fonts use kerning which adjusts the widths for
          # specific letter pairs. For example, the width for "ae" will not
          # always match the width for "a" + "e".

        self.fail() 

    def test_font_file_not_found(self):
        # A per BUG reported by Bo Jangeborg on pygame-user mailing list,
        # http://www.mail-archive.com/pygame-users@seul.org/msg11675.html

        pygame.font.init()

        def fetch():
            font = pygame.font.Font('some-fictional-font.ttf', 20)
        self.failUnlessRaises(IOError, fetch)
        

if __name__ == '__main__':
    unittest.main()
