import sys
if __name__ == '__main__':
    import os
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

    def testFontRendering( self ):
        """ 
        """
        #print __file__
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
        #print "fonttest done"
        pygame.font.quit ()

    def todo_test_SysFont(self):

        # __doc__ (as of 2008-08-02) for pygame.font.SysFont:

          # pygame.font.SysFont(name, size, bold=False, italic=False) -> Font
          # create a pygame Font from system font resources
          # 
          # This will search the system fonts for the given font
          # name. You can also enable bold or italic styles, and
          # the appropriate system font will be selected if available.
          # 
          # This will always return a valid Font object, and will
          # fallback on the builtin pygame font if the given font
          # is not found.
          # 
          # Name can also be a comma separated list of names, in
          # which case set of names will be searched in order. Pygame
          # uses a small set of common font aliases, if the specific
          # font you ask for is not available, a reasonable alternative
          # may be used.
          # 
          # Return a new Font object that is loaded from the system fonts. The
          # font will match the requested bold and italic flags. If a suitable
          # system font is not found this will fallback on loading the default
          # pygame font. The font name can be a comma separated list of font
          # names to look for.
          # 

        self.fail() 

    def todo_test_get_default_font(self):

        # __doc__ (as of 2008-08-02) for pygame.font.get_default_font:

          # pygame.font.get_default_font(): return string
          # get the filename of the default font
          # 
          # Return the filename of the system font. This is not the full path to
          # the file. This file can usually be found in the same directory as
          # the font module, but it can also be bundled in separate archives.
          # 

        self.fail() 

    def test_get_fonts(self):
        fnts = pygame.font.get_fonts()
        self.failUnless(fnts)
        for name in fnts:
            self.failUnless(isinstance(name, str))
            self.failUnless(name.islower(), name)
            self.failUnless(name.isalnum(), name)

    def todo_test_get_init(self):
        self.failUnless(pygame.font.get_init())
        pygame.font.quit()
        self.failIf(pygame.font.get_init())

    def test_init(self):
        pygame.font.init()

    def todo_test_match_font(self):

        # __doc__ (as of 2008-08-02) for pygame.font.match_font:

          # pygame.font.match_font(name, bold=0, italic=0) -> name
          # find the filename for the named system font
          # 
          # This performs the same font search as the SysFont()
          # function, only it returns the path to the TTF file
          # that would be loaded. The font name can be a comma
          # separated list of font names to try.
          # 
          # If no match is found, None is returned.
          # 
          # Returns the full path to a font file on the system. If bold or
          # italic are set to true, this will attempt to find the correct family
          # of font.
          # 
          # The font name can actually be a comma separated list of font names
          # to try. If none of the given names are found, None is returned.
          # 
          # Example: 
          #     print pygame.font.match_font('bitstreamverasans')
          #     # output is: /usr/share/fonts/truetype/ttf-bitstream-vera/Vera.ttf
          #     # (but only if you have Vera on your system)

        self.fail() 

    def todo_test_quit(self):
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

    def todo_test_get_bold(self):

        # __doc__ (as of 2008-08-02) for pygame.font.Font.get_bold:

          # Font.get_bold(): return bool
          # check if text will be rendered bold
          # 
          # Return True when the font bold rendering mode is enabled. 

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

    def todo_test_get_italic(self):

        # __doc__ (as of 2008-08-02) for pygame.font.Font.get_italic:

          # Font.get_italic(): return bool
          # check if the text will be rendered italic
          # 
          # Return True when the font italic rendering mode is enabled. 

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

    def todo_test_get_underline(self):

        # __doc__ (as of 2008-08-02) for pygame.font.Font.get_underline:

          # Font.get_underline(): return bool
          # check if text will be rendered with an underline
          # 
          # Return True when the font underline is enabled. 

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

        self.fail() 

    def todo_test_set_bold(self):

        # __doc__ (as of 2008-08-02) for pygame.font.Font.set_bold:

          # Font.set_bold(bool): return None
          # enable fake rendering of bold text
          # 
          # Enables the bold rendering of text. This is a fake stretching of the
          # font that doesn't look good on many font types. If possible load the
          # font from a real bold font file. While bold, the font will have a
          # different width than when normal. This can be mixed with the italic
          # and underline modes.

        self.fail() 

    def todo_test_set_italic(self):

        # __doc__ (as of 2008-08-02) for pygame.font.Font.set_italic:

          # Font.set_bold(bool): return None
          # enable fake rendering of italic text
          # 
          # Enables fake rendering of italic text. This is a fake skewing of the
          # font that doesn't look good on many font types. If possible load the
          # font from a real italic font file. While italic the font will have a
          # different width than when normal. This can be mixed with the bold
          # and underline modes.

        self.fail() 

    def todo_test_set_underline(self):

        # __doc__ (as of 2008-08-02) for pygame.font.Font.set_underline:

          # Font.set_underline(bool): return None
          # control if text is rendered with an underline
          # 
          # When enabled, all rendered fonts will include an underline. The
          # underline is always one pixel thick, regardless of font size. This
          # can be mixed with the bold and italic modes.

        self.fail() 

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
