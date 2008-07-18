import test_utils
import test.unittest as unittest

from test_utils import test_not_implemented

import pygame

class FontModuleTest( unittest.TestCase ):
    def testFontRendering( self ):
        """ 
        """
        #print __file__
        pygame.font.init ()
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
        #pygame.font.quit ()

    def test_SysFont(self):

        # __doc__ (as of 2008-06-25) for pygame.font.SysFont:

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

        self.assert_(test_not_implemented()) 

    def test_get_default_font(self):

        # __doc__ (as of 2008-06-25) for pygame.font.get_default_font:

          # pygame.font.get_default_font(): return string
          # get the filename of the default font

        self.assert_(test_not_implemented()) 

    def test_get_fonts(self):

        # __doc__ (as of 2008-06-25) for pygame.font.get_fonts:

          # pygame.font.get_fonts() -> list
          # get a list of system font names
          # 
          # Returns the list of all found system fonts. Note that
          # the names of the fonts will be all lowercase with spaces
          # removed. This is how pygame internally stores the font
          # names for matching.

        self.assert_(test_not_implemented()) 

    def test_get_init(self):

        # __doc__ (as of 2008-06-25) for pygame.font.get_init:

          # pygame.font.get_init(): return bool
          # true if the font module is initialized

        self.assert_(test_not_implemented()) 

    def test_init(self):

        # __doc__ (as of 2008-06-25) for pygame.font.init:

          # pygame.font.init(): return None
          # initialize the font module

        self.assert_(test_not_implemented()) 

    def test_match_font(self):

        # __doc__ (as of 2008-06-25) for pygame.font.match_font:

          # pygame.font.match_font(name, bold=0, italic=0) -> name
          # find the filename for the named system font
          # 
          # This performs the same font search as the SysFont()
          # function, only it returns the path to the TTF file
          # that would be loaded. The font name can be a comma
          # separated list of font names to try.
          # 
          # If no match is found, None is returned.

        self.assert_(test_not_implemented()) 

    def test_quit(self):

        # __doc__ (as of 2008-06-25) for pygame.font.quit:

          # pygame.font.quit(): return None
          # uninitialize the font module

        self.assert_(test_not_implemented()) 

class FontTypeTest( unittest.TestCase ):
    def test_get_ascent(self):
    
        # __doc__ (as of 2008-06-25) for pygame.font.Font.get_ascent:
    
          # Font.get_ascent(): return int
          # get the ascent of the font
    
        self.assert_(test_not_implemented()) 
    
    def test_get_bold(self):
    
        # __doc__ (as of 2008-06-25) for pygame.font.Font.get_bold:
    
          # Font.get_bold(): return bool
          # check if text will be rendered bold
    
        self.assert_(test_not_implemented()) 
    
    def test_get_descent(self):
    
        # __doc__ (as of 2008-06-25) for pygame.font.Font.get_descent:
    
          # Font.get_descent(): return int
          # get the descent of the font
    
        self.assert_(test_not_implemented()) 
    
    def test_get_height(self):
    
        # __doc__ (as of 2008-06-25) for pygame.font.Font.get_height:
    
          # Font.get_height(): return int
          # get the height of the font
    
        self.assert_(test_not_implemented()) 
    
    def test_get_italic(self):
    
        # __doc__ (as of 2008-06-25) for pygame.font.Font.get_italic:
    
          # Font.get_italic(): return bool
          # check if the text will be rendered italic
    
        self.assert_(test_not_implemented()) 
    
    def test_get_linesize(self):
    
        # __doc__ (as of 2008-06-25) for pygame.font.Font.get_linesize:
    
          # Font.get_linesize(): return int
          # get the line space of the font text
    
        self.assert_(test_not_implemented()) 
    
    def test_get_underline(self):
    
        # __doc__ (as of 2008-06-25) for pygame.font.Font.get_underline:
    
          # Font.get_underline(): return bool
          # check if text will be rendered with an underline
    
        self.assert_(test_not_implemented()) 
    
    def test_metrics(self):
    
        # __doc__ (as of 2008-06-25) for pygame.font.Font.metrics:
    
          # Font.metrics(text): return list
          # Gets the metrics for each character in the pased string.
    
        self.assert_(test_not_implemented()) 
    
    def test_render(self):
    
        # __doc__ (as of 2008-06-25) for pygame.font.Font.render:
    
          # Font.render(text, antialias, color, background=None): return Surface
          # draw text on a new Surface
    
        self.assert_(test_not_implemented()) 
    
    def test_set_bold(self):
    
        # __doc__ (as of 2008-06-25) for pygame.font.Font.set_bold:
    
          # Font.set_bold(bool): return None
          # enable fake rendering of bold text
    
        self.assert_(test_not_implemented()) 
    
    def test_set_italic(self):
    
        # __doc__ (as of 2008-06-25) for pygame.font.Font.set_italic:
    
          # Font.set_bold(bool): return None
          # enable fake rendering of italic text
    
        self.assert_(test_not_implemented()) 
    
    def test_set_underline(self):
    
        # __doc__ (as of 2008-06-25) for pygame.font.Font.set_underline:
    
          # Font.set_underline(bool): return None
          # control if text is rendered with an underline
    
        self.assert_(test_not_implemented()) 
    
    def test_size(self):
    
        # __doc__ (as of 2008-06-25) for pygame.font.Font.size:
    
          # Font.size(text): return (width, height)
          # determine the amount of space needed to render text
    
        self.assert_(test_not_implemented()) 

if __name__ == '__main__':
    unittest.main()