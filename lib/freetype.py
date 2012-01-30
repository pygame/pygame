"""Enhanced Pygame module for loading and rendering font faces"""

from pygame._freetype import *
from pygame.sysfont import match_font, get_fonts, SysFont as _SysFont

def SysFont(name, size, bold=0, italic=0, constructor=None):
    """pygame.ftfont.SysFont(name, size, bold=False, italic=False, constructor=None) -> Font
       create a pygame Font from system font resources

       This will search the system fonts for the given font
       name. You can also enable bold or italic styles, and
       the appropriate system font will be selected if available.

       This will always return a valid Font object, and will
       fallback on the builtin pygame font if the given font
       is not found.

       Name can also be a comma separated list of names, in
       which case set of names will be searched in order. Pygame
       uses a small set of common font aliases, if the specific
       font you ask for is not available, a reasonable alternative
       may be used.

       if optional contructor is provided, it must be a function with
       signature constructor(fontpath, size, bold, italic) which returns
       a Font instance. If None, a pygame.freetype.Face object is created.
    """
    if constructor is None:
        def constructor(fontpath, size, bold, italic):
            return Face(fontpath, size)

    return _SysFont(name, size, bold, italic, constructor)


