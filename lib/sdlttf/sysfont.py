import pygame2.font
from pygame2.sdlttf import constants
from pygame2.sdlttf.base import Font

def get_sys_font (name, size, style=constants.STYLE_NORMAL):
    """
    get_sys_font (name, size, style=constants.STYLE_NORMAL) -> Font
    
    Create a Font from system font resources

    This will search the system fonts for the given font name. You can also
    enable bold or italic styles, and the appropriate system font will be
    selected if available.

    Name can also be a comma separated list of names, in which case set of
    names will be searched in order.
    """
    if style is None:
        style = constants.STYLE_NORMAL
    bold = (style & constants.STYLE_BOLD) == constants.STYLE_BOLD
    italic = (style & constants.STYLE_ITALIC) == constants.STYLE_ITALIC
    
    if name:
        fontname, gotbold, gotitalic = \
            pygame2.font.find_font (name, bold, italic, "ttf")
    if not fontname:
        return None
        
    font = Font (fontname, size)

    setstyle = constants.STYLE_NORMAL
    if (style & constants.STYLE_UNDERLINE):
        setstyle |= constants.STYLE_UNDERLINE
    if bold and not gotbold:
        style |= constants.STYLE_BOLD
    if italic and not gotitalic:
        style |= constants.STYLE_BOLD
    font.style = setstyle

    return font
