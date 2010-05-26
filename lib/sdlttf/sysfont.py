##    pygame - Python Game Library
##    Copyright (C) 2000-2003  Pete Shinners
##
##    This library is free software; you can redistribute it and/or
##    modify it under the terms of the GNU Library General Public
##    License as published by the Free Software Foundation; either
##    version 2 of the License, or (at your option) any later version.
##
##    This library is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##    Library General Public License for more details.
##
##    You should have received a copy of the GNU Library General Public
##    License along with this library; if not, write to the Free
##    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##

"""Simple module for creating SDL_ttf Font objects from system fonts."""

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
