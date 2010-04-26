##    pygame - Python Game Library
##    Copyright (C) 2008 Marcus von Appen
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

"""
Enhanced and specialised font rendering classes and functions for
SDL-based surfaces.
"""

import os
from pygame2 import Rect, FRect, Color
from pygame2.sdl.video import Surface

DEFAULTMAP = [ "0123456789",
               "ABCDEFGHIJ",
               "KLMNOPQRST",
               "UVWXYZ    ",
               "abcdefghij",
               "klmnopqrst",
               "uvwxyz    ",
               ",;.:!?+-()" ]

class BitmapFont (object):
    """BitmapFont (surface, size, mapping=None) -> BitmapFont
    
    Creates a BitmapFont suitable for bitmap to character mappings.
    
    The BitmapFont class uses an image surface to find and render font
    character glyphs for text. It requires a mapping table, which
    denotes the characters available on the image.

    The mapping table is a list of strings, where each string reflects a
    'line' of characters on the image. Each character within each line
    has the same size as specified by the size argument.

    A typical mapping table might look like

      [ '0123456789',
        'ABCDEFGHIJ',
        'KLMNOPQRST',
        'UVWXYZ    ',
        'abcdefghij',
        'klmnopqrst',
        'uvwxyz    ',
        ',;.:!?+-()' ]
    """
    def __init__ (self, surface, size, mapping=None):
        if mapping is None:
            self.mapping = list (DEFAULTMAP)
        else:
            self.mapping = mapping

        self.offsets = {}
        self.surface = surface
        
        if isinstance (size, Rect) or isinstance (size, FRect):
            self.size = size.size
        else:
            self.size = size[0], size[1]

        self._calculate_offsets ()
    
    def _calculate_offsets (self):
        """
        """
        self.offsets = {}
        offsets = self.offsets
        x, y = 0, 0
        w, h = self.size
        for line in self.mapping:
            x = 0
            for c in line:
                offsets[c] = Rect (x, y, w, h)
                x += w
            y += h

    def render (self, text):
        """B.render (text) -> Surface

        Renders the passed text on a new Surface and returns it.
        """
        x, y = 0, 0
        tw, th = 0, 0
        w, h = self.size
            
        lines = text.split (os.linesep)
        for line in lines:
            tw = max (tw, sum ([w for c in line]))
            th += h
        
        surface = Surface (tw, th)
        blit = surface.blit
        fontsf = self.surface
        offsets = self.offsets
        for line in lines:
            for c in line:
                blit (fontsf, (x, y), offsets[c])
                x += w
            y += h
        return surface

    def render_on (self, surface, text, offset=(0, 0)):
        """B.render_on (surface, text, offset=(0, 0) -> Rect

        Renders a text on the passed surface, starting at offset.

        Renders a text on the existing surface. The top-left start
        position of the text will be the passed offset and a Rect with
        the changed area will be returned.
        """
        x, y = offset
        w, h = self.size
            
        lines = text.split (os.linesep)
        blit = surface.blit
        fontsf = self.surface
        offsets = self.offsets
        for line in lines:
            for c in line:
                blit (fontsf, (x, y), offsets[c])
                x += w
            y += h
        return Rect (offset, (x + w, y + h))
    
    def contains (self, c):
        """B.contains (c) -> bool

        Checks, whether a certain character exists in the font.
        """
        return c in self.offsets
    
    def can_render (self, text):
        """B.can_render (text) -> bool

        Checks, whether all characters in the passed text can be rendered.
        """
        lines = text.split (os.linesep)
        has_key = self.offsets.has_key
        for line in lines:
            for c in line:
                if not has_key (c):
                    return False
        return True
