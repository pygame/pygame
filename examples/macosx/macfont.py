"""
EXPERIMENTAL CODE!

Here we load a .TTF font file, and display it in
a basic pygame window. It demonstrates several of the
Font object attributes. Nothing exciting in here, but
it makes a great example for basic window, event, and
font management.
"""


import pygame
import math
from pygame.locals import *
from pygame import Surface
from pygame.surfarray import blit_array, make_surface, pixels3d, pixels2d
import Numeric

from Foundation import *
from AppKit import *

def _getColor(color=None):
    if color is None:
        return NSColor.clearColor()
    div255 = (0.00390625).__mul__
    if len(color) == 3:
        color = tuple(color) + (255.0,)
    return NSColor.colorWithDeviceRed_green_blue_alpha_(*map(div255, color))

def _getBitmapImageRep(size, hasAlpha=True):
    width, height = map(int, size)
    return NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(None, width, height, 8, 4, hasAlpha, False, NSDeviceRGBColorSpace, width*4, 32)

class SysFont(object):
    def __init__(self, name, size):
        self._font = NSFont.fontWithName_size_(name, size)
        self._isBold = False
        self._isOblique = False
        self._isUnderline = False
        self._family = name
        self._size = size
        self._setupFont()

    def _setupFont(self):
        name = self._family
        if self._isBold or self._isOblique:
            name = '%s-%s%s' % (
                name,
                self._isBold and 'Bold' or '',
                self._isOblique and 'Oblique' or '')
        self._font = NSFont.fontWithName_size_(name, self._size)
        print (name, self._font)
        if self._font is None:
            if self._isBold:
                self._font = NSFont.boldSystemFontOfSize(self._size)
            else:
                self._font = NSFont.systemFontOfSize_(self._size)
        
    def get_ascent(self):
        return self._font.ascender()

    def get_descent(self):
        return -self._font.descender()

    def get_bold(self):
        return self._isBold

    def get_height(self):
        return self._font.defaultLineHeightForFont()

    def get_italic(self):
        return self._isOblique

    def get_linesize(self):
        pass

    def get_underline(self):
        return self._isUnderline

    def set_bold(self, isBold):
        if isBold != self._isBold:
            self._isBold = isBold
            self._setupFont()

    def set_italic(self, isOblique):
        if isOblique != self._isOblique:
            self._isOblique = isOblique
            self._setupFont()

    def set_underline(self, isUnderline):
        self._isUnderline = isUnderline
        
    def size(self, text):
        return tuple(map(int,map(math.ceil, NSString.sizeWithAttributes_(text, {
            NSFontAttributeName: self._font,
            NSUnderlineStyleAttributeName: self._isUnderline and 1.0 or None,
        }))))

    def render(self, text, antialias, forecolor, backcolor=(0,0,0,255)):
        size = self.size(text)
        img = NSImage.alloc().initWithSize_(size)
        img.lockFocus()

        NSString.drawAtPoint_withAttributes_(text, (0.0, 0.0), {
            NSFontAttributeName: self._font,
            NSUnderlineStyleAttributeName: self._isUnderline and 1.0 or None,
            NSBackgroundColorAttributeName: backcolor and _getColor(backcolor) or None,
            NSForegroundColorAttributeName: _getColor(forecolor),
        })

        rep = NSBitmapImageRep.alloc().initWithFocusedViewRect_(((0.0, 0.0), size))
        img.unlockFocus()
        if rep.samplesPerPixel() == 4:
            s = Surface(size, SRCALPHA|SWSURFACE, 32, [-1<<24,0xff<<16,0xff<<8,0xff])
            
            a = Numeric.reshape(Numeric.fromstring(rep.bitmapData(), typecode=Numeric.Int32), (size[1], size[0]))
            blit_array(s, Numeric.swapaxes(a,0,1))
            return s.convert_alpha()

if __name__=='__main__':
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    s = SysFont('Gill Sans', 36)
    s.set_italic(1)
    s.set_underline(1)
    done = False
    surf = s.render('OS X Fonts!', True, (255,0,0,255), (0,0,0,0))
    screen.blit(surf, (0,0))
    screen.blit(surf, (2, 2))
    pygame.display.update()
    while not done:

        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                done = True
                break
