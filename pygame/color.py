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
##    Pete Shinners
##    pete@shinners.org

"""Manipulate colors"""


try:
    from colordict import THECOLORS
except ImportError:
    #the colordict module isn't available
    THECOLORS = {}



def Color(colorname):
    """pygame.color.Color(colorname) -> RGBA
       Get RGB values from common color names

       The color name can be the name of a common english color,
       or a "web" style color in the form of 0xFF00FF. The english
       color names are defined by the standard 'rgb' colors for X11.
       With the hex color formatting you may optionally include an
       alpha value, the formatting is 0xRRGGBBAA. You may also specify
       a hex formatted color by starting the string with a '#'.
       The color name used is case insensitive and whitespace is ignored.
    """

    if colorname[:2] == '0x' or colorname[0] == '#': #webstyle
        if colorname[0] == '#':
            colorname = colorname[1:]
        else:
            colorname = colorname[2:]
        a = 255
        try:
            r = int('0x' + colorname[0:2], 16)
            g = int('0x' + colorname[2:4], 16)
            b = int('0x' + colorname[4:6], 16)
            if len(colorname) > 6:
                a = int('0x' + colorname[6:8], 16)
        except ValueError:
            raise ValueError, "Illegal hex color"
        return r, g, b, a

    else: #color name
        #no spaces and lowercase
        name = colorname.replace(' ', '').lower()
        try:
            return THECOLORS[name]
        except KeyError:
            raise ValueError, "Illegal color name, " + name



def _splitcolor(color, defaultalpha=255):
    try:
        second = int(color)
        r = g = b = color
        a = defaultalpha
    except TypeError:
        if len(color) == 4:
            r, g, b, a = color
        elif len(color) == 3:
            r, g, b = color
            a = defaultalpha
    return r, g, b, a


def add(color1, color2):
    """pygame.color.add(color1, color2) -> RGBA
       add two colors

       Add the RGB values of two colors together. If one of the
       colors is only a single numeric value, it is applied to the
       RGB components of the first color. Color values will be clamped
       to the maximum color value of 255.
    """
    r1, g1, b1, a1 = _splitcolor(color1)
    r2, g2, b2, a2 = _splitcolor(color2)
    m, i = min, int
    return m(i(r1+r2), 255), m(i(g1+g2), 255), m(i(b1+b2), 255), m(i(a1+a2), 255)


def subtract(color1, color2):
    """pygame.color.subtract(color1, color2) -> RGBA
       subtract two colors

       Subtract the RGB values of two colors together. If one of the
       colors is only a single numeric value, it is applied to the
       RGB components of the first color. Color values will be clamped
       to the minimum color value of 0.
    """
    r1, g1, b1, a1 = _splitcolor(color1)
    r2, g2, b2, a2 = _splitcolor(color2, 0)
    m, i = max, int
    return m(i(r1-r2), 0), m(i(g1-g2), 0), m(i(b1-b2), 0), m(i(a1-a2), 0)


def multiply(color1, color2):
    """pygame.color.multiply(color1, color2) -> RGBA
       multiply two colors

       Multiply the RGB values of two colors together. If one of the
       colors is only a single numeric value, it is applied to the
       RGB components of the first color.
    """
    r1, g1, b1, a1 = _splitcolor(color1)
    r2, g2, b2, a2 = _splitcolor(color2)
    m, i = min, int
    return m(i(r1*r2)/255, 255), m(i(g1*g2)/255, 255), m(i(b1*b2)/255, 255), m(i(a1*a2)/255, 255)

