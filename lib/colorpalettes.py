##    pygame - Python Game Library
##    Copyright (C) 2009 Marcus von Appen
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

##
## This file is placed under the Public Domain.
##

"""
Various, indexed color palettes.

Indexed color palettes. The following palettes are currently available:

+--------------------+---------------------------------------------------+
| MONOPALETTE        | 1-bit monochrome palette (black and white).       |
+--------------------+---------------------------------------------------+
| GRAY2PALETTE       | 2-bit grayscale palette with black, white and two |
|                    | shades of gray.                                   |
+--------------------+---------------------------------------------------+
| GRAY4PALETTE       | 4-bit grayscale palette with black, white and     |
|                    | 14 shades shades of gray.                         |
+--------------------+---------------------------------------------------+
| GRAY8PALETTE       | 8-bit grayscale palette with black, white and     |
|                    | 254 shades shades of gray.                        |
+--------------------+---------------------------------------------------+
| RGB3PALETTE        | 3-bit RGB color palette with pure red, green and  |
|                    | blue and their complementary colors as well as    |
|                    | black and white.                                  |
+--------------------+---------------------------------------------------+
| CGAPALETTE         | CGA color palette.                                |
+--------------------+---------------------------------------------------+
| EGAPALETTE         | EGA color palette.                                |
+--------------------+---------------------------------------------------+
| VGAPALETTE         | 8-bit VGA color palette.                          |
+--------------------+---------------------------------------------------+
| WEBPALETTE         | "Safe" web color palette with 225 colors.         |
+--------------------+---------------------------------------------------+
"""

from pygame2.base import Color

def _create_8bpp_gray ():
    """Creates a 8 bit grayscale color palette."""
    l = []
    for x in range (0x00, 0xF1, 0x10):
        for y in range (0x00, 0x10, 0x01):
            l.append (Color(x | y, x | y, x | y))
    return tuple (l)

MONOPALETTE = ( Color(0xFF000000), Color(0xFFFFFFFF), )

GRAY2PALETTE = (
    Color(0xFF000000), Color(0xFF555555), Color(0xFFAAAAAA), Color(0xFFFFFFFF),
    )

GRAY4PALETTE = (
    Color(0xFF000000), Color(0xFF111111), Color(0xFF222222), Color(0xFF333333),
    Color(0xFF444444), Color(0xFF555555), Color(0xFF666666), Color(0xFF777777),
    Color(0xFF888888), Color(0xFF999999), Color(0xFFAAAAAA), Color(0xFFBBBBBB),
    Color(0xFFCCCCCC), Color(0xFFDDDDDD), Color(0xFFEEEEEE), Color(0xFFFFFFFF),
    )

GRAY8PALETTE = _create_8bpp_gray ()

CGAPALETTE = (
    Color(0xFF000000), Color(0xFF0000AA), Color(0xFF00AA00), Color(0xFF00AAAA),
    Color(0xFFAA0000), Color(0xFFAA00AA), Color(0xFFAA5500), Color(0xFFAAAAAA),
    Color(0xFF555555), Color(0xFF5555FF), Color(0xFF55FF55), Color(0xFF55FFFF),
    Color(0xFFFF5555), Color(0xFFFF55FF), Color(0xFFFFFF55), Color(0xFFFFFFFF),
    )

EGAPALETTE = (
    Color(0xFF000000), Color(0xFF0000AA), Color(0xFF00AA00), Color(0xFF00AAAA),
    Color(0xFFAA0000), Color(0xFFAA00AA), Color(0xFFAAAA00), Color(0xFFAAAAAA),
    Color(0xFF000055), Color(0xFF0000FF), Color(0xFF00AA55), Color(0xFF00AAFF),
    Color(0xFFAA0055), Color(0xFFAA00FF), Color(0xFFAAAA55), Color(0xFFAAAAFF),
    Color(0xFF005500), Color(0xFF0055AA), Color(0xFF00FF00), Color(0xFF00FFAA),
    Color(0xFFAA5500), Color(0xFFAA55AA), Color(0xFFAAFF00), Color(0xFFAAFFAA),
    Color(0xFF005555), Color(0xFF0055FF), Color(0xFF00FF55), Color(0xFF00FFFF),
    Color(0xFFAA5555), Color(0xFFAA55FF), Color(0xFFAAFF55), Color(0xFFAAFFFF),
    Color(0xFF550000), Color(0xFF5500AA), Color(0xFF55AA00), Color(0xFF55AAAA),
    Color(0xFFFF0000), Color(0xFFFF00AA), Color(0xFFFFAA00), Color(0xFFFFAAAA),
    Color(0xFF550055), Color(0xFF5500FF), Color(0xFF55AA55), Color(0xFF55AAFF),
    Color(0xFFFF0055), Color(0xFFFF00FF), Color(0xFFFFAA55), Color(0xFFFFAAFF),
    Color(0xFF555500), Color(0xFF5555AA), Color(0xFF55FF00), Color(0xFF55FFAA),
    Color(0xFFFF5500), Color(0xFFFF55AA), Color(0xFFFFFF00), Color(0xFFFFFFAA),
    Color(0xFF555555), Color(0xFF5555FF), Color(0xFF55FF55), Color(0xFF55FFFF),
    Color(0xFFFF5555), Color(0xFFFF55FF), Color(0xFFFFFF55), Color(0xFFFFFFFF),
    )

WEBPALETTE = (
    Color(0xFFFFFFFF), Color(0xFFFFFFCC), Color(0xFFFFFF99), Color(0xFFFFFF66),
    Color(0xFFFFFF33), Color(0xFFFFFF00), Color(0xFFFFCCFF), Color(0xFFFFCCCC),
    Color(0xFFFFCC99), Color(0xFFFFCC66), Color(0xFFFFCC33), Color(0xFFFFCC00),
    Color(0xFFFF99FF), Color(0xFFFF99CC), Color(0xFFFF9999), Color(0xFFFF9966),
    Color(0xFFFF9933), Color(0xFFFF9900), Color(0xFFFF66FF), Color(0xFFFF66CC),
    Color(0xFFFF6699), Color(0xFFFF6666), Color(0xFFFF6633), Color(0xFFFF6600),
    Color(0xFFFF33FF), Color(0xFFFF33CC), Color(0xFFFF3399), Color(0xFFFF3366),
    Color(0xFFFF3333), Color(0xFFFF3300), Color(0xFFFF00FF), Color(0xFFFF00CC),
    Color(0xFFFF0099), Color(0xFFFF0066), Color(0xFFFF0033), Color(0xFFFF0000),
    Color(0xFFCCFFFF), Color(0xFFCCFFCC), Color(0xFFCCFF99), Color(0xFFCCFF66),
    Color(0xFFCCFF33), Color(0xFFCCFF00), Color(0xFFCCCCFF), Color(0xFFCCCCCC),
    Color(0xFFCCCC99), Color(0xFFCCCC66), Color(0xFFCCCC33), Color(0xFFCCCC00),
    Color(0xFFCC99FF), Color(0xFFCC99CC), Color(0xFFCC9999), Color(0xFFCC9966),
    Color(0xFFCC9933), Color(0xFFCC9900), Color(0xFFCC66FF), Color(0xFFCC66CC),
    Color(0xFFCC6699), Color(0xFFCC6666), Color(0xFFCC6633), Color(0xFFCC6600),
    Color(0xFFCC33FF), Color(0xFFCC33CC), Color(0xFFCC3399), Color(0xFFCC3366),
    Color(0xFFCC3333), Color(0xFFCC3300), Color(0xFFCC00FF), Color(0xFFCC00CC),
    Color(0xFFCC0099), Color(0xFFCC0066), Color(0xFFCC0033), Color(0xFFCC0000),
    Color(0xFF99FFFF), Color(0xFF99FFCC), Color(0xFF99FF99), Color(0xFF99FF66),
    Color(0xFF99FF33), Color(0xFF99FF00), Color(0xFF99CCFF), Color(0xFF99CCCC),
    Color(0xFF99CC99), Color(0xFF99CC66), Color(0xFF99CC33), Color(0xFF99CC00),
    Color(0xFF9999FF), Color(0xFF9999CC), Color(0xFF999999), Color(0xFF999966),
    Color(0xFF999933), Color(0xFF999900), Color(0xFF9966FF), Color(0xFF9966CC),
    Color(0xFF996699), Color(0xFF996666), Color(0xFF996633), Color(0xFF996600), 
    Color(0xFF9933FF), Color(0xFF9933CC), Color(0xFF993399), Color(0xFF993366),
    Color(0xFF993333), Color(0xFF993300), Color(0xFF9900FF), Color(0xFF9900CC),
    Color(0xFF990099), Color(0xFF990066), Color(0xFF990033), Color(0xFF990000),
    Color(0xFF66FFFF), Color(0xFF66FFCC), Color(0xFF66FF99), Color(0xFF66FF66),
    Color(0xFF66FF33), Color(0xFF66FF00), Color(0xFF66CCFF), Color(0xFF66CCCC),
    Color(0xFF66CC99), Color(0xFF66CC66), Color(0xFF66CC33), Color(0xFF66CC00),
    Color(0xFF6699FF), Color(0xFF6699CC), Color(0xFF669999), Color(0xFF669966),
    Color(0xFF669933), Color(0xFF669900), Color(0xFF6666FF), Color(0xFF6666CC),
    Color(0xFF666699), Color(0xFF666666), Color(0xFF666633), Color(0xFF666600),
    Color(0xFF6633FF), Color(0xFF6633CC), Color(0xFF663399), Color(0xFF663366),
    Color(0xFF663333), Color(0xFF663300), Color(0xFF6600FF), Color(0xFF6600CC),
    Color(0xFF660099), Color(0xFF660066), Color(0xFF660033), Color(0xFF660000),
    Color(0xFF33FFFF), Color(0xFF33FFCC), Color(0xFF33FF99), Color(0xFF33FF66),
    Color(0xFF33FF33), Color(0xFF33FF00), Color(0xFF33CCFF), Color(0xFF33CCCC),
    Color(0xFF33CC99), Color(0xFF33CC66), Color(0xFF33CC33), Color(0xFF33CC00),
    Color(0xFF3399FF), Color(0xFF3399CC), Color(0xFF339999), Color(0xFF339966),
    Color(0xFF339933), Color(0xFF339900), Color(0xFF3366FF), Color(0xFF3366CC),
    Color(0xFF336699), Color(0xFF336666), Color(0xFF336633), Color(0xFF336600),
    Color(0xFF3333FF), Color(0xFF3333CC), Color(0xFF333399), Color(0xFF333366),
    Color(0xFF333333), Color(0xFF333300), Color(0xFF3300FF), Color(0xFF3300CC),
    Color(0xFF330099), Color(0xFF330066), Color(0xFF330033), Color(0xFF330000),
    Color(0xFF00FFFF), Color(0xFF00FFCC), Color(0xFF00FF99), Color(0xFF00FF66),
    Color(0xFF00FF33), Color(0xFF00FF00), Color(0xFF00CCFF), Color(0xFF00CCCC),
    Color(0xFF00CC99), Color(0xFF00CC66), Color(0xFF00CC33), Color(0xFF00CC00),
    Color(0xFF0099FF), Color(0xFF0099CC), Color(0xFF009999), Color(0xFF009966),
    Color(0xFF009933), Color(0xFF009900), Color(0xFF0066FF), Color(0xFF0066CC),
    Color(0xFF006699), Color(0xFF006666), Color(0xFF006633), Color(0xFF006600),
    Color(0xFF0033FF), Color(0xFF0033CC), Color(0xFF003399), Color(0xFF003366),
    Color(0xFF003333), Color(0xFF003300), Color(0xFF0000FF), Color(0xFF0000CC),
    Color(0xFF000099), Color(0xFF000066), Color(0xFF000033), Color(0xFF000000),
    )

RGB3PALETTE = (
    Color(0xFF000000), Color(0xFF0000FF), Color(0xFF00FF00), Color(0xFF00FFFF),
    Color(0xFFFF0000), Color(0xFFFF00FF), Color(0xFFFFFF00), Color(0xFFFFFFFF),
    )

VGAPALETTE = (
    Color(0xFF000000), Color(0xFF0000AA), Color(0xFF00AA00), Color(0xFF00AAAA),
    Color(0xFFAA0000), Color(0xFFAA00AA), Color(0xFFAA5500), Color(0xFFAAAAAA),
    Color(0xFF555555), Color(0xFF5555FF), Color(0xFF55FF55), Color(0xFF55FFFF),
    Color(0xFFFF5555), Color(0xFFFF55FF), Color(0xFFFFFF55), Color(0xFFFFFFFF),
    Color(0xFF000000), Color(0xFF101010), Color(0xFF202020), Color(0xFF353535),
    Color(0xFF454545), Color(0xFF555555), Color(0xFF656565), Color(0xFF757575),
    Color(0xFF8A8A8A), Color(0xFF9A9A9A), Color(0xFFAAAAAA), Color(0xFFBABABA),
    Color(0xFFCACACA), Color(0xFFDFDFDF), Color(0xFFEFEFEF), Color(0xFFFFFFFF),
    Color(0xFF0000FF), Color(0xFF4100FF), Color(0xFF8200FF), Color(0xFFBE00FF),
    Color(0xFFFF00FF), Color(0xFFFF00BE), Color(0xFFFF0082), Color(0xFFFF0041),
    Color(0xFFFF0000), Color(0xFFFF4100), Color(0xFFFF8200), Color(0xFFFFBE00),
    Color(0xFFFFFF00), Color(0xFFBEFF00), Color(0xFF82FF00), Color(0xFF41FF00),
    Color(0xFF00FF00), Color(0xFF00FF41), Color(0xFF00FF82), Color(0xFF00FFBE),
    Color(0xFF00FFFF), Color(0xFF00BEFF), Color(0xFF0082FF), Color(0xFF0041FF),
    Color(0xFF8282FF), Color(0xFF9E82FF), Color(0xFFBE82FF), Color(0xFFDF82FF),
    Color(0xFFFF82FF), Color(0xFFFF82DF), Color(0xFFFF82BE), Color(0xFFFF829E),
    Color(0xFFFF8282), Color(0xFFFF9E82), Color(0xFFFFBE82), Color(0xFFFFDF82),
    Color(0xFFFFFF82), Color(0xFFDFFF82), Color(0xFFBEFF82), Color(0xFF9EFF82),
    Color(0xFF82FF82), Color(0xFF82FF9E), Color(0xFF82FFBE), Color(0xFF82FFDF),
    Color(0xFF82FFFF), Color(0xFF82DFFF), Color(0xFF82BEFF), Color(0xFF829EFF),
    Color(0xFFBABAFF), Color(0xFFCABAFF), Color(0xFFDFBAFF), Color(0xFFEFBAFF),
    Color(0xFFFFBAFF), Color(0xFFFFBAEF), Color(0xFFFFBADF), Color(0xFFFFBACA),
    Color(0xFFFFBABA), Color(0xFFFFCABA), Color(0xFFFFDFBA), Color(0xFFFFEFBA),
    Color(0xFFFFFFBA), Color(0xFFEFFFBA), Color(0xFFDFFFBA), Color(0xFFCAFFBA),
    Color(0xFFBAFFBA), Color(0xFFBAFFCA), Color(0xFFBAFFDF), Color(0xFFBAFFEF),
    Color(0xFFBAFFFF), Color(0xFFBAEFFF), Color(0xFFBADFFF), Color(0xFFBACAFF),
    Color(0xFF000071), Color(0xFF1C0071), Color(0xFF390071), Color(0xFF550071),
    Color(0xFF710071), Color(0xFF710055), Color(0xFF710039), Color(0xFF71001C),
    Color(0xFF710000), Color(0xFF711C00), Color(0xFF713900), Color(0xFF715500),
    Color(0xFF717100), Color(0xFF557100), Color(0xFF397100), Color(0xFF1C7100),
    Color(0xFF007100), Color(0xFF00711C), Color(0xFF007139), Color(0xFF007155),
    Color(0xFF007171), Color(0xFF005571), Color(0xFF003971), Color(0xFF001C71),
    Color(0xFF393971), Color(0xFF453971), Color(0xFF553971), Color(0xFF613971),
    Color(0xFF713971), Color(0xFF713961), Color(0xFF713955), Color(0xFF713945),
    Color(0xFF713939), Color(0xFF714539), Color(0xFF715539), Color(0xFF716139),
    Color(0xFF717139), Color(0xFF617139), Color(0xFF557139), Color(0xFF457139),
    Color(0xFF397139), Color(0xFF397145), Color(0xFF397155), Color(0xFF397161),
    Color(0xFF397171), Color(0xFF396171), Color(0xFF395571), Color(0xFF394571),
    Color(0xFF515171), Color(0xFF595171), Color(0xFF615171), Color(0xFF695171),
    Color(0xFF715171), Color(0xFF715169), Color(0xFF715161), Color(0xFF715159),
    Color(0xFF715151), Color(0xFF715951), Color(0xFF716151), Color(0xFF716951),
    Color(0xFF717151), Color(0xFF697151), Color(0xFF617151), Color(0xFF597151),
    Color(0xFF517151), Color(0xFF517159), Color(0xFF517161), Color(0xFF517169),
    Color(0xFF517171), Color(0xFF516971), Color(0xFF516171), Color(0xFF515971),
    Color(0xFF000041), Color(0xFF100041), Color(0xFF200041), Color(0xFF310041),
    Color(0xFF410041), Color(0xFF410031), Color(0xFF410020), Color(0xFF410010),
    Color(0xFF410000), Color(0xFF411000), Color(0xFF412000), Color(0xFF413100),
    Color(0xFF414100), Color(0xFF314100), Color(0xFF204100), Color(0xFF104100),
    Color(0xFF004100), Color(0xFF004110), Color(0xFF004120), Color(0xFF004131),
    Color(0xFF004141), Color(0xFF003141), Color(0xFF002041), Color(0xFF001041),
    Color(0xFF202041), Color(0xFF282041), Color(0xFF312041), Color(0xFF392041),
    Color(0xFF412041), Color(0xFF412039), Color(0xFF412031), Color(0xFF412028),
    Color(0xFF412020), Color(0xFF412820), Color(0xFF413120), Color(0xFF413920),
    Color(0xFF414120), Color(0xFF394120), Color(0xFF314120), Color(0xFF284120),
    Color(0xFF204120), Color(0xFF204128), Color(0xFF204131), Color(0xFF204139),
    Color(0xFF204141), Color(0xFF203941), Color(0xFF203141), Color(0xFF202841),
    Color(0xFF2D2D41), Color(0xFF312D41), Color(0xFF352D41), Color(0xFF3D2D41),
    Color(0xFF412D41), Color(0xFF412D3D), Color(0xFF412D35), Color(0xFF412D31),
    Color(0xFF412D2D), Color(0xFF41312D), Color(0xFF41352D), Color(0xFF413D2D),
    Color(0xFF41412D), Color(0xFF3D412D), Color(0xFF35412D), Color(0xFF31412D),
    Color(0xFF2D412D), Color(0xFF2D4131), Color(0xFF2D4135), Color(0xFF2D413D),
    Color(0xFF2D4141), Color(0xFF2D3D41), Color(0xFF2D3541), Color(0xFF2D3141),
    Color(0xFF000000), Color(0xFF000000), Color(0xFF000000), Color(0xFF000000),
    Color(0xFF000000), Color(0xFF000000), Color(0xFF000000), Color(0xFF000000),
    )
