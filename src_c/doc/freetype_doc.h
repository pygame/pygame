/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMEFREETYPE "Enhanced pygame module for loading and rendering computer fonts"

#define DOC_PYGAMEFREETYPEGETERROR "get_error() -> str\nReturn the latest FreeType error"

#define DOC_PYGAMEFREETYPEGETVERSION "get_version() -> (int, int, int)\nReturn the FreeType version"

#define DOC_PYGAMEFREETYPEINIT "init(cache_size=64, resolution=72)\nInitialize the underlying FreeType library."

#define DOC_PYGAMEFREETYPEQUIT "quit()\nShut down the underlying FreeType library."

#define DOC_PYGAMEFREETYPEGETINIT "get_init() -> bool\nReturns True if the FreeType module is currently initialized."

#define DOC_PYGAMEFREETYPEWASINIT "was_init() -> bool\nDEPRECATED: Use get_init() instead."

#define DOC_PYGAMEFREETYPEGETCACHESIZE "get_cache_size() -> long\nReturn the glyph case size"

#define DOC_PYGAMEFREETYPEGETDEFAULTRESOLUTION "get_default_resolution() -> long\nReturn the default pixel size in dots per inch"

#define DOC_PYGAMEFREETYPESETDEFAULTRESOLUTION "set_default_resolution([resolution])\nSet the default pixel size in dots per inch for the module"

#define DOC_PYGAMEFREETYPESYSFONT "SysFont(name, size, bold=False, italic=False) -> Font\ncreate a Font object from the system fonts"

#define DOC_PYGAMEFREETYPEGETDEFAULTFONT "get_default_font() -> string\nGet the filename of the default font"

#define DOC_PYGAMEFREETYPEFONT "Font(file, size=0, font_index=0, resolution=0, ucs4=False) -> Font\nCreate a new Font instance from a supported font file."

#define DOC_FONTNAME "name -> string\nProper font name."

#define DOC_FONTPATH "path -> unicode\nFont file path"

#define DOC_FONTSIZE "size -> float\nsize -> (float, float)\nThe default point size used in rendering"

#define DOC_FONTGETRECT "get_rect(text, style=STYLE_DEFAULT, rotation=0, size=0) -> rect\nReturn the size and offset of rendered text"

#define DOC_FONTGETMETRICS "get_metrics(text, size=0) -> [(...), ...]\nReturn the glyph metrics for the given text"

#define DOC_FONTHEIGHT "height -> int\nThe unscaled height of the font in font units"

#define DOC_FONTASCENDER "ascender -> int\nThe unscaled ascent of the font in font units"

#define DOC_FONTDESCENDER "descender -> int\nThe unscaled descent of the font in font units"

#define DOC_FONTGETSIZEDASCENDER "get_sized_ascender(<size>=0) -> int\nThe scaled ascent of the font in pixels"

#define DOC_FONTGETSIZEDDESCENDER "get_sized_descender(<size>=0) -> int\nThe scaled descent of the font in pixels"

#define DOC_FONTGETSIZEDHEIGHT "get_sized_height(<size>=0) -> int\nThe scaled height of the font in pixels"

#define DOC_FONTGETSIZEDGLYPHHEIGHT "get_sized_glyph_height(<size>=0) -> int\nThe scaled bounding box height of the font in pixels"

#define DOC_FONTGETSIZES "get_sizes() -> [(int, int, int, float, float), ...]\nget_sizes() -> []\nreturn the available sizes of embedded bitmaps"

#define DOC_FONTRENDER "render(text, fgcolor=None, bgcolor=None, style=STYLE_DEFAULT, rotation=0, size=0) -> (Surface, Rect)\nReturn rendered text as a surface"

#define DOC_FONTRENDERTO "render_to(surf, dest, text, fgcolor=None, bgcolor=None, style=STYLE_DEFAULT, rotation=0, size=0) -> Rect\nRender text onto an existing surface"

#define DOC_FONTRENDERRAW "render_raw(text, style=STYLE_DEFAULT, rotation=0, size=0, invert=False) -> (bytes, (int, int))\nReturn rendered text as a string of bytes"

#define DOC_FONTRENDERRAWTO "render_raw_to(array, text, dest=None, style=STYLE_DEFAULT, rotation=0, size=0, invert=False) -> (int, int)\nRender text into an array of ints"

#define DOC_FONTSTYLE "style -> int\nThe font's style flags"

#define DOC_FONTUNDERLINE "underline -> bool\nThe state of the font's underline style flag"

#define DOC_FONTSTRONG "strong -> bool\nThe state of the font's strong style flag"

#define DOC_FONTOBLIQUE "oblique -> bool\nThe state of the font's oblique style flag"

#define DOC_FONTWIDE "wide -> bool\nThe state of the font's wide style flag"

#define DOC_FONTSTRENGTH "strength -> float\nThe strength associated with the strong or wide font styles"

#define DOC_FONTUNDERLINEADJUSTMENT "underline_adjustment -> float\nAdjustment factor for the underline position"

#define DOC_FONTFIXEDWIDTH "fixed_width -> bool\nGets whether the font is fixed-width"

#define DOC_FONTFIXEDSIZES "fixed_sizes -> int\nthe number of available bitmap sizes for the font"

#define DOC_FONTSCALABLE "scalable -> bool\nGets whether the font is scalable"

#define DOC_FONTUSEBITMAPSTRIKES "use_bitmap_strikes -> bool\nallow the use of embedded bitmaps in an outline font file"

#define DOC_FONTANTIALIASED "antialiased -> bool\nFont anti-aliasing mode"

#define DOC_FONTKERNING "kerning -> bool\nCharacter kerning mode"

#define DOC_FONTVERTICAL "vertical -> bool\nFont vertical mode"

#define DOC_FONTROTATION "rotation -> int\ntext rotation in degrees counterclockwise"

#define DOC_FONTFGCOLOR "fgcolor -> Color\ndefault foreground color"

#define DOC_FONTORIGIN "origin -> bool\nFont render to text origin mode"

#define DOC_FONTPAD "pad -> bool\npadded boundary mode"

#define DOC_FONTUCS4 "ucs4 -> bool\nEnable UCS-4 mode"

#define DOC_FONTRESOLUTION "resolution -> int\nPixel resolution in dots per inch"



/* Docs in a comment... slightly easier to read. */

/*

pygame.freetype
Enhanced pygame module for loading and rendering computer fonts

pygame.freetype.get_error
 get_error() -> str
Return the latest FreeType error

pygame.freetype.get_version
 get_version() -> (int, int, int)
Return the FreeType version

pygame.freetype.init
 init(cache_size=64, resolution=72)
Initialize the underlying FreeType library.

pygame.freetype.quit
 quit()
Shut down the underlying FreeType library.

pygame.freetype.get_init
 get_init() -> bool
Returns True if the FreeType module is currently initialized.

pygame.freetype.was_init
 was_init() -> bool
DEPRECATED: Use get_init() instead.

pygame.freetype.get_cache_size
 get_cache_size() -> long
Return the glyph case size

pygame.freetype.get_default_resolution
 get_default_resolution() -> long
Return the default pixel size in dots per inch

pygame.freetype.set_default_resolution
 set_default_resolution([resolution])
Set the default pixel size in dots per inch for the module

pygame.freetype.SysFont
 SysFont(name, size, bold=False, italic=False) -> Font
create a Font object from the system fonts

pygame.freetype.get_default_font
 get_default_font() -> string
Get the filename of the default font

pygame.freetype.Font
 Font(file, size=0, font_index=0, resolution=0, ucs4=False) -> Font
Create a new Font instance from a supported font file.

pygame.freetype.Font.name
 name -> string
Proper font name.

pygame.freetype.Font.path
 path -> unicode
Font file path

pygame.freetype.Font.size
 size -> float
 size -> (float, float)
The default point size used in rendering

pygame.freetype.Font.get_rect
 get_rect(text, style=STYLE_DEFAULT, rotation=0, size=0) -> rect
Return the size and offset of rendered text

pygame.freetype.Font.get_metrics
 get_metrics(text, size=0) -> [(...), ...]
Return the glyph metrics for the given text

pygame.freetype.Font.height
 height -> int
The unscaled height of the font in font units

pygame.freetype.Font.ascender
 ascender -> int
The unscaled ascent of the font in font units

pygame.freetype.Font.descender
 descender -> int
The unscaled descent of the font in font units

pygame.freetype.Font.get_sized_ascender
 get_sized_ascender(<size>=0) -> int
The scaled ascent of the font in pixels

pygame.freetype.Font.get_sized_descender
 get_sized_descender(<size>=0) -> int
The scaled descent of the font in pixels

pygame.freetype.Font.get_sized_height
 get_sized_height(<size>=0) -> int
The scaled height of the font in pixels

pygame.freetype.Font.get_sized_glyph_height
 get_sized_glyph_height(<size>=0) -> int
The scaled bounding box height of the font in pixels

pygame.freetype.Font.get_sizes
 get_sizes() -> [(int, int, int, float, float), ...]
 get_sizes() -> []
return the available sizes of embedded bitmaps

pygame.freetype.Font.render
 render(text, fgcolor=None, bgcolor=None, style=STYLE_DEFAULT, rotation=0, size=0) -> (Surface, Rect)
Return rendered text as a surface

pygame.freetype.Font.render_to
 render_to(surf, dest, text, fgcolor=None, bgcolor=None, style=STYLE_DEFAULT, rotation=0, size=0) -> Rect
Render text onto an existing surface

pygame.freetype.Font.render_raw
 render_raw(text, style=STYLE_DEFAULT, rotation=0, size=0, invert=False) -> (bytes, (int, int))
Return rendered text as a string of bytes

pygame.freetype.Font.render_raw_to
 render_raw_to(array, text, dest=None, style=STYLE_DEFAULT, rotation=0, size=0, invert=False) -> (int, int)
Render text into an array of ints

pygame.freetype.Font.style
 style -> int
The font's style flags

pygame.freetype.Font.underline
 underline -> bool
The state of the font's underline style flag

pygame.freetype.Font.strong
 strong -> bool
The state of the font's strong style flag

pygame.freetype.Font.oblique
 oblique -> bool
The state of the font's oblique style flag

pygame.freetype.Font.wide
 wide -> bool
The state of the font's wide style flag

pygame.freetype.Font.strength
 strength -> float
The strength associated with the strong or wide font styles

pygame.freetype.Font.underline_adjustment
 underline_adjustment -> float
Adjustment factor for the underline position

pygame.freetype.Font.fixed_width
 fixed_width -> bool
Gets whether the font is fixed-width

pygame.freetype.Font.fixed_sizes
 fixed_sizes -> int
the number of available bitmap sizes for the font

pygame.freetype.Font.scalable
 scalable -> bool
Gets whether the font is scalable

pygame.freetype.Font.use_bitmap_strikes
 use_bitmap_strikes -> bool
allow the use of embedded bitmaps in an outline font file

pygame.freetype.Font.antialiased
 antialiased -> bool
Font anti-aliasing mode

pygame.freetype.Font.kerning
 kerning -> bool
Character kerning mode

pygame.freetype.Font.vertical
 vertical -> bool
Font vertical mode

pygame.freetype.Font.rotation
 rotation -> int
text rotation in degrees counterclockwise

pygame.freetype.Font.fgcolor
 fgcolor -> Color
default foreground color

pygame.freetype.Font.origin
 origin -> bool
Font render to text origin mode

pygame.freetype.Font.pad
 pad -> bool
padded boundary mode

pygame.freetype.Font.ucs4
 ucs4 -> bool
Enable UCS-4 mode

pygame.freetype.Font.resolution
 resolution -> int
Pixel resolution in dots per inch

*/