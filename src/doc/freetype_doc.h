/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMEFREETYPE "Enhanced Pygame module for loading and rendering computer fonts"

#define DOC_PYGAMEFREETYPEGETERROR "get_error() -> str\nGet the latest error"

#define DOC_PYGAMEFREETYPEGETVERSION "get_version() -> (int, int, int)\nGet the FreeType version"

#define DOC_PYGAMEFREETYPEINIT "init(cache_size=64, resolution=72) -> None\nInitialize the underlying FreeType 2 library."

#define DOC_PYGAMEFREETYPEQUIT "quit() -> None\nShuts down the underlying FreeType 2 library."

#define DOC_PYGAMEFREETYPEWASINIT "was_init() -> bool\nReturns whether the the FreeType 2 library is initialized."

#define DOC_PYGAMEFREETYPEGETDEFAULTRESOLUTION "get_default_resolution() -> long\nReturn the default pixel size in dots per inch"

#define DOC_PYGAMEFREETYPESETDEFAULTRESOLUTION "set_default_resolution([resolution]) -> None\nSet the default pixel size in dots per inch for the module"

#define DOC_PYGAMEFREETYPEGETDEFAULTFONT "get_default_font() -> string\nGet the filename of the default font"

#define DOC_PYGAMEFREETYPEFONT "Font(file, style=STYLE_NONE, ptsize=-1, font_index=0, vertical=0, ucs4=0, resolution=0) -> Font\nCreates a new Font instance from a supported font file."

#define DOC_FONTNAME "name -> string\nGets the name of the font."

#define DOC_FONTPATH "path -> unicode\nGets the path of the font file"

#define DOC_FONTGETRECT "get_rect(text, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> rect\nGets the size and offset of rendered text"

#define DOC_FONTGETMETRICS "get_metrics(text, ptsize=default) -> [(...), ...]\nGets glyph metrics for the font's characters"

#define DOC_FONTHEIGHT "height -> int\nGets the unscaled height of the font in font units"

#define DOC_FONTASCENDER "ascender -> int\nget the unscaled ascent of the font in font units"

#define DOC_FONTDESCENDER "descender -> int\nget the unscaled descent of the font in font units"

#define DOC_FONTGETSIZEDASCENDER "get_sized_ascender() -> int\nGets the scaled ascent the font in pixels"

#define DOC_FONTGETSIZEDDESCENDER "get_sized_descender() -> int\nGets the scaled descent the font in pixels"

#define DOC_FONTGETSIZEDHEIGHT "get_sized_height() -> int\nGets the scaled height of the font in pixels"

#define DOC_FONTGETSIZEDGLYPHHEIGHT "get_sized_glyph_height() -> int\nGets the scaled height of the font in pixels"

#define DOC_FONTRENDER "render(text, fgcolor, bgcolor=None, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> (Surface, Rect)\nRenders text on a surface"

#define DOC_FONTRENDERTO "render(surf, dest, text, fgcolor, bgcolor=None, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> Rect\nRenders text to an existing surface"

#define DOC_FONTRENDERRAW "render_raw(text, style=STYLE_DEFAULT, rotation=0, ptsize=default, invert=False) -> (bytes, (int, int))\nRenders text as a string of bytes"

#define DOC_FONTRENDERRAWTO "render_raw_to(array, text, dest=None, style=STYLE_DEFAULT, rotation=0, ptsize=default, invert=False) -> (int, int)\nRenders text as a string of ints to an array"

#define DOC_FONTSTYLE "style -> int\nGets or sets the font's style"

#define DOC_FONTUNDERLINE "underline -> bool\nGets or sets the font's underline style"

#define DOC_FONTSTRONG "strong -> bool\nGets or sets the font's strong style"

#define DOC_FONTOBLIQUE "oblique -> bool\nGets or sets the font's oblique style"

#define DOC_FONTWIDE "wide -> bool\nGets or sets the font's wide style"

#define DOC_FONTSTRENGTH "strength -> float\nGets or sets the strength of the strong or wide styles"

#define DOC_FONTUNDERLINEADJUSTMENT "underline_adjustment -> float\nGets or sets an adjustment factor for the underline position"

#define DOC_FONTFIXEDWIDTH "fixed_width -> bool\nGets whether the font is fixed-width"

#define DOC_FONTANTIALIASED "antialiased -> bool\nFont antialiasing mode"

#define DOC_FONTKERNING "kerning -> bool\nCharacter kerning mode"

#define DOC_FONTVERTICAL "vertical -> bool\nFont vertical mode"

#define DOC_FONTORIGIN "vertical -> bool\nFont render to text origin mode"

#define DOC_FONTPAD "pad -> bool\npadded boundary mode"

#define DOC_FONTUCS4 "ucs4 -> bool\nEnables UCS-4 mode"

#define DOC_FONTRESOLUTION "resolution -> int\nOutput pixel resolution in dots per inch"



/* Docs in a comment... slightly easier to read. */

/*

pygame.freetype
Enhanced Pygame module for loading and rendering computer fonts

pygame.freetype.get_error
 get_error() -> str
Get the latest error

pygame.freetype.get_version
 get_version() -> (int, int, int)
Get the FreeType version

pygame.freetype.init
 init(cache_size=64, resolution=72) -> None
Initialize the underlying FreeType 2 library.

pygame.freetype.quit
 quit() -> None
Shuts down the underlying FreeType 2 library.

pygame.freetype.was_init
 was_init() -> bool
Returns whether the the FreeType 2 library is initialized.

pygame.freetype.get_default_resolution
 get_default_resolution() -> long
Return the default pixel size in dots per inch

pygame.freetype.set_default_resolution
 set_default_resolution([resolution]) -> None
Set the default pixel size in dots per inch for the module

pygame.freetype.get_default_font
 get_default_font() -> string
Get the filename of the default font

pygame.freetype.Font
 Font(file, style=STYLE_NONE, ptsize=-1, font_index=0, vertical=0, ucs4=0, resolution=0) -> Font
Creates a new Font instance from a supported font file.

pygame.freetype.Font.name
 name -> string
Gets the name of the font.

pygame.freetype.Font.path
 path -> unicode
Gets the path of the font file

pygame.freetype.Font.get_rect
 get_rect(text, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> rect
Gets the size and offset of rendered text

pygame.freetype.Font.get_metrics
 get_metrics(text, ptsize=default) -> [(...), ...]
Gets glyph metrics for the font's characters

pygame.freetype.Font.height
 height -> int
Gets the unscaled height of the font in font units

pygame.freetype.Font.ascender
 ascender -> int
get the unscaled ascent of the font in font units

pygame.freetype.Font.descender
 descender -> int
get the unscaled descent of the font in font units

pygame.freetype.Font.get_sized_ascender
 get_sized_ascender() -> int
Gets the scaled ascent the font in pixels

pygame.freetype.Font.get_sized_descender
 get_sized_descender() -> int
Gets the scaled descent the font in pixels

pygame.freetype.Font.get_sized_height
 get_sized_height() -> int
Gets the scaled height of the font in pixels

pygame.freetype.Font.get_sized_glyph_height
 get_sized_glyph_height() -> int
Gets the scaled height of the font in pixels

pygame.freetype.Font.render
 render(text, fgcolor, bgcolor=None, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> (Surface, Rect)
Renders text on a surface

pygame.freetype.Font.render_to
 render(surf, dest, text, fgcolor, bgcolor=None, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> Rect
Renders text to an existing surface

pygame.freetype.Font.render_raw
 render_raw(text, style=STYLE_DEFAULT, rotation=0, ptsize=default, invert=False) -> (bytes, (int, int))
Renders text as a string of bytes

pygame.freetype.Font.render_raw_to
 render_raw_to(array, text, dest=None, style=STYLE_DEFAULT, rotation=0, ptsize=default, invert=False) -> (int, int)
Renders text as a string of ints to an array

pygame.freetype.Font.style
 style -> int
Gets or sets the font's style

pygame.freetype.Font.underline
 underline -> bool
Gets or sets the font's underline style

pygame.freetype.Font.strong
 strong -> bool
Gets or sets the font's strong style

pygame.freetype.Font.oblique
 oblique -> bool
Gets or sets the font's oblique style

pygame.freetype.Font.wide
 wide -> bool
Gets or sets the font's wide style

pygame.freetype.Font.strength
 strength -> float
Gets or sets the strength of the strong or wide styles

pygame.freetype.Font.underline_adjustment
 underline_adjustment -> float
Gets or sets an adjustment factor for the underline position

pygame.freetype.Font.fixed_width
 fixed_width -> bool
Gets whether the font is fixed-width

pygame.freetype.Font.antialiased
 antialiased -> bool
Font antialiasing mode

pygame.freetype.Font.kerning
 kerning -> bool
Character kerning mode

pygame.freetype.Font.vertical
 vertical -> bool
Font vertical mode

pygame.freetype.Font.origin
 vertical -> bool
Font render to text origin mode

pygame.freetype.Font.pad
 pad -> bool
padded boundary mode

pygame.freetype.Font.ucs4
 ucs4 -> bool
Enables UCS-4 mode

pygame.freetype.Font.resolution
 resolution -> int
Output pixel resolution in dots per inch

*/