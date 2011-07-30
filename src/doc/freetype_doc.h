/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMEFREETYPE "Enhanced Pygame module for loading and rendering font faces"

#define DOC_PYGAMEFREETYPEGETERROR "get_error() -> str\nGet the latest error"

#define DOC_PYGAMEFREETYPEGETVERSION "get_version() -> (int, int, int)\nGet the FreeType version"

#define DOC_PYGAMEFREETYPEINIT "init(cache_size=64, resolution=72) -> None\nInitialize the underlying FreeType 2 library."

#define DOC_PYGAMEFREETYPEQUIT "quit() -> None\nShuts down the underlying FreeType 2 library."

#define DOC_PYGAMEFREETYPEWASINIT "was_init() -> bool\nReturns whether the the FreeType 2 library is initialized."

#define DOC_PYGAMEFREETYPEGETDEFAULTRESOLUTION "get_default_resolution() -> long\nReturn the default pixel size in dots per inch"

#define DOC_PYGAMEFREETYPESETDEFAULTRESOLUTION "set_default_resolution([resolution]) -> None\nSet the default pixel size in dots per inch for the module"

#define DOC_PYGAMEFREETYPEGETDEFAULTFONT "get_default_font() -> string\nGet the filename of the default font"

#define DOC_PYGAMEFREETYPEFACE "Face(file, style=STYLE_NONE, ptsize=-1, face_index=0, vertical=0, ucs4=0, resolution=0) -> Face\nCreates a new Face instance from a supported font file."

#define DOC_FACENAME "name -> string\nGets the name of the font face."

#define DOC_FACEPATH "path -> unicode\nGets the path of the font file"

#define DOC_FACEGETRECT "get_rect(text, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> rect\nGets the size and offset of rendered text"

#define DOC_FACEGETMETRICS "get_metrics(text, ptsize=default) -> [(...), ...]\nGets glyph metrics for the face's characters"

#define DOC_FACEHEIGHT "height -> int\nGets the unscaled height of the face in font units"

#define DOC_FACEASCENDER "ascender -> int\nget the unscaled ascent of the face in font units"

#define DOC_FACEDESCENDER "descender -> int\nget the unscaled descent of the face in font units"

#define DOC_FACEGETSIZEDASCENDER "get_sized_ascender() -> int\nGets the scaled ascent the face in pixels"

#define DOC_FACEGETSIZEDDESCENDER "get_sized_descender() -> int\nGets the scaled descent the face in pixels"

#define DOC_FACEGETSIZEDHEIGHT "get_sized_height() -> int\nGets the scaled height of the face in pixels"

#define DOC_FACEGETSIZEDGLYPHHEIGHT "get_sized_glyph_height() -> int\nGets the scaled height of the face in pixels"

#define DOC_FACERENDER "render(dest, text, fgcolor, bgcolor=None, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> (Surface, Rect)\nRenders text on a surface"

#define DOC_FACERENDERRAW "render_raw(text, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> (bytes, (int, int))\nRenders text as a string of bytes"

#define DOC_FACESTYLE "style -> int\nGets or sets the face's style"

#define DOC_FACEUNDERLINE "underline -> bool\nGets or sets the face's underline style"

#define DOC_FACEUNDERSCORE "underscore -> bool\nGets or sets the face's underscore style"

#define DOC_FACESTRONG "strong -> bool\nGets or sets the face's strong style"

#define DOC_FACEOBLIQUE "oblique -> bool\nGets or sets the face's oblique style"

#define DOC_FACEWIDE "wide -> bool\nGets or sets the face's wide style"

#define DOC_FACESTRENGTH "strength -> float\nGets or sets the strength of the strong or wide styles"

#define DOC_FACEFIXEDWIDTH "fixed_width -> bool\nGets whether the face is fixed-width"

#define DOC_FACEANTIALIASED "antialiased -> bool\nFace antialiasing mode"

#define DOC_FACEKERNING "kerning -> bool\nCharacter kerning mode"

#define DOC_FACEVERTICAL "vertical -> bool\nFace vertical mode"

#define DOC_FACEORIGIN "vertical -> bool\nFace render to text origin mode"

#define DOC_FACEPAD "pad -> bool\npadded boundary mode"

#define DOC_FACEUCS4 "ucs4 -> bool\nEnables UCS-4 mode"

#define DOC_FACERESOLUTION "resolution -> int\nOutput pixel resolution in dots per inch"

#define DOC_FACESETTRANSFORM "set_transform(xx, xy, yx, yy) -> None\nassign a glyph transformation matrix"

#define DOC_FACEDELETETRANSFORM "set_transform(xx, xy, yx, yy) -> None\ndelete a glyph transformation matrix"

#define DOC_FACEGETTRANSFORM "get_transform() -> (double, double, double, double) or None\nreturn the user assigned transformation matrix, or None"



/* Docs in a comment... slightly easier to read. */

/*

pygame.freetype
Enhanced Pygame module for loading and rendering font faces

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

pygame.freetype.Face
 Face(file, style=STYLE_NONE, ptsize=-1, face_index=0, vertical=0, ucs4=0, resolution=0) -> Face
Creates a new Face instance from a supported font file.

pygame.freetype.Face.name
 name -> string
Gets the name of the font face.

pygame.freetype.Face.path
 path -> unicode
Gets the path of the font file

pygame.freetype.Face.get_rect
 get_rect(text, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> rect
Gets the size and offset of rendered text

pygame.freetype.Face.get_metrics
 get_metrics(text, ptsize=default) -> [(...), ...]
Gets glyph metrics for the face's characters

pygame.freetype.Face.height
 height -> int
Gets the unscaled height of the face in font units

pygame.freetype.Face.ascender
 ascender -> int
get the unscaled ascent of the face in font units

pygame.freetype.Face.descender
 descender -> int
get the unscaled descent of the face in font units

pygame.freetype.Face.get_sized_ascender
 get_sized_ascender() -> int
Gets the scaled ascent the face in pixels

pygame.freetype.Face.get_sized_descender
 get_sized_descender() -> int
Gets the scaled descent the face in pixels

pygame.freetype.Face.get_sized_height
 get_sized_height() -> int
Gets the scaled height of the face in pixels

pygame.freetype.Face.get_sized_glyph_height
 get_sized_glyph_height() -> int
Gets the scaled height of the face in pixels

pygame.freetype.Face.render
 render(dest, text, fgcolor, bgcolor=None, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> (Surface, Rect)
Renders text on a surface

pygame.freetype.Face.render_raw
 render_raw(text, style=STYLE_DEFAULT, rotation=0, ptsize=default) -> (bytes, (int, int))
Renders text as a string of bytes

pygame.freetype.Face.style
 style -> int
Gets or sets the face's style

pygame.freetype.Face.underline
 underline -> bool
Gets or sets the face's underline style

pygame.freetype.Face.underscore
 underscore -> bool
Gets or sets the face's underscore style

pygame.freetype.Face.strong
 strong -> bool
Gets or sets the face's strong style

pygame.freetype.Face.oblique
 oblique -> bool
Gets or sets the face's oblique style

pygame.freetype.Face.wide
 wide -> bool
Gets or sets the face's wide style

pygame.freetype.Face.strength
 strength -> float
Gets or sets the strength of the strong or wide styles

pygame.freetype.Face.fixed_width
 fixed_width -> bool
Gets whether the face is fixed-width

pygame.freetype.Face.antialiased
 antialiased -> bool
Face antialiasing mode

pygame.freetype.Face.kerning
 kerning -> bool
Character kerning mode

pygame.freetype.Face.vertical
 vertical -> bool
Face vertical mode

pygame.freetype.Face.origin
 vertical -> bool
Face render to text origin mode

pygame.freetype.Face.pad
 pad -> bool
padded boundary mode

pygame.freetype.Face.ucs4
 ucs4 -> bool
Enables UCS-4 mode

pygame.freetype.Face.resolution
 resolution -> int
Output pixel resolution in dots per inch

pygame.freetype.Face.set_transform
 set_transform(xx, xy, yx, yy) -> None
assign a glyph transformation matrix

pygame.freetype.Face.delete_transform
 set_transform(xx, xy, yx, yy) -> None
delete a glyph transformation matrix

pygame.freetype.Face.get_transform
 get_transform() -> (double, double, double, double) or None
return the user assigned transformation matrix, or None

*/