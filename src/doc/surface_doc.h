/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMESURFACE "Surface((width, height), flags=0, depth=0, masks=None) -> Surface\nSurface((width, height), flags=0, Surface) -> Surface\npygame object for representing images"

#define DOC_SURFACEBLIT "blit(source, dest, area=None, special_flags = 0) -> Rect\ndraw one image onto another"

#define DOC_SURFACEBLITS "blit((source, dest), ...)) -> (Rect, ...)\nblit((source, dest, area), ...)) -> (Rect, ...)\nblit((source, dest, area, special_flags), ...)) -> (Rect, ...)\ndraw many images onto another"

#define DOC_SURFACECONVERT "convert(Surface) -> Surface\nconvert(depth, flags=0) -> Surface\nconvert(masks, flags=0) -> Surface\nconvert() -> Surface\nchange the pixel format of an image"

#define DOC_SURFACECONVERTALPHA "convert_alpha(Surface) -> Surface\nconvert_alpha() -> Surface\nchange the pixel format of an image including per pixel alphas"

#define DOC_SURFACECOPY "copy() -> Surface\ncreate a new copy of a Surface"

#define DOC_SURFACEFILL "fill(color, rect=None, special_flags=0) -> Rect\nfill Surface with a solid color"

#define DOC_SURFACESCROLL "scroll(dx=0, dy=0) -> None\nShift the surface image in place"

#define DOC_SURFACESETCOLORKEY "set_colorkey(Color, flags=0) -> None\nset_colorkey(None) -> None\nSet the transparent colorkey"

#define DOC_SURFACEGETCOLORKEY "get_colorkey() -> RGB or None\nGet the current transparent colorkey"

#define DOC_SURFACESETALPHA "set_alpha(value, flags=0) -> None\nset_alpha(None) -> None\nset the alpha value for the full Surface image"

#define DOC_SURFACEGETALPHA "get_alpha() -> int_value or None\nget the current Surface transparency value"

#define DOC_SURFACELOCK "lock() -> None\nlock the Surface memory for pixel access"

#define DOC_SURFACEUNLOCK "unlock() -> None\nunlock the Surface memory from pixel access"

#define DOC_SURFACEMUSTLOCK "mustlock() -> bool\ntest if the Surface requires locking"

#define DOC_SURFACEGETLOCKED "get_locked() -> bool\ntest if the Surface is current locked"

#define DOC_SURFACEGETLOCKS "get_locks() -> tuple\nGets the locks for the Surface"

#define DOC_SURFACEGETAT "get_at((x, y)) -> Color\nget the color value at a single pixel"

#define DOC_SURFACESETAT "set_at((x, y), Color) -> None\nset the color value for a single pixel"

#define DOC_SURFACEGETATMAPPED "get_at_mapped((x, y)) -> Color\nget the mapped color value at a single pixel"

#define DOC_SURFACEGETPALETTE "get_palette() -> [RGB, RGB, RGB, ...]\nget the color index palette for an 8-bit Surface"

#define DOC_SURFACEGETPALETTEAT "get_palette_at(index) -> RGB\nget the color for a single entry in a palette"

#define DOC_SURFACESETPALETTE "set_palette([RGB, RGB, RGB, ...]) -> None\nset the color palette for an 8-bit Surface"

#define DOC_SURFACESETPALETTEAT "set_palette_at(index, RGB) -> None\nset the color for a single index in an 8-bit Surface palette"

#define DOC_SURFACEMAPRGB "map_rgb(Color) -> mapped_int\nconvert a color into a mapped color value"

#define DOC_SURFACEUNMAPRGB "unmap_rgb(mapped_int) -> Color\nconvert a mapped integer color value into a Color"

#define DOC_SURFACESETCLIP "set_clip(rect) -> None\nset_clip(None) -> None\nset the current clipping area of the Surface"

#define DOC_SURFACEGETCLIP "get_clip() -> Rect\nget the current clipping area of the Surface"

#define DOC_SURFACESUBSURFACE "subsurface(Rect) -> Surface\ncreate a new surface that references its parent"

#define DOC_SURFACEGETPARENT "get_parent() -> Surface\nfind the parent of a subsurface"

#define DOC_SURFACEGETABSPARENT "get_abs_parent() -> Surface\nfind the top level parent of a subsurface"

#define DOC_SURFACEGETOFFSET "get_offset() -> (x, y)\nfind the position of a child subsurface inside a parent"

#define DOC_SURFACEGETABSOFFSET "get_abs_offset() -> (x, y)\nfind the absolute position of a child subsurface inside its top level parent"

#define DOC_SURFACEGETSIZE "get_size() -> (width, height)\nget the dimensions of the Surface"

#define DOC_SURFACEGETWIDTH "get_width() -> width\nget the width of the Surface"

#define DOC_SURFACEGETHEIGHT "get_height() -> height\nget the height of the Surface"

#define DOC_SURFACEGETRECT "get_rect(**kwargs) -> Rect\nget the rectangular area of the Surface"

#define DOC_SURFACEGETBITSIZE "get_bitsize() -> int\nget the bit depth of the Surface pixel format"

#define DOC_SURFACEGETBYTESIZE "get_bytesize() -> int\nget the bytes used per Surface pixel"

#define DOC_SURFACEGETFLAGS "get_flags() -> int\nget the additional flags used for the Surface"

#define DOC_SURFACEGETPITCH "get_pitch() -> int\nget the number of bytes used per Surface row"

#define DOC_SURFACEGETMASKS "get_masks() -> (R, G, B, A)\nthe bitmasks needed to convert between a color and a mapped integer"

#define DOC_SURFACESETMASKS "set_masks((r,g,b,a)) -> None\nset the bitmasks needed to convert between a color and a mapped integer"

#define DOC_SURFACEGETSHIFTS "get_shifts() -> (R, G, B, A)\nthe bit shifts needed to convert between a color and a mapped integer"

#define DOC_SURFACESETSHIFTS "set_shifts((r,g,b,a)) -> None\nsets the bit shifts needed to convert between a color and a mapped integer"

#define DOC_SURFACEGETLOSSES "get_losses() -> (R, G, B, A)\nthe significant bits used to convert between a color and a mapped integer"

#define DOC_SURFACEGETBOUNDINGRECT "get_bounding_rect(min_alpha = 1) -> Rect\nfind the smallest rect containing data"

#define DOC_SURFACEGETVIEW "get_view(<kind>='2') -> BufferProxy\nreturn a buffer view of the Surface's pixels."

#define DOC_SURFACEGETBUFFER "get_buffer() -> BufferProxy\nacquires a buffer object for the pixels of the Surface."

#define DOC_SURFACEPIXELSADDRESS "_pixels_address -> int\npixel buffer address"



/* Docs in a comment... slightly easier to read. */

/*

pygame.Surface
 Surface((width, height), flags=0, depth=0, masks=None) -> Surface
 Surface((width, height), flags=0, Surface) -> Surface
pygame object for representing images

pygame.Surface.blit
 blit(source, dest, area=None, special_flags = 0) -> Rect
draw one image onto another

pygame.Surface.blits
 blit((source, dest), ...)) -> (Rect, ...)
 blit((source, dest, area), ...)) -> (Rect, ...)
 blit((source, dest, area, special_flags), ...)) -> (Rect, ...)
draw many images onto another

pygame.Surface.convert
 convert(Surface) -> Surface
 convert(depth, flags=0) -> Surface
 convert(masks, flags=0) -> Surface
 convert() -> Surface
change the pixel format of an image

pygame.Surface.convert_alpha
 convert_alpha(Surface) -> Surface
 convert_alpha() -> Surface
change the pixel format of an image including per pixel alphas

pygame.Surface.copy
 copy() -> Surface
create a new copy of a Surface

pygame.Surface.fill
 fill(color, rect=None, special_flags=0) -> Rect
fill Surface with a solid color

pygame.Surface.scroll
 scroll(dx=0, dy=0) -> None
Shift the surface image in place

pygame.Surface.set_colorkey
 set_colorkey(Color, flags=0) -> None
 set_colorkey(None) -> None
Set the transparent colorkey

pygame.Surface.get_colorkey
 get_colorkey() -> RGB or None
Get the current transparent colorkey

pygame.Surface.set_alpha
 set_alpha(value, flags=0) -> None
 set_alpha(None) -> None
set the alpha value for the full Surface image

pygame.Surface.get_alpha
 get_alpha() -> int_value or None
get the current Surface transparency value

pygame.Surface.lock
 lock() -> None
lock the Surface memory for pixel access

pygame.Surface.unlock
 unlock() -> None
unlock the Surface memory from pixel access

pygame.Surface.mustlock
 mustlock() -> bool
test if the Surface requires locking

pygame.Surface.get_locked
 get_locked() -> bool
test if the Surface is current locked

pygame.Surface.get_locks
 get_locks() -> tuple
Gets the locks for the Surface

pygame.Surface.get_at
 get_at((x, y)) -> Color
get the color value at a single pixel

pygame.Surface.set_at
 set_at((x, y), Color) -> None
set the color value for a single pixel

pygame.Surface.get_at_mapped
 get_at_mapped((x, y)) -> Color
get the mapped color value at a single pixel

pygame.Surface.get_palette
 get_palette() -> [RGB, RGB, RGB, ...]
get the color index palette for an 8-bit Surface

pygame.Surface.get_palette_at
 get_palette_at(index) -> RGB
get the color for a single entry in a palette

pygame.Surface.set_palette
 set_palette([RGB, RGB, RGB, ...]) -> None
set the color palette for an 8-bit Surface

pygame.Surface.set_palette_at
 set_palette_at(index, RGB) -> None
set the color for a single index in an 8-bit Surface palette

pygame.Surface.map_rgb
 map_rgb(Color) -> mapped_int
convert a color into a mapped color value

pygame.Surface.unmap_rgb
 unmap_rgb(mapped_int) -> Color
convert a mapped integer color value into a Color

pygame.Surface.set_clip
 set_clip(rect) -> None
 set_clip(None) -> None
set the current clipping area of the Surface

pygame.Surface.get_clip
 get_clip() -> Rect
get the current clipping area of the Surface

pygame.Surface.subsurface
 subsurface(Rect) -> Surface
create a new surface that references its parent

pygame.Surface.get_parent
 get_parent() -> Surface
find the parent of a subsurface

pygame.Surface.get_abs_parent
 get_abs_parent() -> Surface
find the top level parent of a subsurface

pygame.Surface.get_offset
 get_offset() -> (x, y)
find the position of a child subsurface inside a parent

pygame.Surface.get_abs_offset
 get_abs_offset() -> (x, y)
find the absolute position of a child subsurface inside its top level parent

pygame.Surface.get_size
 get_size() -> (width, height)
get the dimensions of the Surface

pygame.Surface.get_width
 get_width() -> width
get the width of the Surface

pygame.Surface.get_height
 get_height() -> height
get the height of the Surface

pygame.Surface.get_rect
 get_rect(**kwargs) -> Rect
get the rectangular area of the Surface

pygame.Surface.get_bitsize
 get_bitsize() -> int
get the bit depth of the Surface pixel format

pygame.Surface.get_bytesize
 get_bytesize() -> int
get the bytes used per Surface pixel

pygame.Surface.get_flags
 get_flags() -> int
get the additional flags used for the Surface

pygame.Surface.get_pitch
 get_pitch() -> int
get the number of bytes used per Surface row

pygame.Surface.get_masks
 get_masks() -> (R, G, B, A)
the bitmasks needed to convert between a color and a mapped integer

pygame.Surface.set_masks
 set_masks((r,g,b,a)) -> None
set the bitmasks needed to convert between a color and a mapped integer

pygame.Surface.get_shifts
 get_shifts() -> (R, G, B, A)
the bit shifts needed to convert between a color and a mapped integer

pygame.Surface.set_shifts
 set_shifts((r,g,b,a)) -> None
sets the bit shifts needed to convert between a color and a mapped integer

pygame.Surface.get_losses
 get_losses() -> (R, G, B, A)
the significant bits used to convert between a color and a mapped integer

pygame.Surface.get_bounding_rect
 get_bounding_rect(min_alpha = 1) -> Rect
find the smallest rect containing data

pygame.Surface.get_view
 get_view(<kind>='2') -> BufferProxy
return a buffer view of the Surface's pixels.

pygame.Surface.get_buffer
 get_buffer() -> BufferProxy
acquires a buffer object for the pixels of the Surface.

pygame.Surface._pixels_address
 _pixels_address -> int
pixel buffer address

*/