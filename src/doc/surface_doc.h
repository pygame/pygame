/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMESURFACE "pygame.Surface((width, height), flags=0, depth=0, masks=None): return Surface\npygame.Surface((width, height), flags=0, Surface): return Surface\npygame object for representing images"

#define DOC_SURFACEBLIT "Surface.blit(source, dest, area=None, special_flags = 0): return Rect\ndraw one image onto another"

#define DOC_SURFACECONVERT "Surface.convert(Surface): return Surface\nSurface.convert(depth, flags=0): return Surface\nSurface.convert(masks, flags=0): return Surface\nSurface.convert(): return Surface\nchange the pixel format of an image"

#define DOC_SURFACECONVERTALPHA "Surface.convert_alpha(Surface): return Surface\nSurface.convert_alpha(): return Surface\nchange the pixel format of an image including per pixel alphas"

#define DOC_SURFACECOPY "Surface.copy(): return Surface\ncreate a new copy of a Surface"

#define DOC_SURFACEFILL "Surface.fill(color, rect=None, special_flags=0): return Rect\nfill Surface with a solid color"

#define DOC_SURFACESCROLL "Surface.scroll(dx=0, dy=0): return None\nShift the surface image in place"

#define DOC_SURFACESETCOLORKEY "Surface.set_colorkey(Color, flags=0): return None\nSurface.set_colorkey(None): return None\nSet the transparent colorkey"

#define DOC_SURFACEGETCOLORKEY "Surface.get_colorkey(): return RGB or None\nGet the current transparent colorkey"

#define DOC_SURFACESETALPHA "Surface.set_alpha(value, flags=0): return None\nSurface.set_alpha(None): return None\nset the alpha value for the full Surface image"

#define DOC_SURFACEGETALPHA "Surface.get_alpha(): return int_value or None\nget the current Surface transparency value"

#define DOC_SURFACELOCK "Surface.lock(): return None\nlock the Surface memory for pixel access"

#define DOC_SURFACEUNLOCK "Surface.unlock(): return None\nunlock the Surface memory from pixel access"

#define DOC_SURFACEMUSTLOCK "Surface.mustlock(): return bool\ntest if the Surface requires locking"

#define DOC_SURFACEGETLOCKED "Surface.get_locked(): return bool\ntest if the Surface is current locked"

#define DOC_SURFACEGETLOCKS "Surface.get_locks(): return tuple\nGets the locks for the Surface"

#define DOC_SURFACEGETAT "Surface.get_at((x, y)): return Color\nget the color value at a single pixel"

#define DOC_SURFACESETAT "Surface.set_at((x, y), Color): return None\nset the color value for a single pixel"

#define DOC_SURFACEGETATMAPPED "Surface.get_at((x, y)): return Color\nget the mapped color value at a single pixel"

#define DOC_SURFACEGETPALETTE "Surface.get_palette(): return [RGB, RGB, RGB, ...]\nget the color index palette for an 8bit Surface"

#define DOC_SURFACEGETPALETTEAT "Surface.get_palette_at(index): return RGB\nget the color for a single entry in a palette"

#define DOC_SURFACESETPALETTE "Surface.set_palette([RGB, RGB, RGB, ...]): return None\nset the color palette for an 8bit Surface"

#define DOC_SURFACESETPALETTEAT "Surface.set_at(index, RGB): return None\nset the color for a single index in an 8bit Surface palette"

#define DOC_SURFACEMAPRGB "Surface.map_rgb(Color): return mapped_int\nconvert a color into a mapped color value"

#define DOC_SURFACEUNMAPRGB "Surface.map_rgb(mapped_int): return Color\nconvert a mapped integer color value into a Color"

#define DOC_SURFACESETCLIP "Surface.set_clip(rect): return None\nSurface.set_clip(None): return None\nset the current clipping area of the Surface"

#define DOC_SURFACEGETCLIP "Surface.get_clip(): return Rect\nget the current clipping area of the Surface"

#define DOC_SURFACESUBSURFACE "Surface.subsurface(Rect): return Surface\ncreate a new surface that references its parent"

#define DOC_SURFACEGETPARENT "Surface.get_parent(): return Surface\nfind the parent of a subsurface"

#define DOC_SURFACEGETABSPARENT "Surface.get_abs_parent(): return Surface\nfind the top level parent of a subsurface"

#define DOC_SURFACEGETOFFSET "Surface.get_offset(): return (x, y)\nfind the position of a child subsurface inside a parent"

#define DOC_SURFACEGETABSOFFSET "Surface.get_abs_offset(): return (x, y)\nfind the absolute position of a child subsurface inside its top level parent"

#define DOC_SURFACEGETSIZE "Surface.get_size(): return (width, height)\nget the dimensions of the Surface"

#define DOC_SURFACEGETWIDTH "Surface.get_width(): return width\nget the width of the Surface"

#define DOC_SURFACEGETHEIGHT "Surface.get_height(): return height\nget the height of the Surface"

#define DOC_SURFACEGETRECT "Surface.get_rect(**kwargs): return Rect\nget the rectangular area of the Surface"

#define DOC_SURFACEGETBITSIZE "Surface.get_bitsize(): return int\nget the bit depth of the Surface pixel format"

#define DOC_SURFACEGETBYTESIZE "Surface.get_bytesize(): return int\nget the bytes used per Surface pixel"

#define DOC_SURFACEGETFLAGS "Surface.get_flags(): return int\nget the additional flags used for the Surface"

#define DOC_SURFACEGETPITCH "Surface.get_pitch(): return int\nget the number of bytes used per Surface row"

#define DOC_SURFACEGETMASKS "Surface.get_masks(): return (R, G, B, A)\nthe bitmasks needed to convert between a color and a mapped integer"

#define DOC_SURFACESETMASKS "Surface.set_masks((r,g,b,a)): return None\nset the bitmasks needed to convert between a color and a mapped integer"

#define DOC_SURFACEGETSHIFTS "Surface.get_shifts(): return (R, G, B, A)\nthe bit shifts needed to convert between a color and a mapped integer"

#define DOC_SURFACESETSHIFTS "Surface.get_shifts((r,g,b,a)): return None\nsets the bit shifts needed to convert between a color and a mapped integer"

#define DOC_SURFACEGETLOSSES "Surface.get_losses(): return (R, G, B, A)\nthe significant bits used to convert between a color and a mapped integer"

#define DOC_SURFACEGETBOUNDINGRECT "Surface.get_bounding_rect(min_alpha = 1): return Rect\nfind the smallest rect containing data"

#define DOC_SURFACEGETVIEW "Surface.get_view(kind='2'): return <view>\nreturn a view of a surface's pixel data."

#define DOC_SURFACEGETBUFFER "Surface.get_buffer(): return BufferProxy\nacquires a buffer object for the pixels of the Surface."



/* Docs in a comments... slightly easier to read. */


/*

pygame.Surface
 pygame.Surface((width, height), flags=0, depth=0, masks=None): return Surface
pygame.Surface((width, height), flags=0, Surface): return Surface
pygame object for representing images



Surface.blit
 Surface.blit(source, dest, area=None, special_flags = 0): return Rect
draw one image onto another



Surface.convert
 Surface.convert(Surface): return Surface
Surface.convert(depth, flags=0): return Surface
Surface.convert(masks, flags=0): return Surface
Surface.convert(): return Surface
change the pixel format of an image



Surface.convert_alpha
 Surface.convert_alpha(Surface): return Surface
Surface.convert_alpha(): return Surface
change the pixel format of an image including per pixel alphas



Surface.copy
 Surface.copy(): return Surface
create a new copy of a Surface



Surface.fill
 Surface.fill(color, rect=None, special_flags=0): return Rect
fill Surface with a solid color



Surface.scroll
 Surface.scroll(dx=0, dy=0): return None
Shift the surface image in place



Surface.set_colorkey
 Surface.set_colorkey(Color, flags=0): return None
Surface.set_colorkey(None): return None
Set the transparent colorkey



Surface.get_colorkey
 Surface.get_colorkey(): return RGB or None
Get the current transparent colorkey



Surface.set_alpha
 Surface.set_alpha(value, flags=0): return None
Surface.set_alpha(None): return None
set the alpha value for the full Surface image



Surface.get_alpha
 Surface.get_alpha(): return int_value or None
get the current Surface transparency value



Surface.lock
 Surface.lock(): return None
lock the Surface memory for pixel access



Surface.unlock
 Surface.unlock(): return None
unlock the Surface memory from pixel access



Surface.mustlock
 Surface.mustlock(): return bool
test if the Surface requires locking



Surface.get_locked
 Surface.get_locked(): return bool
test if the Surface is current locked



Surface.get_locks
 Surface.get_locks(): return tuple
Gets the locks for the Surface



Surface.get_at
 Surface.get_at((x, y)): return Color
get the color value at a single pixel



Surface.set_at
 Surface.set_at((x, y), Color): return None
set the color value for a single pixel



Surface.get_at_mapped
 Surface.get_at((x, y)): return Color
get the mapped color value at a single pixel



Surface.get_palette
 Surface.get_palette(): return [RGB, RGB, RGB, ...]
get the color index palette for an 8bit Surface



Surface.get_palette_at
 Surface.get_palette_at(index): return RGB
get the color for a single entry in a palette



Surface.set_palette
 Surface.set_palette([RGB, RGB, RGB, ...]): return None
set the color palette for an 8bit Surface



Surface.set_palette_at
 Surface.set_at(index, RGB): return None
set the color for a single index in an 8bit Surface palette



Surface.map_rgb
 Surface.map_rgb(Color): return mapped_int
convert a color into a mapped color value



Surface.unmap_rgb
 Surface.map_rgb(mapped_int): return Color
convert a mapped integer color value into a Color



Surface.set_clip
 Surface.set_clip(rect): return None
Surface.set_clip(None): return None
set the current clipping area of the Surface



Surface.get_clip
 Surface.get_clip(): return Rect
get the current clipping area of the Surface



Surface.subsurface
 Surface.subsurface(Rect): return Surface
create a new surface that references its parent



Surface.get_parent
 Surface.get_parent(): return Surface
find the parent of a subsurface



Surface.get_abs_parent
 Surface.get_abs_parent(): return Surface
find the top level parent of a subsurface



Surface.get_offset
 Surface.get_offset(): return (x, y)
find the position of a child subsurface inside a parent



Surface.get_abs_offset
 Surface.get_abs_offset(): return (x, y)
find the absolute position of a child subsurface inside its top level parent



Surface.get_size
 Surface.get_size(): return (width, height)
get the dimensions of the Surface



Surface.get_width
 Surface.get_width(): return width
get the width of the Surface



Surface.get_height
 Surface.get_height(): return height
get the height of the Surface



Surface.get_rect
 Surface.get_rect(**kwargs): return Rect
get the rectangular area of the Surface



Surface.get_bitsize
 Surface.get_bitsize(): return int
get the bit depth of the Surface pixel format



Surface.get_bytesize
 Surface.get_bytesize(): return int
get the bytes used per Surface pixel



Surface.get_flags
 Surface.get_flags(): return int
get the additional flags used for the Surface



Surface.get_pitch
 Surface.get_pitch(): return int
get the number of bytes used per Surface row



Surface.get_masks
 Surface.get_masks(): return (R, G, B, A)
the bitmasks needed to convert between a color and a mapped integer



Surface.set_masks
 Surface.set_masks((r,g,b,a)): return None
set the bitmasks needed to convert between a color and a mapped integer



Surface.get_shifts
 Surface.get_shifts(): return (R, G, B, A)
the bit shifts needed to convert between a color and a mapped integer



Surface.set_shifts
 Surface.get_shifts((r,g,b,a)): return None
sets the bit shifts needed to convert between a color and a mapped integer



Surface.get_losses
 Surface.get_losses(): return (R, G, B, A)
the significant bits used to convert between a color and a mapped integer



Surface.get_bounding_rect
 Surface.get_bounding_rect(min_alpha = 1): return Rect
find the smallest rect containing data



Surface.get_view
 Surface.get_view(kind='2'): return <view>
return a view of a surface's pixel data.



Surface.get_buffer
 Surface.get_buffer(): return BufferProxy
acquires a buffer object for the pixels of the Surface.



*/

