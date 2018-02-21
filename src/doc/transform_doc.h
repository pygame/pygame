/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMETRANSFORM "pygame module to transform surfaces"

#define DOC_PYGAMETRANSFORMFLIP "flip(Surface, xbool, ybool) -> Surface\nflip vertically and horizontally"

#define DOC_PYGAMETRANSFORMSCALE "scale(Surface, (width, height), DestSurface = None) -> Surface\nresize to new resolution"

#define DOC_PYGAMETRANSFORMROTATE "rotate(Surface, angle) -> Surface\nrotate an image"

#define DOC_PYGAMETRANSFORMROTOZOOM "rotozoom(Surface, angle, scale) -> Surface\nfiltered scale and rotation"

#define DOC_PYGAMETRANSFORMSCALE2X "scale2x(Surface, DestSurface = None) -> Surface\nspecialized image doubler"

#define DOC_PYGAMETRANSFORMSMOOTHSCALE "smoothscale(Surface, (width, height), DestSurface = None) -> Surface\nscale a surface to an arbitrary size smoothly"

#define DOC_PYGAMETRANSFORMGETSMOOTHSCALEBACKEND "get_smoothscale_backend() -> String\nreturn smoothscale filter version in use: ‘GENERIC’, ‘MMX’, or ‘SSE’"

#define DOC_PYGAMETRANSFORMSETSMOOTHSCALEBACKEND "set_smoothscale_backend(type) -> None\nset smoothscale filter version to one of: ‘GENERIC’, ‘MMX’, or ‘SSE’"

#define DOC_PYGAMETRANSFORMCHOP "chop(Surface, rect) -> Surface\ngets a copy of an image with an interior area removed"

#define DOC_PYGAMETRANSFORMLAPLACIAN "laplacian(Surface, DestSurface = None) -> Surface\nfind edges in a surface"

#define DOC_PYGAMETRANSFORMAVERAGESURFACES "average_surfaces(Surfaces, DestSurface = None, palette_colors = 1) -> Surface\nfind the average surface from many surfaces."

#define DOC_PYGAMETRANSFORMAVERAGECOLOR "average_color(Surface, Rect = None) -> Color\nfinds the average color of a surface"

#define DOC_PYGAMETRANSFORMTHRESHOLD "threshold(DestSurface, Surface, color, threshold = (0,0,0,0), diff_color = (0,0,0,0), change_return = 1, Surface = None, inverse = False) -> num_threshold_pixels\nfinds which, and how many pixels in a surface are within a threshold of a color."



/* Docs in a comment... slightly easier to read. */

/*

pygame.transform
pygame module to transform surfaces

pygame.transform.flip
 flip(Surface, xbool, ybool) -> Surface
flip vertically and horizontally

pygame.transform.scale
 scale(Surface, (width, height), DestSurface = None) -> Surface
resize to new resolution

pygame.transform.rotate
 rotate(Surface, angle) -> Surface
rotate an image

pygame.transform.rotozoom
 rotozoom(Surface, angle, scale) -> Surface
filtered scale and rotation

pygame.transform.scale2x
 scale2x(Surface, DestSurface = None) -> Surface
specialized image doubler

pygame.transform.smoothscale
 smoothscale(Surface, (width, height), DestSurface = None) -> Surface
scale a surface to an arbitrary size smoothly

pygame.transform.get_smoothscale_backend
 get_smoothscale_backend() -> String
return smoothscale filter version in use: ‘GENERIC’, ‘MMX’, or ‘SSE’

pygame.transform.set_smoothscale_backend
 set_smoothscale_backend(type) -> None
set smoothscale filter version to one of: ‘GENERIC’, ‘MMX’, or ‘SSE’

pygame.transform.chop
 chop(Surface, rect) -> Surface
gets a copy of an image with an interior area removed

pygame.transform.laplacian
 laplacian(Surface, DestSurface = None) -> Surface
find edges in a surface

pygame.transform.average_surfaces
 average_surfaces(Surfaces, DestSurface = None, palette_colors = 1) -> Surface
find the average surface from many surfaces.

pygame.transform.average_color
 average_color(Surface, Rect = None) -> Color
finds the average color of a surface

pygame.transform.threshold
 threshold(DestSurface, Surface, color, threshold = (0,0,0,0), diff_color = (0,0,0,0), change_return = 1, Surface = None, inverse = False) -> num_threshold_pixels
finds which, and how many pixels in a surface are within a threshold of a color.

*/