/* Auto generated file: with makeref.py .  Docs go in docs/reST/ref/ . */
#define DOC_PYGAMETRANSFORM "pygame module to transform surfaces"
#define DOC_PYGAMETRANSFORMFLIP "flip(surface, flip_x, flip_y) -> Surface\nflip vertically and horizontally"
#define DOC_PYGAMETRANSFORMSCALE "scale(surface, size, dest_surface=None) -> Surface\nresize to new resolution"
#define DOC_PYGAMETRANSFORMSCALEBY "scale_by(surface, factor, dest_surface=None) -> Surface\nresize to new resolution, using scalar(s)"
#define DOC_PYGAMETRANSFORMROTATE "rotate(surface, angle) -> Surface\nrotate an image"
#define DOC_PYGAMETRANSFORMROTOZOOM "rotozoom(surface, angle, scale) -> Surface\nfiltered scale and rotation"
#define DOC_PYGAMETRANSFORMSCALE2X "scale2x(surface, dest_surface=None) -> Surface\nspecialized image doubler"
#define DOC_PYGAMETRANSFORMSMOOTHSCALE "smoothscale(surface, size, dest_surface=None) -> Surface\nscale a surface to an arbitrary size smoothly"
#define DOC_PYGAMETRANSFORMSMOOTHSCALEBY "smoothscale_by(surface, factor, dest_surface=None) -> Surface\nresize to new resolution, using scalar(s)"
#define DOC_PYGAMETRANSFORMGETSMOOTHSCALEBACKEND "get_smoothscale_backend() -> string\nreturn smoothscale filter version in use: 'GENERIC', 'MMX', or 'SSE'"
#define DOC_PYGAMETRANSFORMSETSMOOTHSCALEBACKEND "set_smoothscale_backend(backend) -> None\nset smoothscale filter version to one of: 'GENERIC', 'MMX', or 'SSE'"
#define DOC_PYGAMETRANSFORMCHOP "chop(surface, rect) -> Surface\ngets a copy of an image with an interior area removed"
#define DOC_PYGAMETRANSFORMLAPLACIAN "laplacian(surface, dest_surface=None) -> Surface\nfind edges in a surface"
#define DOC_PYGAMETRANSFORMAVERAGESURFACES "average_surfaces(surfaces, dest_surface=None, palette_colors=1) -> Surface\nfind the average surface from many surfaces."
#define DOC_PYGAMETRANSFORMAVERAGECOLOR "average_color(surface, rect=None, consider_alpha=False) -> Color\nfinds the average color of a surface"
#define DOC_PYGAMETRANSFORMGRAYSCALE "grayscale(surface, dest_surface=None) -> Surface\ngrayscale a surface"
#define DOC_PYGAMETRANSFORMTHRESHOLD "threshold(dest_surface, surface, search_color, threshold=(0,0,0,0), set_color=(0,0,0,0), set_behavior=1, search_surf=None, inverse_set=False) -> num_threshold_pixels\nfinds which, and how many pixels in a surface are within a threshold of a 'search_color' or a 'search_surf'."


/* Docs in a comment... slightly easier to read. */

/*

pygame.transform
pygame module to transform surfaces

pygame.transform.flip
 flip(surface, flip_x, flip_y) -> Surface
flip vertically and horizontally

pygame.transform.scale
 scale(surface, size, dest_surface=None) -> Surface
resize to new resolution

pygame.transform.scale_by
 scale_by(surface, factor, dest_surface=None) -> Surface
resize to new resolution, using scalar(s)

pygame.transform.rotate
 rotate(surface, angle) -> Surface
rotate an image

pygame.transform.rotozoom
 rotozoom(surface, angle, scale) -> Surface
filtered scale and rotation

pygame.transform.scale2x
 scale2x(surface, dest_surface=None) -> Surface
specialized image doubler

pygame.transform.smoothscale
 smoothscale(surface, size, dest_surface=None) -> Surface
scale a surface to an arbitrary size smoothly

pygame.transform.smoothscale_by
 smoothscale_by(surface, factor, dest_surface=None) -> Surface
resize to new resolution, using scalar(s)

pygame.transform.get_smoothscale_backend
 get_smoothscale_backend() -> string
return smoothscale filter version in use: 'GENERIC', 'MMX', or 'SSE'

pygame.transform.set_smoothscale_backend
 set_smoothscale_backend(backend) -> None
set smoothscale filter version to one of: 'GENERIC', 'MMX', or 'SSE'

pygame.transform.chop
 chop(surface, rect) -> Surface
gets a copy of an image with an interior area removed

pygame.transform.laplacian
 laplacian(surface, dest_surface=None) -> Surface
find edges in a surface

pygame.transform.average_surfaces
 average_surfaces(surfaces, dest_surface=None, palette_colors=1) -> Surface
find the average surface from many surfaces.

pygame.transform.average_color
 average_color(surface, rect=None, consider_alpha=False) -> Color
finds the average color of a surface

pygame.transform.grayscale
 grayscale(surface, dest_surface=None) -> Surface
grayscale a surface

pygame.transform.threshold
 threshold(dest_surface, surface, search_color, threshold=(0,0,0,0), set_color=(0,0,0,0), set_behavior=1, search_surf=None, inverse_set=False) -> num_threshold_pixels
finds which, and how many pixels in a surface are within a threshold of a 'search_color' or a 'search_surf'.

*/