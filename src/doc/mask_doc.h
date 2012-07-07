/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMEMASK "pygame module for image masks."

#define DOC_PYGAMEMASKFROMSURFACE "from_surface(Surface, threshold = 127) -> Mask\nReturns a Mask from the given surface."

#define DOC_PYGAMEMASKFROMTHRESHOLD "from_threshold(Surface, color, threshold = (0,0,0,255), othersurface = None, palette_colors = 1) -> Mask\nCreates a mask by thresholding Surfaces"

#define DOC_PYGAMEMASKMASK "Mask((width, height)) -> Mask\npygame object for representing 2d bitmasks"

#define DOC_MASKGETSIZE "get_size() -> width,height\nReturns the size of the mask."

#define DOC_MASKGETAT "get_at((x,y)) -> int\nReturns nonzero if the bit at (x,y) is set."

#define DOC_MASKSETAT "set_at((x,y),value) -> None\nSets the position in the mask given by x and y."

#define DOC_MASKOVERLAP "overlap(othermask, offset) -> x,y\nReturns the point of intersection if the masks overlap with the given offset - or None if it does not overlap."

#define DOC_MASKOVERLAPAREA "overlap_area(othermask, offset) -> numpixels\nReturns the number of overlapping 'pixels'."

#define DOC_MASKOVERLAPMASK "overlap_mask(othermask, offset) -> Mask\nReturns a mask of the overlapping pixels"

#define DOC_MASKFILL "fill() -> None\nSets all bits to 1"

#define DOC_MASKCLEAR "clear() -> None\nSets all bits to 0"

#define DOC_MASKINVERT "invert() -> None\nFlips the bits in a Mask"

#define DOC_MASKSCALE "scale((x, y)) -> Mask\nResizes a mask"

#define DOC_MASKDRAW "draw(othermask, offset) -> None\nDraws a mask onto another"

#define DOC_MASKERASE "erase(othermask, offset) -> None\nErases a mask from another"

#define DOC_MASKCOUNT "count() -> pixels\nReturns the number of set pixels"

#define DOC_MASKCENTROID "centroid() -> (x, y)\nReturns the centroid of the pixels in a Mask"

#define DOC_MASKANGLE "angle() -> theta\nReturns the orientation of the pixels"

#define DOC_MASKOUTLINE "outline(every = 1) -> [(x,y), (x,y) ...]\nlist of points outlining an object"

#define DOC_MASKCONVOLVE "convolve(othermask, outputmask = None, offset = (0,0)) -> Mask\nReturn the convolution of self with another mask."

#define DOC_MASKCONNECTEDCOMPONENT "connected_component((x,y) = None) -> Mask\nReturns a mask of a connected region of pixels."

#define DOC_MASKCONNECTEDCOMPONENTS "connected_components(min = 0) -> [Masks]\nReturns a list of masks of connected regions of pixels."

#define DOC_MASKGETBOUNDINGRECTS "get_bounding_rects() -> Rects\nReturns a list of bounding rects of regions of set pixels."



/* Docs in a comment... slightly easier to read. */

/*

pygame.mask
pygame module for image masks.

pygame.mask.from_surface
 from_surface(Surface, threshold = 127) -> Mask
Returns a Mask from the given surface.

pygame.mask.from_threshold
 from_threshold(Surface, color, threshold = (0,0,0,255), othersurface = None, palette_colors = 1) -> Mask
Creates a mask by thresholding Surfaces

pygame.mask.Mask
 Mask((width, height)) -> Mask
pygame object for representing 2d bitmasks

pygame.mask.Mask.get_size
 get_size() -> width,height
Returns the size of the mask.

pygame.mask.Mask.get_at
 get_at((x,y)) -> int
Returns nonzero if the bit at (x,y) is set.

pygame.mask.Mask.set_at
 set_at((x,y),value) -> None
Sets the position in the mask given by x and y.

pygame.mask.Mask.overlap
 overlap(othermask, offset) -> x,y
Returns the point of intersection if the masks overlap with the given offset - or None if it does not overlap.

pygame.mask.Mask.overlap_area
 overlap_area(othermask, offset) -> numpixels
Returns the number of overlapping 'pixels'.

pygame.mask.Mask.overlap_mask
 overlap_mask(othermask, offset) -> Mask
Returns a mask of the overlapping pixels

pygame.mask.Mask.fill
 fill() -> None
Sets all bits to 1

pygame.mask.Mask.clear
 clear() -> None
Sets all bits to 0

pygame.mask.Mask.invert
 invert() -> None
Flips the bits in a Mask

pygame.mask.Mask.scale
 scale((x, y)) -> Mask
Resizes a mask

pygame.mask.Mask.draw
 draw(othermask, offset) -> None
Draws a mask onto another

pygame.mask.Mask.erase
 erase(othermask, offset) -> None
Erases a mask from another

pygame.mask.Mask.count
 count() -> pixels
Returns the number of set pixels

pygame.mask.Mask.centroid
 centroid() -> (x, y)
Returns the centroid of the pixels in a Mask

pygame.mask.Mask.angle
 angle() -> theta
Returns the orientation of the pixels

pygame.mask.Mask.outline
 outline(every = 1) -> [(x,y), (x,y) ...]
list of points outlining an object

pygame.mask.Mask.convolve
 convolve(othermask, outputmask = None, offset = (0,0)) -> Mask
Return the convolution of self with another mask.

pygame.mask.Mask.connected_component
 connected_component((x,y) = None) -> Mask
Returns a mask of a connected region of pixels.

pygame.mask.Mask.connected_components
 connected_components(min = 0) -> [Masks]
Returns a list of masks of connected regions of pixels.

pygame.mask.Mask.get_bounding_rects
 get_bounding_rects() -> Rects
Returns a list of bounding rects of regions of set pixels.

*/