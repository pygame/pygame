/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMEMASK "pygame module for image masks."

#define DOC_PYGAMEMASKFROMSURFACE "pygame.mask.from_surface(Surface, threshold = 127) -> Mask\nReturns a Mask from the given surface."

#define DOC_PYGAMEMASKFROMTHRESHOLD "pygame.mask.from_threshold(Surface, color, threshold = (0,0,0,255), othersurface = None, palette_colors = 1) -> Mask\nCreates a mask by thresholding Surfaces"

#define DOC_PYGAMEMASKMASK "pygame.mask.Mask((width, height)): return Mask\npygame object for representing 2d bitmasks"

#define DOC_MASKGETSIZE "Mask.get_size() -> width,height\nReturns the size of the mask."

#define DOC_MASKGETAT "Mask.get_at((x,y)) -> int\nReturns nonzero if the bit at (x,y) is set."

#define DOC_MASKSETAT "Mask.set_at((x,y),value)\nSets the position in the mask given by x and y."

#define DOC_MASKOVERLAP "Mask.overlap(othermask, offset) -> x,y\nReturns the point of intersection if the masks overlap with the given offset - or None if it does not overlap."

#define DOC_MASKOVERLAPAREA "Mask.overlap_area(othermask, offset) -> numpixels\nReturns the number of overlapping 'pixels'."

#define DOC_MASKOVERLAPMASK "Mask.overlap_mask(othermask, offset) -> Mask\nReturns a mask of the overlapping pixels"

#define DOC_MASKFILL "Mask.fill()\nSets all bits to 1"

#define DOC_MASKCLEAR "Mask.clear()\nSets all bits to 0"

#define DOC_MASKINVERT "Mask.invert()\nFlips the bits in a Mask"

#define DOC_MASKSCALE "Mask.scale((x, y)) -> Mask\nResizes a mask"

#define DOC_MASKDRAW "Mask.draw(othermask, offset)\nDraws a mask onto another"

#define DOC_MASKERASE "Mask.erase(othermask, offset)\nErases a mask from another"

#define DOC_MASKCOUNT "Mask.count() -> pixels\nReturns the number of set pixels"

#define DOC_MASKCENTROID "Mask.centroid() -> (x, y)\nReturns the centroid of the pixels in a Mask"

#define DOC_MASKANGLE "Mask.angle() -> theta\nReturns the orientation of the pixels"

#define DOC_MASKOUTLINE "Mask.outline(every = 1) -> [(x,y), (x,y) ...]\nlist of points outlining an object"

#define DOC_MASKCONVOLVE "Mask.convolve(othermask, outputmask = None, offset = (0,0)) -> Mask\nReturn the convolution of self with another mask."

#define DOC_MASKCONNECTEDCOMPONENT "Mask.connected_component((x,y) = None) -> Mask\nReturns a mask of a connected region of pixels."

#define DOC_MASKCONNECTEDCOMPONENTS "Mask.connected_components(min = 0) -> [Masks]\nReturns a list of masks of connected regions of pixels."

#define DOC_MASKGETBOUNDINGRECTS "Mask.get_bounding_rects() -> Rects\nReturns a list of bounding rects of regions of set pixels."



/* Docs in a comments... slightly easier to read. */


/*

pygame.mask
 pygame module for image masks.



pygame.mask.from_surface
 pygame.mask.from_surface(Surface, threshold = 127) -> Mask
Returns a Mask from the given surface.



pygame.mask.from_threshold
 pygame.mask.from_threshold(Surface, color, threshold = (0,0,0,255), othersurface = None, palette_colors = 1) -> Mask
Creates a mask by thresholding Surfaces



pygame.mask.Mask
 pygame.mask.Mask((width, height)): return Mask
pygame object for representing 2d bitmasks



Mask.get_size
 Mask.get_size() -> width,height
Returns the size of the mask.



Mask.get_at
 Mask.get_at((x,y)) -> int
Returns nonzero if the bit at (x,y) is set.



Mask.set_at
 Mask.set_at((x,y),value)
Sets the position in the mask given by x and y.



Mask.overlap
 Mask.overlap(othermask, offset) -> x,y
Returns the point of intersection if the masks overlap with the given offset - or None if it does not overlap.



Mask.overlap_area
 Mask.overlap_area(othermask, offset) -> numpixels
Returns the number of overlapping 'pixels'.



Mask.overlap_mask
 Mask.overlap_mask(othermask, offset) -> Mask
Returns a mask of the overlapping pixels



Mask.fill
 Mask.fill()
Sets all bits to 1



Mask.clear
 Mask.clear()
Sets all bits to 0



Mask.invert
 Mask.invert()
Flips the bits in a Mask



Mask.scale
 Mask.scale((x, y)) -> Mask
Resizes a mask



Mask.draw
 Mask.draw(othermask, offset)
Draws a mask onto another



Mask.erase
 Mask.erase(othermask, offset)
Erases a mask from another



Mask.count
 Mask.count() -> pixels
Returns the number of set pixels



Mask.centroid
 Mask.centroid() -> (x, y)
Returns the centroid of the pixels in a Mask



Mask.angle
 Mask.angle() -> theta
Returns the orientation of the pixels



Mask.outline
 Mask.outline(every = 1) -> [(x,y), (x,y) ...]
list of points outlining an object



Mask.convolve
 Mask.convolve(othermask, outputmask = None, offset = (0,0)) -> Mask
Return the convolution of self with another mask.



Mask.connected_component
 Mask.connected_component((x,y) = None) -> Mask
Returns a mask of a connected region of pixels.



Mask.connected_components
 Mask.connected_components(min = 0) -> [Masks]
Returns a list of masks of connected regions of pixels.



Mask.get_bounding_rects
 Mask.get_bounding_rects() -> Rects
Returns a list of bounding rects of regions of set pixels.



*/

