/* Auto generated file: with makeref.py .  Docs go in docs/reST/ref/ . */
#define DOC_PYGAMEMASK "pygame module for image masks."
#define DOC_PYGAMEMASKFROMSURFACE "from_surface(surface) -> Mask\nfrom_surface(surface, threshold=127) -> Mask\nCreates a Mask from the given surface"
#define DOC_PYGAMEMASKFROMTHRESHOLD "from_threshold(surface, color) -> Mask\nfrom_threshold(surface, color, threshold=(0, 0, 0, 255), othersurface=None, palette_colors=1) -> Mask\nCreates a mask by thresholding Surfaces"
#define DOC_PYGAMEMASKMASK "Mask(size=(width, height)) -> Mask\nMask(size=(width, height), fill=False) -> Mask\npygame object for representing 2D bitmasks"
#define DOC_MASKCOPY "copy() -> Mask\nReturns a new copy of the mask"
#define DOC_MASKGETSIZE "get_size() -> (width, height)\nReturns the size of the mask"
#define DOC_MASKGETRECT "get_rect(**kwargs) -> Rect\nReturns a Rect based on the size of the mask"
#define DOC_MASKGETAT "get_at(pos) -> int\nGets the bit at the given position"
#define DOC_MASKSETAT "set_at(pos) -> None\nset_at(pos, value=1) -> None\nSets the bit at the given position"
#define DOC_MASKOVERLAP "overlap(other, offset) -> (x, y)\noverlap(other, offset) -> None\nReturns the point of intersection"
#define DOC_MASKOVERLAPAREA "overlap_area(other, offset) -> numbits\nReturns the number of overlapping set bits"
#define DOC_MASKOVERLAPMASK "overlap_mask(other, offset) -> Mask\nReturns a mask of the overlapping set bits"
#define DOC_MASKFILL "fill() -> None\nSets all bits to 1"
#define DOC_MASKCLEAR "clear() -> None\nSets all bits to 0"
#define DOC_MASKINVERT "invert() -> None\nFlips all the bits"
#define DOC_MASKSCALE "scale((width, height)) -> Mask\nResizes a mask"
#define DOC_MASKDRAW "draw(other, offset) -> None\nDraws a mask onto another"
#define DOC_MASKERASE "erase(other, offset) -> None\nErases a mask from another"
#define DOC_MASKCOUNT "count() -> bits\nReturns the number of set bits"
#define DOC_MASKCENTROID "centroid() -> (x, y)\nReturns the centroid of the set bits"
#define DOC_MASKANGLE "angle() -> theta\nReturns the orientation of the set bits"
#define DOC_MASKOUTLINE "outline() -> [(x, y), ...]\noutline(every=1) -> [(x, y), ...]\nReturns a list of points outlining an object"
#define DOC_MASKCONVOLVE "convolve(other) -> Mask\nconvolve(other, output=None, offset=(0, 0)) -> Mask\nReturns the convolution of this mask with another mask"
#define DOC_MASKCONNECTEDCOMPONENT "connected_component() -> Mask\nconnected_component(pos) -> Mask\nReturns a mask containing a connected component"
#define DOC_MASKCONNECTEDCOMPONENTS "connected_components() -> [Mask, ...]\nconnected_components(minimum=0) -> [Mask, ...]\nReturns a list of masks of connected components"
#define DOC_MASKGETBOUNDINGRECTS "get_bounding_rects() -> [Rect, ...]\nReturns a list of bounding rects of connected components"
#define DOC_MASKTOSURFACE "to_surface() -> Surface\nto_surface(surface=None, setsurface=None, unsetsurface=None, setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 255), dest=(0, 0)) -> Surface\nReturns a surface with the mask drawn on it"


/* Docs in a comment... slightly easier to read. */

/*

pygame.mask
pygame module for image masks.

pygame.mask.from_surface
 from_surface(surface) -> Mask
 from_surface(surface, threshold=127) -> Mask
Creates a Mask from the given surface

pygame.mask.from_threshold
 from_threshold(surface, color) -> Mask
 from_threshold(surface, color, threshold=(0, 0, 0, 255), othersurface=None, palette_colors=1) -> Mask
Creates a mask by thresholding Surfaces

pygame.mask.Mask
 Mask(size=(width, height)) -> Mask
 Mask(size=(width, height), fill=False) -> Mask
pygame object for representing 2D bitmasks

pygame.mask.Mask.copy
 copy() -> Mask
Returns a new copy of the mask

pygame.mask.Mask.get_size
 get_size() -> (width, height)
Returns the size of the mask

pygame.mask.Mask.get_rect
 get_rect(**kwargs) -> Rect
Returns a Rect based on the size of the mask

pygame.mask.Mask.get_at
 get_at(pos) -> int
Gets the bit at the given position

pygame.mask.Mask.set_at
 set_at(pos) -> None
 set_at(pos, value=1) -> None
Sets the bit at the given position

pygame.mask.Mask.overlap
 overlap(other, offset) -> (x, y)
 overlap(other, offset) -> None
Returns the point of intersection

pygame.mask.Mask.overlap_area
 overlap_area(other, offset) -> numbits
Returns the number of overlapping set bits

pygame.mask.Mask.overlap_mask
 overlap_mask(other, offset) -> Mask
Returns a mask of the overlapping set bits

pygame.mask.Mask.fill
 fill() -> None
Sets all bits to 1

pygame.mask.Mask.clear
 clear() -> None
Sets all bits to 0

pygame.mask.Mask.invert
 invert() -> None
Flips all the bits

pygame.mask.Mask.scale
 scale((width, height)) -> Mask
Resizes a mask

pygame.mask.Mask.draw
 draw(other, offset) -> None
Draws a mask onto another

pygame.mask.Mask.erase
 erase(other, offset) -> None
Erases a mask from another

pygame.mask.Mask.count
 count() -> bits
Returns the number of set bits

pygame.mask.Mask.centroid
 centroid() -> (x, y)
Returns the centroid of the set bits

pygame.mask.Mask.angle
 angle() -> theta
Returns the orientation of the set bits

pygame.mask.Mask.outline
 outline() -> [(x, y), ...]
 outline(every=1) -> [(x, y), ...]
Returns a list of points outlining an object

pygame.mask.Mask.convolve
 convolve(other) -> Mask
 convolve(other, output=None, offset=(0, 0)) -> Mask
Returns the convolution of this mask with another mask

pygame.mask.Mask.connected_component
 connected_component() -> Mask
 connected_component(pos) -> Mask
Returns a mask containing a connected component

pygame.mask.Mask.connected_components
 connected_components() -> [Mask, ...]
 connected_components(minimum=0) -> [Mask, ...]
Returns a list of masks of connected components

pygame.mask.Mask.get_bounding_rects
 get_bounding_rects() -> [Rect, ...]
Returns a list of bounding rects of connected components

pygame.mask.Mask.to_surface
 to_surface() -> Surface
 to_surface(surface=None, setsurface=None, unsetsurface=None, setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 255), dest=(0, 0)) -> Surface
Returns a surface with the mask drawn on it

*/