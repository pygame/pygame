.. include:: common.txt

:mod:`pygame.mask`
==================

.. module:: pygame.mask
   :synopsis: pygame module for image masks.

| :sl:`pygame module for image masks.`

Useful for fast pixel perfect collision detection. A Mask uses 1bit per pixel
to store which parts collide.

New in pygame 1.8.

.. function:: from_surface

   | :sl:`Returns a Mask from the given surface.`
   | :sg:`from_surface(Surface, threshold = 127) -> Mask`

   Makes the transparent parts of the Surface not set, and the opaque parts
   set.

   The alpha of each pixel is checked to see if it is greater than the given
   threshold.

   If the Surface is color keyed, then threshold is not used.

   .. ## pygame.mask.from_surface ##

.. function:: from_threshold

   | :sl:`Creates a mask by thresholding Surfaces`
   | :sg:`from_threshold(Surface, color, threshold = (0,0,0,255), othersurface = None, palette_colors = 1) -> Mask`

   This is a more featureful method of getting a Mask from a Surface. If
   supplied with only one Surface, all pixels within the threshold of the
   supplied color are set in the Mask. If given the optional othersurface, all
   pixels in Surface that are within the threshold of the corresponding pixel
   in othersurface are set in the Mask.

   .. ## pygame.mask.from_threshold ##

.. class:: Mask

   | :sl:`pygame object for representing 2d bitmasks`
   | :sg:`Mask((width, height)) -> Mask`

   .. method:: get_size

      | :sl:`Returns the size of the mask.`
      | :sg:`get_size() -> width,height`

      .. ## Mask.get_size ##

   .. method:: get_at

      | :sl:`Returns nonzero if the bit at (x,y) is set.`
      | :sg:`get_at((x,y)) -> int`

      Coordinates start at (0,0) is top left - just like Surfaces.

      .. ## Mask.get_at ##

   .. method:: set_at

      | :sl:`Sets the position in the mask given by x and y.`
      | :sg:`set_at((x,y),value) -> None`

      .. ## Mask.set_at ##

   .. method:: overlap

      | :sl:`Returns the point of intersection if the masks overlap with the given offset - or None if it does not overlap.`
      | :sg:`overlap(othermask, offset) -> x,y`

      The overlap tests uses the following offsets (which may be negative):

      ::

         +----+----------..
         |A   | yoffset
         |  +-+----------..
         +--|B
         |xoffset
         |  |
         :  :

      .. ## Mask.overlap ##

   .. method:: overlap_area

      | :sl:`Returns the number of overlapping 'pixels'.`
      | :sg:`overlap_area(othermask, offset) -> numpixels`

      You can see how many pixels overlap with the other mask given. This can
      be used to see in which direction things collide, or to see how much the
      two masks collide. An approximate collision normal can be found by
      calculating the gradient of the overlap area through the finite
      difference.

      ::

       dx = Mask.overlap_area(othermask,(x+1,y)) - Mask.overlap_area(othermask,(x-1,y))
       dy = Mask.overlap_area(othermask,(x,y+1)) - Mask.overlap_area(othermask,(x,y-1))

      .. ## Mask.overlap_area ##

   .. method:: overlap_mask

      | :sl:`Returns a mask of the overlapping pixels`
      | :sg:`overlap_mask(othermask, offset) -> Mask`

      Returns a Mask the size of the original Mask containing only the
      overlapping pixels between Mask and othermask.

      .. ## Mask.overlap_mask ##

   .. method:: fill

      | :sl:`Sets all bits to 1`
      | :sg:`fill() -> None`

      Sets all bits in a Mask to 1.

      .. ## Mask.fill ##

   .. method:: clear

      | :sl:`Sets all bits to 0`
      | :sg:`clear() -> None`

      Sets all bits in a Mask to 0.

      .. ## Mask.clear ##

   .. method:: invert

      | :sl:`Flips the bits in a Mask`
      | :sg:`invert() -> None`

      Flips all of the bits in a Mask, so that the set pixels turn to unset
      pixels and the unset pixels turn to set pixels.

      .. ## Mask.invert ##

   .. method:: scale

      | :sl:`Resizes a mask`
      | :sg:`scale((x, y)) -> Mask`

      Returns a new Mask of the Mask scaled to the requested size.

      .. ## Mask.scale ##

   .. method:: draw

      | :sl:`Draws a mask onto another`
      | :sg:`draw(othermask, offset) -> None`

      Performs a bitwise ``OR``, drawing othermask onto Mask.

      .. ## Mask.draw ##

   .. method:: erase

      | :sl:`Erases a mask from another`
      | :sg:`erase(othermask, offset) -> None`

      Erases all pixels set in othermask from Mask.

      .. ## Mask.erase ##

   .. method:: count

      | :sl:`Returns the number of set pixels`
      | :sg:`count() -> pixels`

      Returns the number of set pixels in the Mask.

      .. ## Mask.count ##

   .. method:: centroid

      | :sl:`Returns the centroid of the pixels in a Mask`
      | :sg:`centroid() -> (x, y)`

      Finds the centroid, the center of pixel mass, of a Mask. Returns a
      coordinate tuple for the centroid of the Mask. In the event the Mask is
      empty, it will return (0,0).

      .. ## Mask.centroid ##

   .. method:: angle

      | :sl:`Returns the orientation of the pixels`
      | :sg:`angle() -> theta`

      Finds the approximate orientation of the pixels in the image from -90 to
      90 degrees. This works best if performed on one connected component of
      pixels. It will return 0.0 on an empty Mask.

      .. ## Mask.angle ##

   .. method:: outline

      | :sl:`list of points outlining an object`
      | :sg:`outline(every = 1) -> [(x,y), (x,y) ...]`

      Returns a list of points of the outline of the first object it comes
      across in a Mask. For this to be useful, there should probably only be
      one connected component of pixels in the Mask. The every option allows
      you to skip pixels in the outline. For example, setting it to 10 would
      return a list of every 10th pixel in the outline.

      .. ## Mask.outline ##

   .. method:: convolve

      | :sl:`Return the convolution of self with another mask.`
      | :sg:`convolve(othermask, outputmask = None, offset = (0,0)) -> Mask`

      Returns a mask with the (i-offset[0],j-offset[1]) bit set if shifting
      othermask so that its lower right corner pixel is at (i,j) would cause
      it to overlap with self.

      If an outputmask is specified, the output is drawn onto outputmask and
      outputmask is returned. Otherwise a mask of size ``self.get_size()`` +
      ``othermask.get_size()`` - (1,1) is created.

      .. ## Mask.convolve ##

   .. method:: connected_component

      | :sl:`Returns a mask of a connected region of pixels.`
      | :sg:`connected_component((x,y) = None) -> Mask`

      This uses the ``SAUF`` algorithm to find a connected component in the
      Mask. It checks 8 point connectivity. By default, it will return the
      largest connected component in the image. Optionally, a coordinate pair
      of a pixel can be specified, and the connected component containing it
      will be returned. In the event the pixel at that location is not set, the
      returned Mask will be empty. The Mask returned is the same size as the
      original Mask.

      .. ## Mask.connected_component ##

   .. method:: connected_components

      | :sl:`Returns a list of masks of connected regions of pixels.`
      | :sg:`connected_components(min = 0) -> [Masks]`

      Returns a list of masks of connected regions of pixels. An optional
      minimum number of pixels per connected region can be specified to filter
      out noise.

      .. ## Mask.connected_components ##

   .. method:: get_bounding_rects

      | :sl:`Returns a list of bounding rects of regions of set pixels.`
      | :sg:`get_bounding_rects() -> Rects`

      This gets a bounding rect of connected regions of set pixels. A bounding
      rect is one for which each of the connected pixels is inside the rect.

      .. ## Mask.get_bounding_rects ##

   .. ## pygame.mask.Mask ##

.. ## pygame.mask ##
