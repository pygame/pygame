.. include:: common.txt

:mod:`pygame.mask`
==================

.. module:: pygame.mask
   :synopsis: pygame module for image masks.

| :sl:`pygame module for image masks.`

Useful for fast pixel perfect collision detection. A mask uses 1 bit per-pixel
to store which parts collide.

.. versionadded:: 1.8

.. versionchanged:: 1.9.5
   Added support for masks with a width and/or a height of 0.


.. function:: from_surface

   | :sl:`Creates a Mask from the given surface.`
   | :sg:`from_surface(Surface) -> Mask`
   | :sg:`from_surface(Surface, threshold=127) -> Mask`

   Creates a :class:`Mask` object from the given surface by setting all the
   opaque pixels and not setting the transparent pixels.

   If the surface uses a color-key, then it is used to decide which bits in
   the resulting mask are set. All the pixels that are **not** equal to the
   color-key are **set** and the pixels equal to the color-key are not set.

   If a color-key is not used, then the alpha value of each pixel is used to
   decide which bits in the resulting mask are set. All the pixels that have an
   alpha value **greater than** the ``threshold`` parameter are **set** and the
   pixels with an alpha value less than or equal to the ``threshold`` are
   not set.

   :param Surface surface: the surface to create the mask from
   :param int threshold: (optional) the alpha threshold (default is 127) to
      compare with each surface pixel's alpha value, if the ``surface`` is
      color-keyed this parameter is ignored

   :returns: a newly created :class:`Mask` object from the given surface
   :rtype: Mask

   .. ## pygame.mask.from_surface ##

.. function:: from_threshold

   | :sl:`Creates a mask by thresholding Surfaces`
   | :sg:`from_threshold(Surface, color) -> Mask`
   | :sg:`from_threshold(Surface, color, threshold=(0, 0, 0, 255), othersurface=None, palette_colors=1) -> Mask`

   This is a more featureful method of getting a :class:`Mask` from a surface.
   If supplied with only one surface, all pixels within the threshold of the
   supplied color are set in the mask. If given the optional ``othersurface``,
   all pixels in the first surface that are within the threshold of the
   corresponding pixel in ``othersurface`` are set in the mask.

   :returns: a newly created :class:`Mask` object from the given surface
   :rtype: Mask

   .. ## pygame.mask.from_threshold ##

.. class:: Mask

   | :sl:`pygame object for representing 2D bitmasks`
   | :sg:`Mask(size=(width, height)) -> Mask`
   | :sg:`Mask(size=(width, height), fill=False) -> Mask`

   A ``Mask`` object is used to represent a 2D bitmask. Each bit in
   the mask represents a pixel. 1 is used to indicate a set bit and 0 is used
   to indicate an unset bit. Set bits in a mask can be used to detect collisions
   with other masks and their set bits.

   A filled mask has all of its bits set to 1, conversely an
   unfilled/cleared/empty mask has all of its bits set to 0. Masks can be
   created unfilled (default) or filled by using the ``fill`` parameter. Masks
   can also be cleared or filled using the :func:`pygame.mask.Mask.clear()` and
   :func:`pygame.mask.Mask.fill()` methods respectively.

   A mask's coordinates start in the top left corner at ``(0, 0)`` just like
   :mod:`pygame.Surface`. Individual bits can be accessed using the
   :func:`pygame.mask.Mask.get_at()` and :func:`pygame.mask.Mask.set_at()`
   methods.

   .. _mask-offset-label:

   The methods :meth:`overlap`, :meth:`overlap_area`, :meth:`overlap_mask`,
   :meth:`draw`, :meth:`erase`, and :meth:`convolve` use an offset parameter
   to indicate the offset of another mask's top left corner from the calling
   mask's top left corner. The calling mask's top left corner is considered to
   be the origin ``(0, 0)``. Offsets are a tuple or list of 2 integer values
   ``(x_offset, y_offset)``. Positive and negative offset values are supported.

   ::

                 0 to x (x_offset)
                 :    :
         0 ..... +----:---------+
         to      |    :         |
         y .......... +-----------+
      (y_offset) |    | othermask |
                 |    +-----------+
                 | calling_mask |
                 +--------------+

   :param size: the dimensions of the mask (width and height)
   :type size: tuple(int, int) or list[int, int]
   :param bool fill: (optional) create an unfilled mask (default: ``False``) or
      filled mask (``True``)

   :returns: a newly created :class:`Mask` object
   :rtype: Mask

   .. versionadded:: 1.9.5 Added support for keyword arguments.
   .. versionadded:: 1.9.5 Added the optional keyword parameter ``fill``.
   .. versionadded:: 2.0.0 Subclassing support added. The :class:`Mask` class
      can be used as a base class.

   .. method:: get_size

      | :sl:`Returns the size of the mask.`
      | :sg:`get_size() -> (width, height)`

      :returns: the size of the mask, (width, height)
      :rtype: tuple(int, int)

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
      | :sg:`convolve(othermask) -> Mask`
      | :sg:`convolve(othermask, outputmask=None, offset=(0,0)) -> Mask`

      Convolve self with the given ``othermask``.

      :param Mask othermask: mask to convolve with self
      :param outputmask: mask for output, default is None
      :type outputmask: Mask or None
      :param offset: offset used in convolution of masks, default is (0, 0)
      :type offset: tuple(int, int) or list[int, int]

      :returns: A mask with the ``(i - offset[0], j - offset[1])`` bit set, if
         shifting ``othermask`` (such that its bottom right corner pixel is at
         ``(i, j)``) causes it to overlap with self.

         If an ``outputmask`` is specified, the output is drawn onto it and
         it is returned. Otherwise a mask of size ``(MAX(0, width + othermask's
         width - 1), MAX(0, height + othermask's height - 1))`` is created and
         returned.
      :rtype: Mask

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
