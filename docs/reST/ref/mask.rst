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

      Sets all bits in the mask to 1.

      :returns: ``None``
      :rtype: NoneType

      .. ## Mask.fill ##

   .. method:: clear

      | :sl:`Sets all bits to 0`
      | :sg:`clear() -> None`

      Sets all bits in the mask to 0.

      :returns: ``None``
      :rtype: NoneType

      .. ## Mask.clear ##

   .. method:: invert

      | :sl:`Flips all the bits`
      | :sg:`invert() -> None`

      Flips all of the bits in the mask. All the set bits are cleared to 0 and
      all the unset bits are set to 1.

      :returns: ``None``
      :rtype: NoneType

      .. ## Mask.invert ##

   .. method:: scale

      | :sl:`Resizes a mask`
      | :sg:`scale((width, height)) -> Mask`

      Creates a new :class:`Mask` of the requested size with its bits scaled
      from this mask.

      :param size: the width and height (size) of the mask to create
      :type size: tuple(int, int) or list[int, int]

      :returns: a new :class:`Mask` object with its bits scaled from this mask
      :rtype: Mask

      :raises ValueError: if ``width < 0`` or ``height < 0``

      .. ## Mask.scale ##

   .. method:: draw

      | :sl:`Draws a mask onto another`
      | :sg:`draw(othermask, offset) -> None`

      Performs a bitwise OR, drawing ``othermask`` onto this mask.

      :param Mask othermask: the mask to draw onto this mask
      :param offset: the offset of ``othermask`` from this mask
      :type offset: tuple(int, int) or list[int, int]

      :returns: ``None``
      :rtype: NoneType

      .. ## Mask.draw ##

   .. method:: erase

      | :sl:`Erases a mask from another`
      | :sg:`erase(othermask, offset) -> None`

      Erases (clears) all bits set in ``othermask`` from this mask.

      :param Mask othermask: the mask to erase from this mask
      :param offset: the offset of ``othermask`` from this mask
      :type offset: tuple(int, int) or list[int, int]

      :returns: ``None``
      :rtype: NoneType

      .. ## Mask.erase ##

   .. method:: count

      | :sl:`Returns the number of set bits`
      | :sg:`count() -> bits`

      :returns: the number of set bits in the mask
      :rtype: int

      .. ## Mask.count ##

   .. method:: centroid

      | :sl:`Returns the centroid of the set bits`
      | :sg:`centroid() -> (x, y)`

      Finds the centroid (the center mass of the set bits) for this mask.

      :returns: a coordinate tuple indicating the centroid of the mask, it will
         return ``(0, 0)`` if the mask has no bits set
      :rtype: tuple(int, int)

      .. ## Mask.centroid ##

   .. method:: angle

      | :sl:`Returns the orientation of the set bits`
      | :sg:`angle() -> theta`

      Finds the approximate orientation (from -90 to 90 degrees) of the set bits
      in the mask. This works best if performed on a mask with only one
      connected component.

      :returns: the orientation of the set bits in the mask, it will return
         ``0.0`` if the mask has no bits set
      :rtype: float

      .. note::
         See :meth:`connected_component` for details on how a connected
         component is calculated.

      .. ## Mask.angle ##

   .. method:: outline

      | :sl:`Returns a list of points outlining an object`
      | :sg:`outline() -> [(x, y), ...]`
      | :sg:`outline(every=1) -> [(x, y), ...]`

      Returns a list of points of the outline of the first connected component
      encountered in the mask. To find a connected component, the mask is
      searched per row (left to right) starting in the top left corner.

      The ``every`` optional parameter skips set bits in the outline. For
      example, setting it to 10 would return a list of every 10th set bit in the
      outline.

      :param int every: (optional) indicates the number of bits to skip over in
         the outline, the default is 1

      :returns: a list of points outlining the first connected component
         encountered, an empty list is returned if the mask has no bits set
      :rtype: list[tuple(int, int)]

      .. note::
         See :meth:`connected_component` for details on how a connected
         component is calculated.

      .. ## Mask.outline ##

   .. method:: convolve

      | :sl:`Returns the convolution of this mask with another mask`
      | :sg:`convolve(othermask) -> Mask`
      | :sg:`convolve(othermask, outputmask=None, offset=(0, 0)) -> Mask`

      Convolve this mask with the given ``othermask``.

      :param Mask othermask: mask to convolve this mask with
      :param outputmask: (optional) mask for output, the default is ``None``
      :type outputmask: Mask or None
      :param offset: the offset of ``othermask`` from this mask, the default is
         ``(0, 0)``
      :type offset: tuple(int, int) or list[int, int]

      :returns: a :class:`Mask` with the ``(i - offset[0], j - offset[1])`` bit
         set, if shifting ``othermask`` (such that its bottom right corner is at
         ``(i, j)``) causes it to overlap with this mask

         If an ``outputmask`` is specified, the output is drawn onto it and
         it is returned. Otherwise a mask of size ``(MAX(0, width + othermask's
         width - 1), MAX(0, height + othermask's height - 1))`` is created and
         returned.
      :rtype: Mask

      .. ## Mask.convolve ##

   .. method:: connected_component

      | :sl:`Returns a Mask containing a connected component`
      | :sg:`connected_component() -> Mask`
      | :sg:`connected_component((x, y)) -> Mask`

      A connected component is a group (1 or more) of connected set bits
      (orthogonally and diagonally). The SAUF algorithm, which checks 8 point
      connectivity, is used to find a connected component in the mask.

      By default this method will return a :class:`Mask` containing the largest
      connected component in the mask. Optionally, a bit coordinate can be
      specified and the connected component containing it will be returned. If
      the bit at the given location is not set, the returned :class:`Mask` will
      be empty (no bits set).

      :param pos: (optional) selects the connected component that contains the
         bit at this position
      :type pos: tuple(int, int) or list[int, int]

      :returns: a :class:`Mask` object (same size as this mask) with the largest
         connected component from this mask, if this mask has no bits set then
         an empty mask will be returned

         If the ``pos`` parameter is provided then the mask returned will have
         the connected component that contains this position. An empty mask will
         be returned if the ``pos`` parameter selects an unset bit.
      :rtype: Mask

      :raises IndexError: if the optional ``pos`` parameter is outside of the
         mask's bounds

      .. ## Mask.connected_component ##

   .. method:: connected_components

      | :sl:`Returns a list of Masks of connected components`
      | :sg:`connected_components() -> [Mask, ...]`
      | :sg:`connected_components(min=0) -> [Mask, ...]`

      Provides a list containing a ``Mask`` object for each connected component.

      :param int min: (optional) indicates the minimum number of bits (to filter
         out noise) per connected component, the default is 0 (this equates to
         no minimum, which is equivalent to setting it to 1 as a connected
         component must have at least 1 bit set)

      :returns: a list containing a ``Mask`` object for each connected
         component, an empty list is returned if the mask has no bits set
      :rtype: list[Mask]

      .. note::
         See :meth:`connected_component` for details on how a connected
         component is calculated.

      .. ## Mask.connected_components ##

   .. method:: get_bounding_rects

      | :sl:`Returns a list of bounding rects of connected components`
      | :sg:`get_bounding_rects() -> [Rect, ...]`

      Provides a list containing a bounding rect for each connected component.

      :returns: a list containing a bounding rect for each connected component,
         an empty list is returned if the mask has no bits set
      :rtype: list[Rect]

      .. note::
         See :meth:`connected_component` for details on how a connected
         component is calculated.

      .. ## Mask.get_bounding_rects ##

   .. ## pygame.mask.Mask ##

.. ## pygame.mask ##
