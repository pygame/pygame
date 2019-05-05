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

   | :sl:`Creates a Mask from the given surface`
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

   If the optional ``othersurface`` is not used, all the pixels **within** the
   ``threshold`` of the ``color`` parameter are **set** in the resulting mask.

   If the optional ``othersurface`` is used, every pixel in the first surface
   that is **within** the ``threshold`` of the corresponding pixel in
   ``othersurface`` is **set** in the resulting mask.

   :param Surface surface: the surface to create the mask from
   :param color: color used to check if the surface's pixels are within the
      given ``threshold`` range, this parameter is ignored if the optional
      ``othersurface`` parameter is supplied
   :type color: Color or int or tuple(int, int, int, [int]) or list[int, int, int, [int]]
   :param threshold: (optional) the threshold range used to check the difference
      between two colors (default is ``(0, 0, 0, 255)``)
   :type threshold: Color or int or tuple(int, int, int, [int]) or list[int, int, int, [int]]
   :param Surface othersurface: (optional) used to check whether the pixels of
      the first surface are within the given ``threshold`` range of the pixels
      from this surface (default is ``None``)
   :param int palette_colors: (optional) indicates whether to use the palette
      colors or not, a nonzero value causes the palette colors to be used and a
      0 causes them not to be used (default is 1)

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

      | :sl:`Returns the size of the mask`
      | :sg:`get_size() -> (width, height)`

      :returns: the size of the mask, (width, height)
      :rtype: tuple(int, int)

      .. ## Mask.get_size ##

   .. method:: get_at

      | :sl:`Gets the bit at the given position`
      | :sg:`get_at((x, y)) -> int`

      :param pos: the position of the bit to get
      :type pos: tuple(int, int) or list[int, int]

      :returns: 1 if the bit is set, 0 if the bit is not set
      :rtype: int

      :raises IndexError: if the position is outside of the mask's bounds

      .. ## Mask.get_at ##

   .. method:: set_at

      | :sl:`Sets the bit at the given position`
      | :sg:`set_at((x, y)) -> None`
      | :sg:`set_at((x, y), value=1) -> None`

      :param pos: the position of the bit to set
      :type pos: tuple(int, int) or list[int, int]
      :param int value: any nonzero int will set the bit to 1, 0 will set the
         bit to 0 (default is 1)

      :returns: ``None``
      :rtype: NoneType

      :raises IndexError: if the position is outside of the mask's bounds

      .. ## Mask.set_at ##

   .. method:: overlap

      | :sl:`Returns the point of intersection`
      | :sg:`overlap(othermask, offset) -> (x, y)`
      | :sg:`overlap(othermask, offset) -> None`

      Returns the first point of intersection encountered between this mask and
      ``othermask``. A point of intersection is 2 overlapping set bits.

      The current algorithm searches the overlapping area in 32 bit wide blocks.
      Starting at the top left corner (``(0, 0)``), it checks bits 0 to 31 of
      the first row (``(0, 0)`` to ``(31, 0)``) then continues to the next row. 
      Once this entire 32 bit column is checked, it continues to the next 32 bit
      column (32 to 63). This is repeated until it finds a point of intersection
      or the entire overlapping area is checked.

      :param Mask othermask: the other mask to overlap with this mask
      :param offset: the offset of ``othermask`` from this mask, for more
         details refer to the :ref:`Mask offset notes <mask-offset-label>`
      :type offset: tuple(int, int) or list[int, int]

      :returns: point of intersection or ``None`` if no intersection
      :rtype: tuple(int, int) or NoneType

      .. ## Mask.overlap ##

   .. method:: overlap_area

      | :sl:`Returns the number of overlapping set bits`
      | :sg:`overlap_area(othermask, offset) -> numbits`

      Returns the number of overlapping set bits between between this mask and
      ``othermask``.

      This can be useful for collision detection. An approximate collision
      normal can be found by calculating the gradient of the overlapping area
      through the finite difference.

      ::

         dx = mask.overlap_area(othermask, (x + 1, y)) - mask.overlap_area(othermask, (x - 1, y))
         dy = mask.overlap_area(othermask, (x, y + 1)) - mask.overlap_area(othermask, (x, y - 1))

      :param Mask othermask: the other mask to overlap with this mask
      :param offset: the offset of ``othermask`` from this mask, for more
         details refer to the :ref:`Mask offset notes <mask-offset-label>`
      :type offset: tuple(int, int) or list[int, int]

      :returns: the number of overlapping set bits
      :rtype: int

      .. ## Mask.overlap_area ##

   .. method:: overlap_mask

      | :sl:`Returns a mask of the overlapping set bits`
      | :sg:`overlap_mask(othermask, offset) -> Mask`

      Returns a :class:`Mask`, the same size as this mask, containing the
      overlapping set bits between this mask and ``othermask``.

      :param Mask othermask: the other mask to overlap with this mask
      :param offset: the offset of ``othermask`` from this mask, for more
         details refer to the :ref:`Mask offset notes <mask-offset-label>`
      :type offset: tuple(int, int) or list[int, int]

      :returns: a newly created :class:`Mask` with the overlapping bits set
      :rtype: Mask

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
      :param offset: the offset of ``othermask`` from this mask, for more
         details refer to the :ref:`Mask offset notes <mask-offset-label>`
      :type offset: tuple(int, int) or list[int, int]

      :returns: ``None``
      :rtype: NoneType

      .. ## Mask.draw ##

   .. method:: erase

      | :sl:`Erases a mask from another`
      | :sg:`erase(othermask, offset) -> None`

      Erases (clears) all bits set in ``othermask`` from this mask.

      :param Mask othermask: the mask to erase from this mask
      :param offset: the offset of ``othermask`` from this mask, for more
         details refer to the :ref:`Mask offset notes <mask-offset-label>`
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
         the outline (default is 1)

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
      :param outputmask: (optional) mask for output (default is ``None``)
      :type outputmask: Mask or NoneType
      :param offset: the offset of ``othermask`` from this mask, (default is
         ``(0, 0)``)
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

      | :sl:`Returns a mask containing a connected component`
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

      | :sl:`Returns a list of masks of connected components`
      | :sg:`connected_components() -> [Mask, ...]`
      | :sg:`connected_components(min=0) -> [Mask, ...]`

      Provides a list containing a :class:`Mask` object for each connected
      component.

      :param int min: (optional) indicates the minimum number of bits (to filter
         out noise) per connected component (default is 0, which equates to
         no minimum and is equivalent to setting it to 1, as a connected
         component must have at least 1 bit set)

      :returns: a list containing a :class:`Mask` object for each connected
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
