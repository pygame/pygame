.. include:: common.txt

:class:`pygame.PixelArray`
==========================

.. currentmodule:: pygame

.. class:: PixelArray

   | :sl:`pygame object for direct pixel access of surfaces`
   | :sg:`PixelArray(Surface) -> PixelArray`

   The PixelArray wraps a Surface and provides direct access to the
   surface's pixels. A pixel array can be one or two dimensional.
   A two dimensional array, likes its surface, is indexed [column, row].
   Pixel arrays support slicing, both for returning a subarray or
   for assignment. A pixel array sliced on a single column or row
   returns a one dimensional pixel array. Arithmetic and other operations
   are not supported. A pixel array can be safely assigned to itself.
   Finally, pixel arrays export an array struct interface, allowing
   them to interact with :mod:`pygame.pixelcopy` methods and NumPy
   arrays.

   While it is possible to assign both, integer color values and ``RGB(A)``
   color tuples, the PixelArray will only use integers for the color
   representation. Thus, checking for certain colors has to be done using the
   ``Surface.map_rgb()`` method of the surface, the PixelArray was created for.

   ::

     pxarray = pygame.PixelArray (surface)
     # Check, if the first pixel at the topleft corner is blue
     if pxarray[0, 0] == surface.map_rgb ((0, 0, 255)):
         ...

   Pixels can be manipulated using integer values or color tuples.

   ::

     pxarray[x, y] = 0xFF00FF
     pxarray[x, y] = (255, 0, 255)

   If you operate on a slice, you also can use arbitrary sequences or other
   PixelArray objects to modify the pixels. They have to match the size of the
   PixelArray however.

   ::

     pxarray[a:b] = 0xFF00FF                   # set all pixels to 0xFF00FF
     pxarray[a:b] = (0xFF00FF, 0xAACCEE, ... ) # first pixel = 0xFF00FF,
                                               # second pixel  = 0xAACCEE, ...
     pxarray[a:b] = ((255, 0, 255), (170, 204, 238), ...) # same as above
     pxarray[a:b] = ((255, 0, 255), 0xAACCEE, ...)        # same as above
     pxarray[a:b] = otherarray[x:y]            # slice sizes must match

   Note, that something like

   ::

     pxarray[2:4, 3:5] = ...

   will not cause a rectangular manipulation. Instead it will be first sliced
   to a two-column array, which then shall be sliced by columns once more,
   which will fail due an IndexError. This is caused by the slicing mechanisms
   in python and an absolutely correct behaviour. Create a single columned
   slice first, which you can manipulate then:

   ::

     pxarray[2, 3:5] = ...
     pxarray[3, 3:5] = ...

   If you want to make a rectangular manipulation or create a view of a part of
   the PixelArray, you also can use the subscript abilities. You can easily
   create different view by creating 'subarrays' using the subscripts.

   ::

     # Create some new PixelArray objects providing a different view
     # of the original array/surface.
     newarray = pxarray[2:4, 3:5]
     otherarray = pxarray[::2, ::2]

   Subscripts also can be used to do fast rectangular pixel manipulations
   instead of iterating over the x or y axis as above.

   ::

     pxarray[::2,:] = (0, 0, 0)                # Make each second column black.

   During its lifetime, the PixelArray locks the surface, thus you explicitly
   have to delete it once its not used anymore and the surface should perform
   operations in the same scope.

   New in pygame 1.9.2 - array struct interface

   Changed in pyame 1.9.2 - It is now possible to have a PixelArray with
   a dimension of length one. Only an integer index on a 2D PixelArray
   returns a 1D array.

   .. versionadded: 1.8.0
      Subscript support

   .. versionadded: 1.8.1
      Methods :meth:`make_surface`, :meth:`replace`, :meth:`extract`, and
      :meth:`compare`

   .. versionadded: 1.9.2
      Properties :attr:`itemsize`, :attr:`ndim`, :attr:`shape`,
      and :attr:`strides`

   .. versionadded: 1.9.2
      Array struct interface


   .. attribute:: surface

      | :sl:`Gets the Surface the PixelArray uses.`
      | :sg:`surface -> Surface`

      The Surface the PixelArray was created for.

      .. ## PixelArray.surface ##

   .. attribute:: itemsize

      | :sl:`Returns the byte size of a pixel array item`
      | :sg:`itemsize -> int`

      This is the same as :meth:`Surface.get_bytesize` for the
      pixel array's surface.

      New in pygame 1.9.2

   .. attribute:: ndim

      | :sl:`Returns the number of dimensions.`
      | :sg:`ndim -> int`

      A pixel array can be 1 or 2 dimensional.

      New in pygame 1.9.2

   .. attribute:: shape

      | :sl:`Returns the array size.`
      | :sg:`shape -> tuple of int's`

      A tuple or length :attr:`ndim` giving the length of each
      dimension. Analogous to :meth:`Surface.get_size`.

      New in pygame 1.9.2

   .. attribute:: strides

      | :sl:`Returns byte offsets for each array dimension.`
      | :sg:`strides -> tuple of int's`

      A tuple or length :attr:`ndim` byte counts. When a stride is
      multiplied by the corresponding index it gives the offset
      of that index from the start of the array. A stride is negative
      for an array that has is inverted (has a negative step).
 
      New in pygame 1.9.2

   .. method:: make_surface

      | :sl:`Creates a new Surface from the current PixelArray.`
      | :sg:`make_surface () -> Surface`

      Creates a new Surface from the current PixelArray. Depending on the
      current PixelArray the size, pixel order etc. will be different from the
      original Surface.

      ::

        # Create a new surface flipped around the vertical axis.
        sf = pxarray[:,::-1].make_surface ()

      New in pygame 1.8.1.

      .. ## PixelArray.make_surface ##

   .. method:: replace

      | :sl:`Replaces the passed color in the PixelArray with another one.`
      | :sg:`replace (color, repcolor, distance=0, weights=(0.299, 0.587, 0.114)) -> None`

      Replaces the pixels with the passed color in the PixelArray by changing
      them them to the passed replacement color.

      It uses a simple weighted euclidian distance formula to calculate the
      distance between the colors. The distance space ranges from 0.0 to 1.0
      and is used as threshold for the color detection. This causes the
      replacement to take pixels with a similar, but not exactly identical
      color, into account as well.

      This is an in place operation that directly affects the pixels of the
      PixelArray.

      New in pygame 1.8.1.

      .. ## PixelArray.replace ##

   .. method:: extract

      | :sl:`Extracts the passed color from the PixelArray.`
      | :sg:`extract (color, distance=0, weights=(0.299, 0.587, 0.114)) -> PixelArray`

      Extracts the passed color by changing all matching pixels to white, while
      non-matching pixels are changed to black. This returns a new PixelArray
      with the black/white color mask.

      It uses a simple weighted euclidian distance formula to calculate the
      distance between the colors. The distance space ranges from 0.0 to 1.0
      and is used as threshold for the color detection. This causes the
      extraction to take pixels with a similar, but not exactly identical
      color, into account as well.

      New in pygame 1.8.1.

      .. ## PixelArray.extract ##

   .. method:: compare

      | :sl:`Compares the PixelArray with another one.`
      | :sg:`compare (array, distance=0, weights=(0.299, 0.587, 0.114)) -> PixelArray`

      Compares the contents of the PixelArray with those from the passed
      PixelArray. It returns a new PixelArray with a black/white color mask
      that indicates the differences (white) of both arrays. Both PixelArray
      objects must have indentical bit depths and dimensions.

      It uses a simple weighted euclidian distance formula to calculate the
      distance between the colors. The distance space ranges from 0.0 to 1.0
      and is used as threshold for the color detection. This causes the
      comparision to mark pixels with a similar, but not exactly identical
      color, as black.

      New in pygame 1.8.1.

      .. ## PixelArray.compare ##

   .. ## pygame.PixelArray ##
