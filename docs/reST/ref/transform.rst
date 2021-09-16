.. include:: common.txt

:mod:`pygame.transform`
=======================

.. module:: pygame.transform
   :synopsis: pygame module to transform surfaces

| :sl:`pygame module to transform surfaces`

A Surface transform is an operation that moves or resizes the pixels. All these
functions take a Surface to operate on and return a new Surface with the
results.

Some of the transforms are considered destructive. These means every time they
are performed they lose pixel data. Common examples of this are resizing and
rotating. For this reason, it is better to re-transform the original surface
than to keep transforming an image multiple times. (For example, suppose you
are animating a bouncing spring which expands and contracts. If you applied the
size changes incrementally to the previous images, you would lose detail.
Instead, always begin with the original image and scale to the desired size.)

.. function:: flip

   | :sl:`flip vertically and horizontally`
   | :sg:`flip(Surface, xbool, ybool) -> Surface`

   This can flip a Surface either vertically, horizontally, or both. Flipping a
   Surface is non-destructive and returns a new Surface with the same
   dimensions.

   .. ## pygame.transform.flip ##

.. function:: scale

   | :sl:`resize to new resolution`
   | :sg:`scale(Surface, (width, height), DestSurface = None) -> Surface`

   Resizes the Surface to a new resolution. This is a fast scale operation that
   does not sample the results.

   An optional destination surface can be used, rather than have it create a
   new one. This is quicker if you want to repeatedly scale something. However
   the destination must be the same size as the (width, height) passed in. Also
   the destination surface must be the same format.

   .. ## pygame.transform.scale ##

.. function:: rotate

   | :sl:`rotate an image`
   | :sg:`rotate(Surface, angle) -> Surface`

   Unfiltered counterclockwise rotation. The angle argument represents degrees
   and can be any floating point value. Negative angle amounts will rotate
   clockwise.

   Unless rotating by 90 degree increments, the image will be padded larger to
   hold the new size. If the image has pixel alphas, the padded area will be
   transparent. Otherwise pygame will pick a color that matches the Surface
   colorkey or the topleft pixel value.

   .. ## pygame.transform.rotate ##

.. function:: rotozoom

   | :sl:`filtered scale and rotation`
   | :sg:`rotozoom(Surface, angle, scale) -> Surface`

   This is a combined scale and rotation transform. The resulting Surface will
   be a filtered 32-bit Surface. The scale argument is a floating point value
   that will be multiplied by the current resolution. The angle argument is a
   floating point value that represents the counterclockwise degrees to rotate.
   A negative rotation angle will rotate clockwise.

   .. ## pygame.transform.rotozoom ##

.. function:: scale2x

   | :sl:`specialized image doubler`
   | :sg:`scale2x(Surface, DestSurface = None) -> Surface`

   This will return a new image that is double the size of the original. It
   uses the AdvanceMAME Scale2X algorithm which does a 'jaggie-less' scale of
   bitmap graphics.

   This really only has an effect on simple images with solid colors. On
   photographic and antialiased images it will look like a regular unfiltered
   scale.

   An optional destination surface can be used, rather than have it create a
   new one. This is quicker if you want to repeatedly scale something. However
   the destination must be twice the size of the source surface passed in. Also
   the destination surface must be the same format.

   .. ## pygame.transform.scale2x ##

.. function:: smoothscale

   | :sl:`scale a surface to an arbitrary size smoothly`
   | :sg:`smoothscale(Surface, (width, height), DestSurface = None) -> Surface`

   Uses one of two different algorithms for scaling each dimension of the input
   surface as required. For shrinkage, the output pixels are area averages of
   the colors they cover. For expansion, a bilinear filter is used. For the
   x86-64 and i686 architectures, optimized ``MMX`` routines are included and
   will run much faster than other machine types. The size is a 2 number
   sequence for (width, height). This function only works for 24-bit or 32-bit
   surfaces. An exception will be thrown if the input surface bit depth is less
   than 24.

   .. versionadded:: 1.8

   .. ## pygame.transform.smoothscale ##

.. function:: get_smoothscale_backend

   | :sl:`return smoothscale filter version in use: 'GENERIC', 'MMX', or 'SSE'`
   | :sg:`get_smoothscale_backend() -> String`

   Shows whether or not smoothscale is using ``MMX`` or ``SSE`` acceleration.
   If no acceleration is available then "GENERIC" is returned. For a x86
   processor the level of acceleration to use is determined at runtime.

   This function is provided for pygame testing and debugging.

   .. ## pygame.transform.get_smoothscale_backend ##

.. function:: set_smoothscale_backend

   | :sl:`set smoothscale filter version to one of: 'GENERIC', 'MMX', or 'SSE'`
   | :sg:`set_smoothscale_backend(type) -> None`

   Sets smoothscale acceleration. Takes a string argument. A value of 'GENERIC'
   turns off acceleration. 'MMX' uses ``MMX`` instructions only. 'SSE' allows
   ``SSE`` extensions as well. A value error is raised if type is not
   recognized or not supported by the current processor.

   This function is provided for pygame testing and debugging. If smoothscale
   causes an invalid instruction error then it is a pygame/SDL bug that should
   be reported. Use this function as a temporary fix only.

   .. ## pygame.transform.set_smoothscale_backend ##

.. function:: chop

   | :sl:`gets a copy of an image with an interior area removed`
   | :sg:`chop(Surface, rect) -> Surface`

   Extracts a portion of an image. All vertical and horizontal pixels
   surrounding the given rectangle area are removed. The corner areas (diagonal
   to the rect) are then brought together. (The original image is not altered
   by this operation.)

   ``NOTE``: If you want a "crop" that returns the part of an image within a
   rect, you can blit with a rect to a new surface or copy a subsurface.

   .. ## pygame.transform.chop ##

.. function:: laplacian

   | :sl:`find edges in a surface`
   | :sg:`laplacian(Surface, DestSurface = None) -> Surface`

   Finds the edges in a surface using the laplacian algorithm.

   .. versionadded:: 1.8

   .. ## pygame.transform.laplacian ##

.. function:: average_surfaces

   | :sl:`find the average surface from many surfaces.`
   | :sg:`average_surfaces(Surfaces, DestSurface = None, palette_colors = 1) -> Surface`

   Takes a sequence of surfaces and returns a surface with average colors from
   each of the surfaces.

   palette_colors - if true we average the colors in palette, otherwise we
   average the pixel values. This is useful if the surface is actually
   greyscale colors, and not palette colors.

   Note, this function currently does not handle palette using surfaces
   correctly.

   .. versionadded:: 1.8
   .. versionadded:: 1.9 ``palette_colors`` argument

   .. ## pygame.transform.average_surfaces ##

.. function:: average_color

   | :sl:`finds the average color of a surface`
   | :sg:`average_color(Surface, Rect = None) -> Color`

   Finds the average color of a Surface or a region of a surface specified by a
   Rect, and returns it as a Color.

   .. ## pygame.transform.average_color ##

.. function:: threshold

   | :sl:`finds which, and how many pixels in a surface are within a threshold of a 'search_color' or a 'search_surf'.`
   | :sg:`threshold(dest_surf, surf, search_color, threshold=(0,0,0,0), set_color=(0,0,0,0), set_behavior=1, search_surf=None, inverse_set=False) -> num_threshold_pixels`

   This versatile function can be used for find colors in a 'surf' close to a 'search_color'
   or close to colors in a separate 'search_surf'.

   It can also be used to transfer pixels into a 'dest_surf' that match or don't match.

   By default it sets pixels in the 'dest_surf' where all of the pixels NOT within the
   threshold are changed to set_color. If inverse_set is optionally set to True,
   the pixels that ARE within the threshold are changed to set_color.

   If the optional 'search_surf' surface is given, it is used to threshold against
   rather than the specified 'set_color'. That is, it will find each pixel in the
   'surf' that is within the 'threshold' of the pixel at the same coordinates
   of the 'search_surf'.

   :param dest_surf: Surface we are changing. See 'set_behavior'.
    Should be None if counting (set_behavior is 0).
   :type dest_surf: pygame.Surface or None

   :param pygame.Surface surf: Surface we are looking at.

   :param pygame.Color search_color: Color we are searching for.

   :param pygame.Color threshold: Within this distance from search_color (or search_surf).
     You can use a threshold of (r,g,b,a) where the r,g,b can have different
     thresholds. So you could use an r threshold of 40 and a blue threshold of 2
     if you like.

   :param set_color: Color we set in dest_surf.
   :type set_color: pygame.Color or None

   :param int set_behavior:
    - set_behavior=1 (default). Pixels in dest_surface will be changed to 'set_color'.
    - set_behavior=0 we do not change 'dest_surf', just count. Make dest_surf=None.
    - set_behavior=2 pixels set in 'dest_surf' will be from 'surf'.

   :param search_surf:
    - search_surf=None (default). Search against 'search_color' instead.
    - search_surf=Surface. Look at the color in 'search_surf' rather than using 'search_color'.
   :type search_surf: pygame.Surface or None

   :param bool inverse_set:
     - False, default. Pixels outside of threshold are changed.
     - True, Pixels within threshold are changed.

   :rtype: int
   :returns: The number of pixels that are within the 'threshold' in 'surf'
     compared to either 'search_color' or `search_surf`.

   :Examples:

   See the threshold tests for a full of examples: https://github.com/pygame/pygame/blob/master/test/transform_test.py

   .. literalinclude:: ../../../test/transform_test.py
      :pyobject: TransformModuleTest.test_threshold_dest_surf_not_change


   .. versionadded:: 1.8
   .. versionchanged:: 1.9.4
      Fixed a lot of bugs and added keyword arguments. Test your code.

   .. ## pygame.transform.threshold ##

.. ## pygame.transform ##
