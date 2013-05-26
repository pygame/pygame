.. include:: common.txt

:mod:`pygame.Surface`
=====================

.. currentmodule:: pygame

.. class:: Surface

   | :sl:`pygame object for representing images`
   | :sg:`Surface((width, height), flags=0, depth=0, masks=None) -> Surface`
   | :sg:`Surface((width, height), flags=0, Surface) -> Surface`

   A pygame Surface is used to represent any image. The Surface has a fixed
   resolution and pixel format. Surfaces with 8bit pixels use a color palette
   to map to 24bit color.

   Call ``pygame.Surface()`` to create a new image object. The Surface will be
   cleared to all black. The only required arguments are the sizes. With no
   additional arguments, the Surface will be created in a format that best
   matches the display Surface.

   The pixel format can be controlled by passing the bit depth or an existing
   Surface. The flags argument is a bitmask of additional features for the
   surface. You can pass any combination of these flags:

   ::

     HWSURFACE, creates the image in video memory
     SRCALPHA, the pixel format will include a per-pixel alpha

   Both flags are only a request, and may not be possible for all displays and
   formats.

   Advance users can combine a set of bitmasks with a depth value. The masks
   are a set of 4 integers representing which bits in a pixel will represent
   each color. Normal Surfaces should not require the masks argument.

   Surfaces can have many extra attributes like alpha planes, colorkeys, source
   rectangle clipping. These functions mainly effect how the Surface is blitted
   to other Surfaces. The blit routines will attempt to use hardware
   acceleration when possible, otherwise they will use highly optimized
   software blitting methods.

   There are three types of transparency supported in Pygame: colorkeys,
   surface alphas, and pixel alphas. Surface alphas can be mixed with
   colorkeys, but an image with per pixel alphas cannot use the other modes.
   Colorkey transparency makes a single color value transparent. Any pixels
   matching the colorkey will not be drawn. The surface alpha value is a single
   value that changes the transparency for the entire image. A surface alpha of
   255 is opaque, and a value of 0 is completely transparent.

   Per pixel alphas are different because they store a transparency value for
   every pixel. This allows for the most precise transparency effects, but it
   also the slowest. Per pixel alphas cannot be mixed with surface alpha and
   colorkeys.

   There is support for pixel access for the Surfaces. Pixel access on hardware
   surfaces is slow and not recommended. Pixels can be accessed using the
   ``get_at()`` and ``set_at()`` functions. These methods are fine for simple
   access, but will be considerably slow when doing of pixel work with them. If
   you plan on doing a lot of pixel level work, it is recommended to use a
   :class:`pygame.PixelArray`, which gives an array like view of the surface.
   For involved mathematical manipulations try the :mod:`pygame.surfarray`
   module (It's quite quick, but requires NumPy.)

   Any functions that directly access a surface's pixel data will need that
   surface to be lock()'ed. These functions can ``lock()`` and ``unlock()`` the
   surfaces themselves without assistance. But, if a function will be called
   many times, there will be a lot of overhead for multiple locking and
   unlocking of the surface. It is best to lock the surface manually before
   making the function call many times, and then unlocking when you are
   finished. All functions that need a locked surface will say so in their
   docs. Remember to leave the Surface locked only while necessary.

   Surface pixels are stored internally as a single number that has all the
   colors encoded into it. Use the ``Surface.map_rgb()`` and
   ``Surface.unmap_rgb()`` to convert between individual red, green, and blue
   values into a packed integer for that Surface.

   Surfaces can also reference sections of other Surfaces. These are created
   with the ``Surface.subsurface()`` method. Any change to either Surface will
   effect the other.

   Each Surface contains a clipping area. By default the clip area covers the
   entire Surface. If it is changed, all drawing operations will only effect
   the smaller area.

   .. method:: blit

      | :sl:`draw one image onto another`
      | :sg:`blit(source, dest, area=None, special_flags = 0) -> Rect`

      Draws a source Surface onto this Surface. The draw can be positioned with
      the dest argument. Dest can either be pair of coordinates representing
      the upper left corner of the source. A Rect can also be passed as the
      destination and the topleft corner of the rectangle will be used as the
      position for the blit. The size of the destination rectangle does not
      effect the blit.

      An optional area rectangle can be passed as well. This represents a
      smaller portion of the source Surface to draw.

      An optional special flags is for passing in new in 1.8.0: ``BLEND_ADD``,
      ``BLEND_SUB``, ``BLEND_MULT``, ``BLEND_MIN``, ``BLEND_MAX`` new in 1.8.1:
      ``BLEND_RGBA_ADD``, ``BLEND_RGBA_SUB``, ``BLEND_RGBA_MULT``,
      ``BLEND_RGBA_MIN``, ``BLEND_RGBA_MAX`` ``BLEND_RGB_ADD``,
      ``BLEND_RGB_SUB``, ``BLEND_RGB_MULT``, ``BLEND_RGB_MIN``,
      ``BLEND_RGB_MAX`` With other special blitting flags perhaps added in the
      future.

      The return rectangle is the area of the affected pixels, excluding any
      pixels outside the destination Surface, or outside the clipping area.

      Pixel alphas will be ignored when blitting to an 8 bit Surface.

      special_flags new in pygame 1.8.

      For a surface with colorkey or blanket alpha, a blit to self may give
      slightly different colors than a non self-blit.

      .. ## Surface.blit ##

   .. method:: convert

      | :sl:`change the pixel format of an image`
      | :sg:`convert(Surface) -> Surface`
      | :sg:`convert(depth, flags=0) -> Surface`
      | :sg:`convert(masks, flags=0) -> Surface`
      | :sg:`convert() -> Surface`

      Creates a new copy of the Surface with the pixel format changed. The new
      pixel format can be determined from another existing Surface. Otherwise
      depth, flags, and masks arguments can be used, similar to the
      ``pygame.Surface()`` call.

      If no arguments are passed the new Surface will have the same pixel
      format as the display Surface. This is always the fastest format for
      blitting. It is a good idea to convert all Surfaces before they are
      blitted many times.

      The converted Surface will have no pixel alphas. They will be stripped if
      the original had them. See ``Surface.convert_alpha()`` for preserving or
      creating per-pixel alphas.

      .. ## Surface.convert ##

   .. method:: convert_alpha

      | :sl:`change the pixel format of an image including per pixel alphas`
      | :sg:`convert_alpha(Surface) -> Surface`
      | :sg:`convert_alpha() -> Surface`

      Creates a new copy of the surface with the desired pixel format. The new
      surface will be in a format suited for quick blitting to the given format
      with per pixel alpha. If no surface is given, the new surface will be
      optimized for blitting to the current display.

      Unlike the ``Surface.convert()`` method, the pixel format for the new
      image will not be exactly the same as the requested source, but it will
      be optimized for fast alpha blitting to the destination.

      .. ## Surface.convert_alpha ##

   .. method:: copy

      | :sl:`create a new copy of a Surface`
      | :sg:`copy() -> Surface`

      Makes a duplicate copy of a Surface. The new Surface will have the same
      pixel formats, color palettes, and transparency settings as the original.

      .. ## Surface.copy ##

   .. method:: fill

      | :sl:`fill Surface with a solid color`
      | :sg:`fill(color, rect=None, special_flags=0) -> Rect`

      Fill the Surface with a solid color. If no rect argument is given the
      entire Surface will be filled. The rect argument will limit the fill to a
      specific area. The fill will also be contained by the Surface clip area.

      The color argument can be either a ``RGB`` sequence, a ``RGBA`` sequence
      or a mapped color index. If using ``RGBA``, the Alpha (A part of
      ``RGBA``) is ignored unless the surface uses per pixel alpha (Surface has
      the ``SRCALPHA`` flag).

      An optional special_flags is for passing in new in 1.8.0: ``BLEND_ADD``,
      ``BLEND_SUB``, ``BLEND_MULT``, ``BLEND_MIN``, ``BLEND_MAX`` new in 1.8.1:
      ``BLEND_RGBA_ADD``, ``BLEND_RGBA_SUB``, ``BLEND_RGBA_MULT``,
      ``BLEND_RGBA_MIN``, ``BLEND_RGBA_MAX`` ``BLEND_RGB_ADD``,
      ``BLEND_RGB_SUB``, ``BLEND_RGB_MULT``, ``BLEND_RGB_MIN``,
      ``BLEND_RGB_MAX`` With other special blitting flags perhaps added in the
      future.

      This will return the affected Surface area.

      .. ## Surface.fill ##

   .. method:: scroll

      | :sl:`Shift the surface image in place`
      | :sg:`scroll(dx=0, dy=0) -> None`

      Move the image by dx pixels right and dy pixels down. dx and dy may be
      negative for left and up scrolls respectively. Areas of the surface that
      are not overwritten retain their original pixel values. Scrolling is
      contained by the Surface clip area. It is safe to have dx and dy values
      that exceed the surface size.

      New in Pygame 1.9

      .. ## Surface.scroll ##

   .. method:: set_colorkey

      | :sl:`Set the transparent colorkey`
      | :sg:`set_colorkey(Color, flags=0) -> None`
      | :sg:`set_colorkey(None) -> None`

      Set the current color key for the Surface. When blitting this Surface
      onto a destination, and pixels that have the same color as the colorkey
      will be transparent. The color can be an ``RGB`` color or a mapped color
      integer. If None is passed, the colorkey will be unset.

      The colorkey will be ignored if the Surface is formatted to use per pixel
      alpha values. The colorkey can be mixed with the full Surface alpha
      value.

      The optional flags argument can be set to ``pygame.RLEACCEL`` to provide
      better performance on non accelerated displays. An ``RLEACCEL`` Surface
      will be slower to modify, but quicker to blit as a source.

      .. ## Surface.set_colorkey ##

   .. method:: get_colorkey

      | :sl:`Get the current transparent colorkey`
      | :sg:`get_colorkey() -> RGB or None`

      Return the current colorkey value for the Surface. If the colorkey is not
      set then None is returned.

      .. ## Surface.get_colorkey ##

   .. method:: set_alpha

      | :sl:`set the alpha value for the full Surface image`
      | :sg:`set_alpha(value, flags=0) -> None`
      | :sg:`set_alpha(None) -> None`

      Set the current alpha value fo r the Surface. When blitting this Surface
      onto a destination, the pixels will be drawn slightly transparent. The
      alpha value is an integer from 0 to 255, 0 is fully transparent and 255
      is fully opaque. If None is passed for the alpha value, then the Surface
      alpha will be disabled.

      This value is different than the per pixel Surface alpha. If the Surface
      format contains per pixel alphas, then this alpha value will be ignored.
      If the Surface contains per pixel alphas, setting the alpha value to None
      will disable the per pixel transparency.

      The optional flags argument can be set to ``pygame.RLEACCEL`` to provide
      better performance on non accelerated displays. An ``RLEACCEL`` Surface
      will be slower to modify, but quicker to blit as a source.

      .. ## Surface.set_alpha ##

   .. method:: get_alpha

      | :sl:`get the current Surface transparency value`
      | :sg:`get_alpha() -> int_value or None`

      Return the current alpha value for the Surface. If the alpha value is not
      set then None is returned.

      .. ## Surface.get_alpha ##

   .. method:: lock

      | :sl:`lock the Surface memory for pixel access`
      | :sg:`lock() -> None`

      Lock the pixel data of a Surface for access. On accelerated Surfaces, the
      pixel data may be stored in volatile video memory or nonlinear compressed
      forms. When a Surface is locked the pixel memory becomes available to
      access by regular software. Code that reads or writes pixel values will
      need the Surface to be locked.

      Surfaces should not remain locked for more than necessary. A locked
      Surface can often not be displayed or managed by Pygame.

      Not all Surfaces require locking. The ``Surface.mustlock()`` method can
      determine if it is actually required. There is no performance penalty for
      locking and unlocking a Surface that does not need it.

      All pygame functions will automatically lock and unlock the Surface data
      as needed. If a section of code is going to make calls that will
      repeatedly lock and unlock the Surface many times, it can be helpful to
      wrap the block inside a lock and unlock pair.

      It is safe to nest locking and unlocking calls. The surface will only be
      unlocked after the final lock is released.

      .. ## Surface.lock ##

   .. method:: unlock

      | :sl:`unlock the Surface memory from pixel access`
      | :sg:`unlock() -> None`

      Unlock the Surface pixel data after it has been locked. The unlocked
      Surface can once again be drawn and managed by Pygame. See the
      ``Surface.lock()`` documentation for more details.

      All pygame functions will automatically lock and unlock the Surface data
      as needed. If a section of code is going to make calls that will
      repeatedly lock and unlock the Surface many times, it can be helpful to
      wrap the block inside a lock and unlock pair.

      It is safe to nest locking and unlocking calls. The surface will only be
      unlocked after the final lock is released.

      .. ## Surface.unlock ##

   .. method:: mustlock

      | :sl:`test if the Surface requires locking`
      | :sg:`mustlock() -> bool`

      Returns True if the Surface is required to be locked to access pixel
      data. Usually pure software Surfaces do not require locking. This method
      is rarely needed, since it is safe and quickest to just lock all Surfaces
      as needed.

      All pygame functions will automatically lock and unlock the Surface data
      as needed. If a section of code is going to make calls that will
      repeatedly lock and unlock the Surface many times, it can be helpful to
      wrap the block inside a lock and unlock pair.

      .. ## Surface.mustlock ##

   .. method:: get_locked

      | :sl:`test if the Surface is current locked`
      | :sg:`get_locked() -> bool`

      Returns True when the Surface is locked. It doesn't matter how many times
      the Surface is locked.

      .. ## Surface.get_locked ##

   .. method:: get_locks

      | :sl:`Gets the locks for the Surface`
      | :sg:`get_locks() -> tuple`

      Returns the currently existing locks for the Surface.

      .. ## Surface.get_locks ##

   .. method:: get_at

      | :sl:`get the color value at a single pixel`
      | :sg:`get_at((x, y)) -> Color`

      Return a copy of the ``RGBA`` Color value at the given pixel. If the
      Surface has no per pixel alpha, then the alpha value will always be 255
      (opaque). If the pixel position is outside the area of the Surface an
      IndexError exception will be raised.

      Getting and setting pixels one at a time is generally too slow to be used
      in a game or realtime situation. It is better to use methods which
      operate on many pixels at a time like with the blit, fill and draw
      methods - or by using surfarray/PixelArray.

      This function will temporarily lock and unlock the Surface as needed.

      Returning a Color instead of tuple, New in pygame 1.9.0. Use
      ``tuple(surf.get_at((x,y)))`` if you want a tuple, and not a Color. This
      should only matter if you want to use the color as a key in a dict.

      .. ## Surface.get_at ##

   .. method:: set_at

      | :sl:`set the color value for a single pixel`
      | :sg:`set_at((x, y), Color) -> None`

      Set the ``RGBA`` or mapped integer color value for a single pixel. If the
      Surface does not have per pixel alphas, the alpha value is ignored.
      Settting pixels outside the Surface area or outside the Surface clipping
      will have no effect.

      Getting and setting pixels one at a time is generally too slow to be used
      in a game or realtime situation.

      This function will temporarily lock and unlock the Surface as needed.

      .. ## Surface.set_at ##

   .. method:: get_at_mapped

      | :sl:`get the mapped color value at a single pixel`
      | :sg:`get_at_mapped((x, y)) -> Color`

      Return the integer value of the given pixel. If the pixel position is
      outside the area of the Surface an IndexError exception will be raised.

      This method is intended for Pygame unit testing. It unlikely has any use
      in an application.

      This function will temporarily lock and unlock the Surface as needed.

      New in pygame. 1.9.2.

      .. ## Surface.get_at_mapped ##

   .. method:: get_palette

      | :sl:`get the color index palette for an 8bit Surface`
      | :sg:`get_palette() -> [RGB, RGB, RGB, ...]`

      Return a list of up to 256 color elements that represent the indexed
      colors used in an 8bit Surface. The returned list is a copy of the
      palette, and changes will have no effect on the Surface.

      Returning a list of ``Color(with length 3)`` instances instead of tuples,
      New in pygame 1.9.0

      .. ## Surface.get_palette ##

   .. method:: get_palette_at

      | :sl:`get the color for a single entry in a palette`
      | :sg:`get_palette_at(index) -> RGB`

      Returns the red, green, and blue color values for a single index in a
      Surface palette. The index should be a value from 0 to 255.

      Returning ``Color(with length 3)`` instance instead of a tuple, New in
      pygame 1.9.0

      .. ## Surface.get_palette_at ##

   .. method:: set_palette

      | :sl:`set the color palette for an 8bit Surface`
      | :sg:`set_palette([RGB, RGB, RGB, ...]) -> None`

      Set the full palette for an 8bit Surface. This will replace the colors in
      the existing palette. A partial palette can be passed and only the first
      colors in the original palette will be changed.

      This function has no effect on a Surface with more than 8bits per pixel.

      .. ## Surface.set_palette ##

   .. method:: set_palette_at

      | :sl:`set the color for a single index in an 8bit Surface palette`
      | :sg:`set_palette_at(index, RGB) -> None`

      Set the palette value for a single entry in a Surface palette. The index
      should be a value from 0 to 255.

      This function has no effect on a Surface with more than 8bits per pixel.

      .. ## Surface.set_palette_at ##

   .. method:: map_rgb

      | :sl:`convert a color into a mapped color value`
      | :sg:`map_rgb(Color) -> mapped_int`

      Convert an ``RGBA`` color into the mapped integer value for this Surface.
      The returned integer will contain no more bits than the bit depth of the
      Surface. Mapped color values are not often used inside Pygame, but can be
      passed to most functions that require a Surface and a color.

      See the Surface object documentation for more information about colors
      and pixel formats.

      .. ## Surface.map_rgb ##

   .. method:: unmap_rgb

      | :sl:`convert a mapped integer color value into a Color`
      | :sg:`unmap_rgb(mapped_int) -> Color`

      Convert an mapped integer color into the ``RGB`` color components for
      this Surface. Mapped color values are not often used inside Pygame, but
      can be passed to most functions that require a Surface and a color.

      See the Surface object documentation for more information about colors
      and pixel formats.

      .. ## Surface.unmap_rgb ##

   .. method:: set_clip

      | :sl:`set the current clipping area of the Surface`
      | :sg:`set_clip(rect) -> None`
      | :sg:`set_clip(None) -> None`

      Each Surface has an active clipping area. This is a rectangle that
      represents the only pixels on the Surface that can be modified. If None
      is passed for the rectangle the full Surface will be available for
      changes.

      The clipping area is always restricted to the area of the Surface itself.
      If the clip rectangle is too large it will be shrunk to fit inside the
      Surface.

      .. ## Surface.set_clip ##

   .. method:: get_clip

      | :sl:`get the current clipping area of the Surface`
      | :sg:`get_clip() -> Rect`

      Return a rectangle of the current clipping area. The Surface will always
      return a valid rectangle that will never be outside the bounds of the
      image. If the Surface has had None set for the clipping area, the Surface
      will return a rectangle with the full area of the Surface.

      .. ## Surface.get_clip ##

   .. method:: subsurface

      | :sl:`create a new surface that references its parent`
      | :sg:`subsurface(Rect) -> Surface`

      Returns a new Surface that shares its pixels with its new parent. The new
      Surface is considered a child of the original. Modifications to either
      Surface pixels will effect each other. Surface information like clipping
      area and color keys are unique to each Surface.

      The new Surface will inherit the palette, color key, and alpha settings
      from its parent.

      It is possible to have any number of subsurfaces and subsubsurfaces on
      the parent. It is also possible to subsurface the display Surface if the
      display mode is not hardware accelerated.

      See the ``Surface.get_offset()``, ``Surface.get_parent()`` to learn more
      about the state of a subsurface.

      .. ## Surface.subsurface ##

   .. method:: get_parent

      | :sl:`find the parent of a subsurface`
      | :sg:`get_parent() -> Surface`

      Returns the parent Surface of a subsurface. If this is not a subsurface
      then None will be returned.

      .. ## Surface.get_parent ##

   .. method:: get_abs_parent

      | :sl:`find the top level parent of a subsurface`
      | :sg:`get_abs_parent() -> Surface`

      Returns the parent Surface of a subsurface. If this is not a subsurface
      then this surface will be returned.

      .. ## Surface.get_abs_parent ##

   .. method:: get_offset

      | :sl:`find the position of a child subsurface inside a parent`
      | :sg:`get_offset() -> (x, y)`

      Get the offset position of a child subsurface inside of a parent. If the
      Surface is not a subsurface this will return (0, 0).

      .. ## Surface.get_offset ##

   .. method:: get_abs_offset

      | :sl:`find the absolute position of a child subsurface inside its top level parent`
      | :sg:`get_abs_offset() -> (x, y)`

      Get the offset position of a child subsurface inside of its top level
      parent Surface. If the Surface is not a subsurface this will return (0,
      0).

      .. ## Surface.get_abs_offset ##

   .. method:: get_size

      | :sl:`get the dimensions of the Surface`
      | :sg:`get_size() -> (width, height)`

      Return the width and height of the Surface in pixels.

      .. ## Surface.get_size ##

   .. method:: get_width

      | :sl:`get the width of the Surface`
      | :sg:`get_width() -> width`

      Return the width of the Surface in pixels.

      .. ## Surface.get_width ##

   .. method:: get_height

      | :sl:`get the height of the Surface`
      | :sg:`get_height() -> height`

      Return the height of the Surface in pixels.

      .. ## Surface.get_height ##

   .. method:: get_rect

      | :sl:`get the rectangular area of the Surface`
      | :sg:`get_rect(**kwargs) -> Rect`

      Returns a new rectangle covering the entire surface. This rectangle will
      always start at 0, 0 with a width. and height the same size as the image.

      You can pass keyword argument values to this function. These named values
      will be applied to the attributes of the Rect before it is returned. An
      example would be 'mysurf.get_rect(center=(100,100))' to create a
      rectangle for the Surface centered at a given position.

      .. ## Surface.get_rect ##

   .. method:: get_bitsize

      | :sl:`get the bit depth of the Surface pixel format`
      | :sg:`get_bitsize() -> int`

      Returns the number of bits used to represent each pixel. This value may
      not exactly fill the number of bytes used per pixel. For example a 15 bit
      Surface still requires a full 2 bytes.

      .. ## Surface.get_bitsize ##

   .. method:: get_bytesize

      | :sl:`get the bytes used per Surface pixel`
      | :sg:`get_bytesize() -> int`

      Return the number of bytes used per pixel.

      .. ## Surface.get_bytesize ##

   .. method:: get_flags

      | :sl:`get the additional flags used for the Surface`
      | :sg:`get_flags() -> int`

      Returns a set of current Surface features. Each feature is a bit in the
      flags bitmask. Typical flags are ``HWSURFACE``, ``RLEACCEL``,
      ``SRCALPHA``, and ``SRCCOLORKEY``.

      Here is a more complete list of flags. A full list can be found in
      ``SDL_video.h``

      ::

        SWSURFACE	0x00000000	# Surface is in system memory
        HWSURFACE	0x00000001	# Surface is in video memory
        ASYNCBLIT	0x00000004	# Use asynchronous blits if possible

      Available for ``pygame.display.set_mode()``

      ::

        ANYFORMAT	0x10000000	# Allow any video depth/pixel-format
        HWPALETTE	0x20000000	# Surface has exclusive palette
        DOUBLEBUF	0x40000000	# Set up double-buffered video mode
        FULLSCREEN	0x80000000	# Surface is a full screen display
        OPENGL        0x00000002      # Create an OpenGL rendering context
        OPENGLBLIT	0x0000000A	# Create an OpenGL rendering context
                                      #   and use it for blitting.  Obsolete.
        RESIZABLE	0x00000010	# This video mode may be resized
        NOFRAME       0x00000020	# No window caption or edge frame

      Used internally (read-only)

      ::

        HWACCEL       0x00000100	# Blit uses hardware acceleration
        SRCCOLORKEY	0x00001000	# Blit uses a source color key
        RLEACCELOK	0x00002000	# Private flag
        RLEACCEL	0x00004000	# Surface is RLE encoded
        SRCALPHA	0x00010000	# Blit uses source alpha blending
        PREALLOC	0x01000000	# Surface uses preallocated memory

      .. ## Surface.get_flags ##

   .. method:: get_pitch

      | :sl:`get the number of bytes used per Surface row`
      | :sg:`get_pitch() -> int`

      Return the number of bytes separating each row in the Surface. Surfaces
      in video memory are not always linearly packed. Subsurfaces will also
      have a larger pitch than their real width.

      This value is not needed for normal Pygame usage.

      .. ## Surface.get_pitch ##

   .. method:: get_masks

      | :sl:`the bitmasks needed to convert between a color and a mapped integer`
      | :sg:`get_masks() -> (R, G, B, A)`

      Returns the bitmasks used to isolate each color in a mapped integer.

      This value is not needed for normal Pygame usage.

      .. ## Surface.get_masks ##

   .. method:: set_masks

      | :sl:`set the bitmasks needed to convert between a color and a mapped integer`
      | :sg:`set_masks((r,g,b,a)) -> None`

      This is not needed for normal Pygame usage. New in pygame 1.8.1

      .. ## Surface.set_masks ##

   .. method:: get_shifts

      | :sl:`the bit shifts needed to convert between a color and a mapped integer`
      | :sg:`get_shifts() -> (R, G, B, A)`

      Returns the pixel shifts need to convert between each color and a mapped
      integer.

      This value is not needed for normal Pygame usage.

      .. ## Surface.get_shifts ##

   .. method:: set_shifts

      | :sl:`sets the bit shifts needed to convert between a color and a mapped integer`
      | :sg:`set_shifts((r,g,b,a)) -> None`

      This is not needed for normal Pygame usage. New in pygame 1.8.1

      .. ## Surface.set_shifts ##

   .. method:: get_losses

      | :sl:`the significant bits used to convert between a color and a mapped integer`
      | :sg:`get_losses() -> (R, G, B, A)`

      Return the least significant number of bits stripped from each color in a
      mapped integer.

      This value is not needed for normal Pygame usage.

      .. ## Surface.get_losses ##

   .. method:: get_bounding_rect

      | :sl:`find the smallest rect containing data`
      | :sg:`get_bounding_rect(min_alpha = 1) -> Rect`

      Returns the smallest rectangular region that contains all the pixels in
      the surface that have an alpha value greater than or equal to the minimum
      alpha value.

      This function will temporarily lock and unlock the Surface as needed.

      New in pygame 1.8.

      .. ## Surface.get_bounding_rect ##

   .. method:: get_buffer

      | :sl:`return a buffer view of the Surface's pixels.`
      | :sg:`get_buffer(<kind>='&') -> BufferProxy`

      Return an object which exports a surface's internal pixel buffer as
      a C level array struct, Python level array interface or a C level 
      buffer interface. The pixel buffer is writeable. The new buffer protocol
      is supported for Python 2.6 and up in CPython. The old buffer protocol
      is also supported for Python 2.x. The old buffer data is in one segment
      for kind '&' and '0', multi-segment for other buffer view kinds.

      The kind argument is the length 1 string '&', '0', '1', '2', '3',
      'r', 'g', 'b', or 'a'. The letters are case insensitive;
      'A' will work as well. The argument can be either a Unicode or byte (char)
      string. The default is '&'.

      A kind '&' view is unstructured bytes. The surface pixels are treated
      as a single, contiguous stretch of bytes. No shape or pitch information
      is provided. The importer needs to get this information elsewhere.
      
      '0' returns a continguous unstructured bytes view. No surface shape
      information is given. A ValueError is raised if the surface's pixels
      are discontinuous.
      
      '1' returns a (surface-width * surface-height) array of continguous
      pixels. A ValueError is raised if the surface pixels are discontinuous.
      
      '2' returns a (surface-width, surface-height) array of raw pixels.
      The pixels are surface bytesized unsigned integers. The pixel format is
      surface specific. The 3 byte unsigned integers of 24 bit surfaces are
      unlikely accepted by anything other than other Pygame functions.

      '3' returns a (surface-width, surface-height, 3) array of ``RGB`` color
      components. Each of the red, green, and blue components are unsigned
      bytes. Only 24-bit and 32-bit surfaces are supported. The color
      components must be in either ``RGB`` or ``BGR`` order within the pixel.

      'r' for red, 'g' for green, 'b' for blue, and 'a' for alpha return a
      (surface-width, surface-height) view of a single color component within a
      surface: a color plane. Color components are unsigned bytes. Both 24-bit
      and 32-bit surfaces support 'r', 'g', and 'b'. Only 32-bit surfaces with
      ``SRCALPHA`` support 'a'.

      For kind '&', the method call also locks the surface. The lock is released
      when the BufferProxy object is deleted. With all other kinds, the surface
      is locked only when an exposed interface is accessed. For new buffer
      interace accesses, the surface is unlocked once the last buffer view is
      released. For array interface accesses, the surface remains locked until
      the BufferProxy object is released.

      New in pygame 1.8.
      Extended in pygame 1.9.2.

      .. ## Surface.get_buffer ##

   .. ## pygame.Surface ##
