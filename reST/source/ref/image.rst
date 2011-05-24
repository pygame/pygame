.. include:: common.txt

:mod:`pygame.image`
===================

.. module:: pygame.image
   :synopsis: pygame module for image transfer

| :sl:`pygame module for image transfer`

The image module contains functions for loading and saving pictures, as well as
transferring Surfaces to formats usable by other packages.

Note that there is no Image class; an image is loaded as a Surface object. The
Surface class allows manipulation (drawing lines, setting pixels, capturing
regions, etc.).

The image module is a required dependency of Pygame, but it only optionally
supports any extended file formats. By default it can only load uncompressed
``BMP`` images. When built with full image support, the ``pygame.image.load()``
function can support the following formats.

   * ``JPG``

   * ``PNG``

   * ``GIF`` (non animated)

   * ``BMP``

   * ``PCX``

   * ``TGA`` (uncompressed)

   * ``TIF``

   * ``LBM`` (and ``PBM``)

   * ``PBM`` (and ``PGM``, ``PPM``)

   * ``XPM``

Saving images only supports a limited set of formats. You can save to the
following formats.

   * ``BMP``

   * ``TGA``

   * ``PNG``

   * ``JPEG``

``PNG``, ``JPEG`` saving new in pygame 1.8.

.. function:: load

   | :sl:`load new image from a file`
   | :sg:`load(filename) -> Surface`
   | :sg:`load(fileobj, namehint="") -> Surface`

   Load an image from a file source. You can pass either a filename or a Python
   file-like object.

   Pygame will automatically determine the image type (e.g., ``GIF`` or bitmap)
   and create a new Surface object from the data. In some cases it will need to
   know the file extension (e.g., ``GIF`` images should end in ".gif"). If you
   pass a raw file-like object, you may also want to pass the original filename
   as the namehint argument.

   The returned Surface will contain the same color format, colorkey and alpha
   transparency as the file it came from. You will often want to call
   ``Surface.convert()`` with no arguments, to create a copy that will draw
   more quickly on the screen.

   For alpha transparency, like in .png images use the ``convert_alpha()``
   method after loading so that the image has per pixel transparency.

   Pygame may not always be built to support all image formats. At minimum it
   will support uncompressed ``BMP``. If ``pygame.image.get_extended()``
   returns 'True', you should be able to load most images (including png, jpg
   and gif).

   You should use ``os.path.join()`` for compatibility.

   ::

     eg. asurf = pygame.image.load(os.path.join('data', 'bla.png'))

   .. ## pygame.image.load ##

.. function:: save

   | :sl:`save an image to disk`
   | :sg:`save(Surface, filename) -> None`

   This will save your Surface as either a ``BMP``, ``TGA``, ``PNG``, or
   ``JPEG`` image. If the filename extension is unrecognized it will default to
   ``TGA``. Both ``TGA``, and ``BMP`` file formats create uncompressed files.

   ``PNG``, ``JPEG`` saving new in pygame 1.8.

   .. ## pygame.image.save ##

.. function:: get_extended

   | :sl:`test if extended image formats can be loaded`
   | :sg:`get_extended() -> bool`

   If pygame is built with extended image formats this function will return
   True. It is still not possible to determine which formats will be available,
   but generally you will be able to load them all.

   .. ## pygame.image.get_extended ##

.. function:: tostring

   | :sl:`transfer image to string buffer`
   | :sg:`tostring(Surface, format, flipped=False) -> string`

   Creates a string that can be transferred with the 'fromstring' method in
   other Python imaging packages. Some Python image packages prefer their
   images in bottom-to-top format (PyOpenGL for example). If you pass True for
   the flipped argument, the string buffer will be vertically flipped.

   The format argument is a string of one of the following values. Note that
   only 8bit Surfaces can use the "P" format. The other formats will work for
   any Surface. Also note that other Python image packages support more formats
   than Pygame.

      * ``P``, 8bit palettized Surfaces

      * ``RGB``, 24bit image

      * ``RGBX``, 32bit image with unused space

      * ``RGBA``, 32bit image with an alpha channel

      * ``ARGB``, 32bit image with alpha channel first

      * ``RGBA_PREMULT``, 32bit image with colors scaled by alpha channel

      * ``ARGB_PREMULT``, 32bit image with colors scaled by alpha channel, alpha channel first

   .. ## pygame.image.tostring ##

.. function:: fromstring

   | :sl:`create new Surface from a string buffer`
   | :sg:`fromstring(string, size, format, flipped=False) -> Surface`

   This function takes arguments similar to ``pygame.image.tostring()``. The
   size argument is a pair of numbers representing the width and height. Once
   the new Surface is created you can destroy the string buffer.

   The size and format image must compute the exact same size as the passed
   string buffer. Otherwise an exception will be raised.

   See the ``pygame.image.frombuffer()`` method for a potentially faster way to
   transfer images into Pygame.

   .. ## pygame.image.fromstring ##

.. function:: frombuffer

   | :sl:`create a new Surface that shares data inside a string buffer`
   | :sg:`frombuffer(string, size, format) -> Surface`

   Create a new Surface that shares pixel data directly from the string buffer.
   This method takes the same arguments as ``pygame.image.fromstring()``, but
   is unable to vertically flip the source data.

   This will run much faster than :func:`pygame.image.fromstring`, since no
   pixel data must be allocated and copied.

   .. ## pygame.image.frombuffer ##

.. ## pygame.image ##
