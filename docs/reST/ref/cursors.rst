.. include:: common.txt

:mod:`pygame.cursors`
=====================

.. module:: pygame.cursors
   :synopsis: pygame module for cursor resources

| :sl:`pygame module for cursor resources`

Pygame offers control over the system hardware cursor. Pygame only supports
black and white cursors for the system. You control the cursor with functions
inside :mod:`pygame.mouse`.

This cursors module contains functions for loading and decoding various
cursor formats. These allow you to easily store your cursors in external files
or directly as encoded python strings.

The module includes several standard cursors. The ``pygame.mouse.set_cursor()``
function takes several arguments. All those arguments have been stored in a
single tuple you can call like this:

::

   >>> pygame.mouse.set_cursor(*pygame.cursors.arrow)

This module also contains a few cursors as formatted strings. You'll need to
pass these to ``pygame.cursors.compile()`` function before you can use them.
The example call would look like this:

::

   >>> cursor = pygame.cursors.compile(pygame.cursors.textmarker_strings)
   >>> pygame.mouse.set_cursor(*cursor)

The following variables are cursor bitmaps that can be used as cursor:

   * ``pygame.cursors.arrow``

   * ``pygame.cursors.diamond``

   * ``pygame.cursors.broken_x``

   * ``pygame.cursors.tri_left``

   * ``pygame.cursors.tri_right``

The following strings can be converted into cursor bitmaps with
``pygame.cursors.compile()`` :

   * ``pygame.cursors.thickarrow_strings``

   * ``pygame.cursors.sizer_x_strings``

   * ``pygame.cursors.sizer_y_strings``

   * ``pygame.cursors.sizer_xy_strings``

.. function:: compile

   | :sl:`create binary cursor data from simple strings`
   | :sg:`compile(strings, black='X', white='.', xor='o') -> data, mask`

   A sequence of strings can be used to create binary cursor data for the
   system cursor. The return values are the same format needed by
   ``pygame.mouse.set_cursor()``.

   If you are creating your own cursor strings, you can use any value represent
   the black and white pixels. Some system allow you to set a special toggle
   color for the system color, this is also called the xor color. If the system
   does not support xor cursors, that color will simply be black.

   The width of the strings must all be equal and be divisible by 8. An example
   set of cursor strings looks like this

   ::

       thickarrow_strings = (               #sized 24x24
         "XX                      ",
         "XXX                     ",
         "XXXX                    ",
         "XX.XX                   ",
         "XX..XX                  ",
         "XX...XX                 ",
         "XX....XX                ",
         "XX.....XX               ",
         "XX......XX              ",
         "XX.......XX             ",
         "XX........XX            ",
         "XX........XXX           ",
         "XX......XXXXX           ",
         "XX.XXX..XX              ",
         "XXXX XX..XX             ",
         "XX   XX..XX             ",
         "     XX..XX             ",
         "      XX..XX            ",
         "      XX..XX            ",
         "       XXXX             ",
         "       XX               ",
         "                        ",
         "                        ",
         "                        ")

   .. ## pygame.cursors.compile ##

.. function:: load_xbm

   | :sl:`load cursor data from an XBM file`
   | :sg:`load_xbm(cursorfile) -> cursor_args`
   | :sg:`load_xbm(cursorfile, maskfile) -> cursor_args`

   This loads cursors for a simple subset of ``XBM`` files. ``XBM`` files are
   traditionally used to store cursors on UNIX systems, they are an ASCII
   format used to represent simple images.

   Sometimes the black and white color values will be split into two separate
   ``XBM`` files. You can pass a second maskfile argument to load the two
   images into a single cursor.

   The cursorfile and maskfile arguments can either be filenames or file-like
   object with the readlines method.

   The return value cursor_args can be passed directly to the
   ``pygame.mouse.set_cursor()`` function.

   .. ## pygame.cursors.load_xbm ##

.. ## pygame.cursors ##
