:mod:`pygame2.sdlttf.constants` -- Constants for SDL_ttf
========================================================

This module contains the constants used throughout the :mod:`pygame2.sdlttf`
modules.

.. module:: pygame2.sdlttf.constants
   :synopsis: Constants used throughout the :mod:`pygame2.sdlttf` modules.

Style Constants
---------------

Those constants denote the font style and can be used as bit-wise OR'd
combinations.

.. data:: STYLE_NORMAL
   
   The default font style.
   
.. data:: STYLE_BOLD

   Indicates a bold font style.

.. data:: STYLE_ITALIC

   Indicates an italic font style.

.. data:: STYLE_UNDERLINE

   Indicates an underlined font style.

Render Constants
----------------

The following constants are used for rendering text for a certain font using
:meth:`pygame2.sdlttf.Font.render`.

.. data:: RENDER_SOLID
   
   Creates an 8-bit palettized surface and renders  the given text at fast
   quality with the given color. The 0 pixel is the colorkey, giving a
   transparent background, and the 1 pixel is set to the text color.

.. data:: RENDER_SHADED

   Creates an 8-bit palettized surface and renders the given text at high
   quality with the given colors. The 0 pixel is background, other pixels have
   varying degrees of the foreground color.

.. data:: RENDER_BLENDED
   
   Creates a 32-bit ARGB surface and renders the given text at high quality,
   alpha blending to dither the font with the given color.
