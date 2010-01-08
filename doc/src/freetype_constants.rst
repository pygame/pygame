:mod:`pygame2.freetype.constants` -- Constants for the FreeType extension
=========================================================================

This module contains the constants used within the :mod:`pygame2.freetype`
module.

.. module:: pygame2.freetype.constants
   :synopsis: Constants used within the :mod:`pygame2.freetype` module.

Style Constants
---------------

Those constants denote the available font styles for the various
:class:`pygame2.freetype.Font` methods.

.. data:: STYLE_NORMAL
   
   Use the default style as given by the font.
   
.. data:: STYLE_BOLD

   Use the bold style of the font. If the font does not contain information for
   bold text and glyph rendering, it will be emulated.

.. data:: STYLE_ITALIC

   Use the italic style of the font. If the font does not contain information
   for italic text and glyph rendering, it will be emulated.

.. data:: STYLE_UNDERLINE

   Use an underlined style of the font. This will cause the glyphs and texts
   to be rendered with an additional line beneath the glyph baseline.

Bounding Box Constants
----------------------

Those constants are used or getting the glyph and text metrics of a specific
:class:`pygame2.freetype.Font` in the :meth:`pygame2.freetype.Font.get_metrics`
method.

.. data:: BBOX_EXACT

    Return accurate floating point values for each individual glyph.
    In contrast to the :data:`BBOX_EXACT_GRIDFIT` constant, this can return
    different minimum and maximum y extents for each glyph.

.. data:: BBOX_EXACT_GRIDFIT

    Return accurate floating point values aligned to the surrounding drawing
    grid for each glyph.

.. data:: BBOX_PIXEL
    
    Return pixel coordinates (integer values) for each individual glyph.
    In contrast to the :data:`BBOX_EXACT_GRIDFIT` constant, this can return
    different minimum and maximum y extents for each glyph.

.. data:: BBOX_PIXEL_GRIDFIT

    Return  pixel coordinates (integer values) aligned to the surrounding
    drawing grid for each glyph.
