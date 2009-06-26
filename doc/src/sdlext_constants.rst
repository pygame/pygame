:mod:`pygame2.sdlext.constants` -- Constants for the SDL extensions
===================================================================

This module contains the constants used throughout the :mod:`pygame2.sdlext`
modules.

.. module:: pygame2.sdlext.constants
   :synopsis: Constants used throughout the :mod:`pygame2.sdlext` modules.

Filter Constants
----------------

Those constants denote the available filter types for the
:func:`pygame2.sdlext.transform.smoothscale` function, which can be set
and get using :func:`pygame2.sdlext.transform.get_filtertype` and
:func:`pygame2.sdlext.transform.set_filtertype`.

.. data:: FILTER_C
   
   Use the plain C-language based filter. This is possibly the slowest of all
   filter types, but guaranteed to work on any platform.
   
.. data:: FILTER_MMX

   Use the MMX-based filter. This uses the Intel MMX instruction set to
   perform a fast filter operation. It is only available on Intel and AMD
   platforms.

.. data:: FILTER_SSE

   Use the SSE-based filter. This uses the Intel SSE instruction set to
   perform a fast filter operation. It is only available on newer Intel and AMD
   platforms (Pentium III and above).
