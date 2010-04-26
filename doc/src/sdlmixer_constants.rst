:mod:`pygame2.sdlmixer.constants` -- Constants for SDL_mixer
============================================================

This module contains the constants used throughout the
:mod:`pygame2.sdlmixer` modules.

.. module:: pygame2.sdlmixer.constants
   :synopsis: Constants used throughout the :mod:`pygame2.sdlmixer` modules.

Initialisation Constants
------------------------

Those constants are used by the :func:`pygame2.sdlmixer.init` function.

.. data:: INIT_FLAC

   Initialises the FLAC library bindings.

.. data:: INIT_MOD

   Initialises the timidity library bindings.

.. data:: INIT_MP3

   Initialises the smpeg/mad library bindings.
    
.. data:: INIT_OGG

   Initialises the ogg/vorbis library bindings.

Format constants
----------------

Those constants are used by the :func:`pygame2.sdlmixer.open_audio` and
:func:`pygame2.sdlmixer.query_spec` functions.

.. data:: AUDIO_U8

   Unsigned 8-bit data in system byte order.

.. data:: AUDIO_S8

   Signed 8-bit data in system byte order.

.. data:: AUDIO_U16LSB

   Unsigned 16-bit data in little-endian byte order.
   
.. data:: AUDIO_S16LSB

   Signed 16-bit data in little-endian byte order.
    
.. data:: AUDIO_U16MSB

   Unsigned 16-bit data in big-endian byte order.

.. data:: AUDIO_S16MSB
   
   Signed 16-bit data in big-endian byte order.

.. data:: AUDIO_U16

   Unsigned 16-bit data in system byte order.

.. data:: AUDIO_S16
   
   Signed 16-bit data in system byte order.

.. data:: AUDIO_U16SYS

   Unsigned 16-bit data in system byte order.

.. data:: AUDIO_S16SYS
   
   Signed 16-bit data in system byte order.

.. todo::

   Complete the constants

