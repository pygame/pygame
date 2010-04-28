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

Format Constants
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

Fading Status Constants
-----------------------

Those constants indicate the fading status for :class:`pygame2.sdlmixer.Channel`
and :class:`pygame2.sdlmixer.Music` objects.

.. data:: NO_FADING

   Indicates that the Channel or Music is currently not faded.

.. data:: FADING_IN

   Indicates that the Channel or Music is currently fading in.

.. data:: FADING_OUT

   Indicates that the Channel or Music is currently fading out.

Music Type Constants
--------------------

Those constants are used by the :attr:`pygame2.sdlmixer.Music.type` attribute.

.. data:: MUS_NONE

   Indicates no music being played at all.

.. data:: MUS_CMD

   Indicates an external command to be used for music playback.

.. data:: MUS_WAV

   The music format is WAV data.

.. data:: MUS_MOD

   The music format is MOD data.
   
.. data:: MUS_MID

   The music format is MIDI data.

.. data:: MUS_OGG

   The music format is Ogg/Vorbis encoded data.

.. data:: MUS_MP3

   The music format is MP3 encoded data (using smpeg as decoder).

.. data:: MUS_MP3_MAD

   The music format is MP3 encoded data (using libmad as decoder).

Miscellaneous Constants
-----------------------

.. data:: CHANNELS

   The default amount of ready-to-use allocated
   :class:`pygame2.sdlmixer.Channel` objects after the initial call to
   :func:`pygame2.sdlmixer.open_audio`.

.. data:: DEFAULT_FREQUENCY

   A good default sample rate in Hz for most sound cards (22050).

.. data:: DEFAULT_FORMAT

   The suggested default audio format (:const:`AUDIO_S16SYS`).
   
.. data:: DEFAULT_CHANNELS

   The suggested default channel setting for :func:`pygame2.sdlmixer.open_audio`.
   This is 2 for stereo sound.

.. data:: MAX_VOLUME

   The maximum value for any volume setting.
