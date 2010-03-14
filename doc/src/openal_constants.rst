:mod:`pygame2.openal.constants` -- Constants for OpenAL
=======================================================

This module contains the constants used throughout the
:mod:`pygame2.openal` module.

.. module:: pygame2.openal.constants
   :synopsis: Constants used throughout the :mod:`pygame2.openal` module.

Format Constants
----------------

Those constants are used by the :meth:`pygame2.openal.Buffers.buffer_data`,
and :meth:`pygame2.openal.CaptureDevice.get_samples` methods.

.. todo::

   Provide a detailled description about how the data will be streamed
   to the sound system.


.. data:: AL_FORMAT_MONO8

   Mono (1 channel) 8-bit data.

.. data:: AL_FORMAT_MONO16

   Mono (1 channel) 16-bit data.

.. data:: AL_FORMAT_MONO_FLOAT32

   Mono (1 channel) 32-bit floating point data.

.. data:: AL_FORMAT_STEREO8

   Stereo (2 channel) 8-bit data.

.. data:: AL_FORMAT_STEREO16

   Stereo (2 channel) 16-bit data.

.. data:: AL_FORMAT_STEREO_FLOAT32

   Stereo (2 channel) 32-bit floating point data.

.. data:: AL_FORMAT_QUAD8

   4 channel (4.0 Quad) 8-bit data.

.. data:: AL_FORMAT_QUAD8_LOKI

   4 channel (4.0 Quad) 8-bit data.

.. data:: AL_FORMAT_QUAD16

   4 channel (4.0 Quad) 16-bit data.

.. data:: AL_FORMAT_QUAD16_LOKI

   4 channel (4.0 Quad) 16-bit data.

.. data:: AL_FORMAT_QUAD32

   4 channel (4.0 Quad) 32-bit data.

.. data:: AL_FORMAT_51CHN8

   6 channel (5.1 Surround Sound) 8-bit data.

.. data:: AL_FORMAT_51CHN16

   6 channel (5.1 Surround Sound) 16-bit data.

.. data:: AL_FORMAT_51CHN32

   6 channel (5.1 Surround Sound) 32-bit data.

.. data:: AL_FORMAT_61CHN8

   7 channel (6.1 Surround Sound) 8-bit data.

.. data:: AL_FORMAT_61CHN16

   7 channel (6.1 Surround Sound) 16-bit data.

.. data:: AL_FORMAT_61CHN32

   7 channel (6.1 Surround Sound) 32-bit data.

.. data:: AL_FORMAT_71CHN8

   8 channel (7.1 Surround Sound) 8-bit data.

.. data:: AL_FORMAT_71CHN16

   8 channel (7.1 Surround Sound) 16-bit data.

.. data:: AL_FORMAT_71CHN32

   8 channel (7.1 Surround Sound) 32-bit data.

Buffer Constants
----------------

Those constants are used by the :meth:`pygame2.openal.Buffers.set_prop`
and :meth:`pygame2.openal.Buffers.get_prop` methods.

.. data:: AL_FREQUENCY

   The frequency of the buffer (and its data) in Hz. The provided value
   must be an integer.

.. data:: AL_BITS

   The bit depth of the buffer. The provided value must be an integer.

.. data:: AL_CHANNELS

   The number of channels for the buffered data. The provided value
   must be an integer greater than 0.

.. data:: AL_SIZE

   The size of the buffered data in bytes.

.. data:: AL_DATA

   The original location the buffered data came from. This is generally
   useless, as the original data location was probably freed after the
   data was buffered.

Source Constants
----------------

Those constants are used by the :meth:`pygame2.openal.Sources.set_prop`
and :meth:`pygame2.openal.Sources.get_prop` methods.

.. data:: AL_PITCH

   The pitch multiplier. The provided value must be a positive floating
   point value.

.. data:: AL_GAIN

   The source gain. The provided value should be a positive floating
   point value.

.. data:: AL_MAX_DISTANCE

   Used with the Inverse Clamped Distance Model to set the distance
   where there will no longer be any attenuation of the source. The
   provided value must be an integer or floating point value.

.. data:: AL_ROLLOFF_FACTOR

   The rolloff ratge for the source. The provided value should be an
   integer or floating point value.

.. data:: AL_REFERENCE_DISTANCE

   The distance under which the volume for the source would normally
   drop by half. The provided value should be an integer or floating
   point value.

.. data:: AL_MIN_GAIN

   The minimum source gain. The provided value must be a postive
   floating point value.

.. data:: AL_MAX_GAIN

   The maximum source gain. The provided value must be a postive
   floating point value.

.. data:: AL_CONE_OUTER_GAIN

   The gain when outside the oriented cone. The provided value must be a
   postive floating point value.

.. data:: AL_CONE_INNER_ANGLE

   The gain when inside the oriented cone. The provided value must be a
   postive floating point value.

.. data:: AL_CONE_OUTER_ANGLE

   The outer angle of the sound cone in degrees. The provided value must
   be a integer or floating point value.

.. data:: AL_POSITION

   The X, Y, Z position of the source. The provided value must be a
   triplet of floating point values.

.. data:: AL_VELOCITY

   The X, Y, Z velocity of the source. The provided value must be a
   triplet of floating point values.

.. data:: AL_DIRECTION

   The X, Y, Z direction of the source. The provided value must be a
   triplet of integer or floating point values.

.. data:: AL_SOURCE_RELATIVE

   Determines if the positions of the source are relative to the
   listener. The provided value must be an integer.

.. data:: AL_SOURCE_TYPE

   The type of the source. This will be a value of

     * AL_UNDETERMINED
     * AL_STATIC
     * AL_STREAMING

.. data:: AL_LOOPING

   Turns looping on (AL_TRUE) or off (AL_FALSE).

.. data:: AL_BUFFER

   The ID of the attached buffer

.. data:: AL_SOURCE_STATE

   The state of the source. This will be a value of

     * AL_STOPPED
     * AL_PLAYING
     * AL_PAUSED

.. data:: AL_BUFFERS_QUEUED

   The number of buffers queued on this source.

.. data:: AL_BUFFERS_PROCESSED

   The number of buffers in the queue that have been processed.

.. data:: AL_SEC_OFFSET

   The source playback position, in seconds.

.. data:: AL_SAMPLE_OFFSET

   The source playback position, in samples.

.. data:: AL_BYTE_OFFSET

   The source playback position, in bytes.

Listener Constants
------------------

Those constants are used by the :meth:`pygame2.openal.Listener.set_prop`
and :meth:`pygame2.openal.Listener.get_prop` methods.

.. data:: AL_GAIN

   The master gain. The provided value should be a positive floating
   point value.

.. data:: AL_POSITION

   The X, Y, Z position of the listener. The provided value must be a
   triplet of floating point values.

.. data:: AL_VELOCITY

   The X, Y, Z velocity of the listener. The provided value must be a
   triplet of floating point values.

.. data:: AL_ORIENTATION

   The orientation of the listener, expressed as "at" and "up" vectors
   (6 elements) of floating point values.

Distance Model Constants
------------------------

Those constance are used by the
:attr:`pygame2.openal.Context.distance_model` property.

.. data:: AL_INVERSE_DISTANCE

   TODO

.. data:: AL_INVERSE_DISTANCE_CLAMPED

   TODO

.. data:: AL_LINEAR_DISTANCE

   TODO

.. data:: AL_LINEAR_DISTANCE_CLAMPED

   TODO

.. data:: AL_EXPONENT_DISTANCE

   TODO

.. data:: AL_EXPONENT_DISTANCE_CLAMPED

   TODO

.. data:: AL_NONE

   TODO


.. todo::

   Complete the constant descriptions.
