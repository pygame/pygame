:mod:`pygame2.openal.constants` -- Constants for OpenAL
=======================================================

This module contains the constants used throughout the
:mod:`pygame2.openal` module.

.. module:: pygame2.openal.constants
   :synopsis: Constants used throughout the :mod:`pygame2.openal` module.

OpenAL constants separate into two different types, general constants, which
are prefixed with *AL_* and context/device specific constants, which are
prefixed with *ALC_*.

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

     * :data:`AL_UNDETERMINED`
     * :data:`AL_STATIC`
     * :data:`AL_STREAMING`

.. data:: AL_LOOPING

   Turns looping on (AL_TRUE) or off (AL_FALSE).

.. data:: AL_BUFFER

   The ID of the attached buffer.

.. data:: AL_SOURCE_STATE

   The state of the source. This will be a value of

     * :data:`AL_STOPPED`
     * :data:`AL_PLAYING`
     * :data:`AL_PAUSED`

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

Source Type Constants
^^^^^^^^^^^^^^^^^^^^^

The following constants indicate the type of the source.

.. data:: AL_UNDETERMINED

   The type could not be determined.

.. data:: AL_STATIC

   It is a static source (e.g. a fixed buffer).

.. data:: AL_STREAMING

   It is a streaming source.
   
Source State Constants
^^^^^^^^^^^^^^^^^^^^^^

The following constants indicate the state of the source.

.. data:: AL_STOPPED

   The source playback is stopped.

.. data:: AL_PLAYING

   The source is currently playing.

.. data:: AL_PAUSED

   The source playback is paused.

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

.. data:: AL_DISTANCE_MODEL

   Constant for querying or setting the current distance model. This is
   implicitly done in :attr:`pygame2.openal.Context.distance_model`.

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

Doppler Shift Constants
-----------------------

The following constants are used for doppler effects using
:attr:`pygame2.openal.Context.doppler_factor` and
:attr:`pygame2.openal.Context.speed_of_sound`. They are used for compatibility
reasons and interoperability with other code, but usually not needed within
the :mod:`pygame2.openal` module.

.. data:: AL_DOPPLER_FACTOR

   Used to receive the currently set doppler factor value.

.. data:: AL_SPEED_OF_SOUND

    Used to receive the currently set speed of sound value.

Context and Device Constants
----------------------------

The following constants are used in conjunction with OpenAL Contexts and
Devices.

.. data:: ALC_FREQUENCY

   The :class:`pygame2.openal.Context` output frequency. This can be set by
   passing a matching value tuple to the :class:`pygame2.openal.Context`
   constructor.

.. data:: ALC_MONO_SOURCES

   The requested number of mono source for the :class:`pygame2.openal.Context`.
   This can be set by passing a matching value tuple to the
   :class:`pygame2.openal.Context` constructor.

.. data:: ALC_STEREO_SOURCES

   The requested number of stereo source for the :class:`pygame2.openal.Context`.
   This can be set by passing a matching value tuple to the
   :class:`pygame2.openal.Context` constructor.

.. data:: ALC_REFRESH

   The update rate of the context processing. This can be set by passing a
   matching value tuple to the :class:`pygame2.openal.Context` constructor.

.. data:: ALC_SYNC

   Indicates a synchronous context. This can be set by passing a matching value
   tuple to the :class:`pygame2.openal.Context` constructor.

Error Constants
---------------

The following constants indicate OpenAL errors. They are usually mapped against
:exc:`pygame2.Error` instances with a matching description. They are only used
for compatibility reasons and interoperability with other code, but usually not
needed within the :mod:`pygame2.openal` module.

.. data:: AL_NO_ERROR

   No error.

.. data:: AL_INVALID_NAME

   A bad name (ID) was passed to an OpenAL function.

.. data:: AL_INVALID_ENUM

   An invalid enum value was passed to an OpenAL function.

.. data:: AL_INVALID_VALUE

   An invalid value was passed to an OpenAL function.

.. data:: AL_INVALID_OPERATION

   The requested operation is not valid.

.. data:: AL_OUT_OF_MEMORY

   The requested operation resulted in OpenAL running out of memory.

.. data:: ALC_NO_ERROR

   No error.

.. data:: ALC_INVALID_DEVICE

   A bad device was passed to an OpenAL function.

.. data:: ALC_INVALID_CONTEXT

   A bad context was passed to an OpenAL function.

.. data:: ALC_INVALID_ENUM

   An unknown enum value was passed to an OpenAL function.

.. data:: ALC_INVALID_VALUE

   An invalid value was passed to an OpenAL function.

.. data:: ALC_OUT_OF_MEMORY

   The requested operation resulted in OpenAL running out of memory.

Miscellaneous Constants
-----------------------

.. data:: AL_TRUE

   OpenAL boolean representing 'True'.
   
.. data:: AL_FALSE
   
   OpenAL boolean representing 'False'.
   
.. data:: ALC_TRUE

   OpenAL Context boolean representing 'True'.

.. data:: ALC_FALSE
   
   OpenAL Context boolean representing 'False'.

.. data:: ALC_MAJOR_VERSION

    Context constant for querying the context/device major version
    (currently not supported in :mod:`pygame2.openal`).
   
.. data:: ALC_MINOR_VERSION

   Context constant for querying the context/device minor version
   (currently not supported in :mod:`pygame2.openal`).

.. data:: ALC_ATTRIBUTES_SIZE

   Context constant for querying the default attribute size
   (currently not supported in :mod:`pygame2.openal`).

.. data:: ALC_ALL_ATTRIBUTES

   Context constant for querying all attributes
   (currently not supported in :mod:`pygame2.openal`).

.. data:: ALC_CAPTURE_SAMPLES

   Context constant for querying a :class:`pygame2.openal.CaptureDevice` for
   available samples to receive. This is implicitly done in
   :meth:`pygame2.openal.CaptureDevice.get_samples`.

.. data:: ALC_DEFAULT_DEVICE_SPECIFIER

   Constant for querying the default output device name, OpenAL detected.
   This is implicitly done in
   :func:`pygame2.openal.get_default_ouput_device_name`.
   
.. data:: ALC_CAPTURE_DEFAULT_DEVICE_SPECIFIER

   Constant for querying the default capture device name, OpenAL detected.
   This is implicitly done in
   :func:`pygame2.openal.get_default_capture_device_name`.

.. data:: ALC_DEVICE_SPECIFIER

   Constant for querying the available output devices, OpenAL detected.
   This is is implicitly done in :func:`pygame2.openal.list_output_devices`.

.. data:: ALC_ALL_DEVICES_SPECIFIER
   
   Constant for querying the available output devices, OpenAL detected.
   This is is implicitly done in :func:`pygame2.openal.list_output_devices`.

.. data:: ALC_CAPTURE_DEVICE_SPECIFIER

   Constant for querying the available capture devices, OpenAL detected.
   This is is implicitly done in :func:`pygame2.openal.list_capture_devices`.

.. data:: ALC_EXTENSIONS

   Constant for querying the available context extensions, OpenAL detected
   (currently not supported in :mod:`pygame2.openal`)

.. data:: AL_VENDOR

   Queries the OpenAL vendor (use with :func:`pygame2.openal.al_get_string`).

.. data:: AL_VERSION

   Queries the OpenAL version (use with :func:`pygame2.openal.al_get_string`).
  
.. data:: AL_RENDERER

   Queries the OpenAL renderer (use with :func:`pygame2.openal.al_get_string`).

.. data:: AL_EXTENSIONS

   Queries the OpenAL extensions (use with :func:`pygame2.openal.al_get_string`).
   
TODO
----

.. data:: ALC_CHAN_CD_LOKI
.. data:: ALC_CHAN_MAIN_LOKI
.. data:: ALC_CHAN_PCM_LOKI
.. data:: ALC_CONNECTED
.. data:: ALC_DEFAULT_ALL_DEVICES_SPECIFIER
.. data:: AL_BYTE_RW_OFFSETS_EXT
.. data:: AL_DOPPLER_VELOCITY
.. data:: AL_DYNAMIC_COPY_EXT
.. data:: AL_DYNAMIC_READ_EXT
.. data:: AL_DYNAMIC_WRITE_EXT
.. data:: AL_EXTENSIONS
.. data:: AL_FORMAT_IMA_ADPCM_MONO16_EXT
.. data:: AL_FORMAT_IMA_ADPCM_STEREO16_EXT
.. data:: AL_FORMAT_MONO_DOUBLE_EXT
.. data:: AL_FORMAT_MONO_IMA4
.. data:: AL_FORMAT_REAR8
.. data:: AL_FORMAT_REAR16
.. data:: AL_FORMAT_REAR32
.. data:: AL_FORMAT_STEREO_DOUBLE_EXT
.. data:: AL_FORMAT_STEREO_IMA4
.. data:: AL_FORMAT_VORBIS_EXT
.. data:: AL_FORMAT_WAVE_EXT
.. data:: AL_ILLEGAL_COMMAND
.. data:: AL_ILLEGAL_ENUM
.. data:: AL_PENDING
.. data:: AL_PROCESSED
.. data:: AL_READ_ONLY_EXT
.. data:: AL_READ_WRITE_EXT
.. data:: AL_WRITE_ONLY_EXT
.. data:: AL_SAMPLE_RW_OFFSETS_EXT
.. data:: AL_SAMPLE_SINK_EXT
.. data:: AL_SAMPLE_SOURCE_EXT
.. data:: AL_STATIC_COPY_EXT
.. data:: AL_STATIC_READ_EXT
.. data:: AL_STATIC_WRITE_EXT
.. data:: AL_UNUSED 
