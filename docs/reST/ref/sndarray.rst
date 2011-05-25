.. include:: common.txt

:mod:`pygame.sndarray`
======================

.. module:: pygame.sndarray
   :synopsis: pygame module for accessing sound sample data

| :sl:`pygame module for accessing sound sample data`

Functions to convert between Numeric or numpy arrays and Sound objects. This
module will only be available when pygame can use the external numpy or Numeric
package.

Sound data is made of thousands of samples per second, and each sample is the
amplitude of the wave at a particular moment in time. For example, in 22-kHz
format, element number 5 of the array is the amplitude of the wave after
5/22000 seconds.

Each sample is an 8-bit or 16-bit integer, depending on the data format. A
stereo sound file has two values per sample, while a mono sound file only has
one.

Supported array systems are

::

  numpy
  numeric (deprecated; to be removed in Pygame 1.9.3.)

The default will be numpy, if installed. Otherwise, Numeric will be set as
default if installed, and a deprecation warning will be issued. If neither
numpy nor Numeric are installed, the module will raise an ImportError.

The array type to use can be changed at runtime using the ``use_arraytype()``
method, which requires one of the above types as string.

Note: numpy and Numeric are not completely compatible. Certain array
manipulations, which work for one type, might behave differently or even
completely break for the other.

Additionally, in contrast to Numeric numpy can use unsigned 16-bit integers.
Sounds with 16-bit data will be treated as unsigned integers, if the sound
sample type requests this. Numeric instead always uses signed integers for the
representation, which is important to keep in mind, if you use the module's
functions and wonder about the values.

numpy support added in Pygame 1.8 Official Numeric deprecation begins in Pygame
1.9.2.

.. function:: array

   | :sl:`copy Sound samples into an array`
   | :sg:`array(Sound) -> array`

   Creates a new array for the sound data and copies the samples. The array
   will always be in the format returned from ``pygame.mixer.get_init()``.

   .. ## pygame.sndarray.array ##

.. function:: samples

   | :sl:`reference Sound samples into an array`
   | :sg:`samples(Sound) -> array`

   Creates a new array that directly references the samples in a Sound object.
   Modifying the array will change the Sound. The array will always be in the
   format returned from ``pygame.mixer.get_init()``.

   .. ## pygame.sndarray.samples ##

.. function:: make_sound

   | :sl:`convert an array into a Sound object`
   | :sg:`make_sound(array) -> Sound`

   Create a new playable Sound object from an array. The mixer module must be
   initialized and the array format must be similar to the mixer audio format.

   .. ## pygame.sndarray.make_sound ##

.. function:: use_arraytype

   | :sl:`Sets the array system to be used for sound arrays`
   | :sg:`use_arraytype (arraytype) -> None`

   Uses the requested array type for the module functions. Currently supported
   array types are:

   ::

     numeric (deprecated; will be removed in Pygame 1.9.3.)
     numpy

   If the requested type is not available, a ValueError will be raised.

   New in pygame 1.8.

   .. ## pygame.sndarray.use_arraytype ##

.. function:: get_arraytype

   | :sl:`Gets the currently active array type.`
   | :sg:`get_arraytype () -> str`

   Returns the currently active array type. This will be a value of the
   ``get_arraytypes()`` tuple and indicates which type of array module is used
   for the array creation.

   New in pygame 1.8

   .. ## pygame.sndarray.get_arraytype ##

.. function:: get_arraytypes

   | :sl:`Gets the array system types currently supported.`
   | :sg:`get_arraytypes () -> tuple`

   Checks, which array systems are available and returns them as a tuple of
   strings. The values of the tuple can be used directly in the
   :func:`pygame.sndarray.use_arraytype` () method. If no supported array
   system could be found, None will be returned.

   New in pygame 1.8.

   .. ## pygame.sndarray.get_arraytypes ##

.. ## pygame.sndarray ##
