##    pygame - Python Game Library
##    Copyright (C) 2008 Marcus von Appen
##
##    This library is free software; you can redistribute it and/or
##    modify it under the terms of the GNU Library General Public
##    License as published by the Free Software Foundation; either
##    version 2 of the License, or (at your option) any later version.
##
##    This library is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##    Library General Public License for more details.
##
##    You should have received a copy of the GNU Library General Public
##    License along with this library; if not, write to the Free
##    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##

"""pygame module for accessing sound sample data

Functions to convert between Numeric or numpy arrays and Sound
objects. This module will only be available when pygame can use the
external numpy or Numeric package.

Sound data is made of thousands of samples per second, and each sample
is the amplitude of the wave at a particular moment in time. For
example, in 22-kHz format, element number 5 of the array is the
amplitude of the wave after 5/22000 seconds.

Each sample is an 8-bit or 16-bit integer, depending on the data format.
A stereo sound file has two values per sample, while a mono sound file
only has one.

Supported array systems are

  numeric
  numpy

The default will be Numeric, if installed. Otherwise, numpy will be set
as default if installed. If neither Numeric nor numpy are installed, the
module will raise an ImportError.

The array type to use can be changed at runtime using the use_arraytype()
method, which requires one of the above types as string.

Note: numpy and Numeric are not completely compatible. Certain array
manipulations, which work for one type, might behave differently or even
completely break for the other.

Additionally, in contrast to Numeric numpy can use unsigned 16-bit
integers. Sounds with 16-bit data will be treated as unsigned integers,
if the sound sample type requests this. Numeric instead always uses
signed integers for the representation, which is important to keep in
mind, if you use the module's functions and wonder about the values.
"""

import pygame2.compat
pygame2.compat.deprecation ("""The sndarray package is deprecated and
will be changed or removed in future versions""")

# Global array type setting. See use_arraytype().
__arraytype = None

# Try to import the necessary modules.
try:
    import pygame2.sdlmixer.numericsndarray as numericsnd
    __hasnumeric = True
    __arraytype = "numeric"
except ImportError:
    __hasnumeric = False

try:
    import pygame2.sdlmixer.numpysndarray as numpysnd
    __hasnumpy = True
    if not __hasnumeric:
        __arraytype = "numpy"
except ImportError:
    __hasnumpy = False

#if not __hasnumpy and not __hasnumeric:
#    raise ImportError ("no module named numpy or Numeric found")

def make_array (sound):
    """pygame2.sndarray.make_array(Sound): return array

    Copy Sound samples into an array.

    Creates a new array for the sound data and copies the samples. The
    array will always be in the format returned from
    pygame2.sdlmixer.get_init().
    """
    if __arraytype == "numeric":
        return numericsnd.make_array (sound)
    elif __arraytype == "numpy":
        return numpysnd.make_array (sound)
    raise NotImplementedError ("sound arrays are not supported")

def samples (sound):
    """pygame2.sndarray.samples(Sound): return array

    Reference Sound samples into an array.

    Creates a new array that directly references the samples in a Sound
    object. Modifying the array will change the Sound. The array will
    always be in the format returned from pygame2.mixer.get_init().
    """
    if __arraytype == "numeric":
        return numericsnd.samples (sound)
    elif __arraytype == "numpy":
        return numpysnd.samples (sound)
    raise NotImplementedError ("sound arrays are not supported")

def make_sound (array):
    """pygame2.sndarray.make_sound(array): return Sound

    Convert an array into a Sound object.
    
    Create a new playable Sound object from an array. The mixer module
    must be initialized and the array format must be similar to the mixer
    audio format.
    """
    if __arraytype == "numeric":
        return numericsnd.make_sound (array)
    elif __arraytype == "numpy":
        return numpysnd.make_sound (array)
    raise NotImplementedError ("sound arrays are not supported")

def use_arraytype (arraytype):
    """pygame2.sndarray.use_arraytype (arraytype): return None

    Sets the array system to be used for sound arrays.

    Uses the requested array type for the module functions.
    Currently supported array types are:

      numeric 
      numpy

    If the requested type is not available, a ValueError will be raised.
    """
    global __arraytype

    arraytype = arraytype.lower ()
    if arraytype == "numeric":
        if __hasnumeric:
            __arraytype = arraytype
        else:
            raise ValueError ("Numeric arrays are not available")
        
    elif arraytype == "numpy":
        if __hasnumpy:
            __arraytype = arraytype
        else:
            raise ValueError ("numpy arrays are not available")
    else:
        raise ValueError ("invalid array type")

def get_arraytype ():
    """pygame2.sndarray.get_arraytype (): return str

    Gets the currently active array type.

    Returns the currently active array type. This will be a value of the
    get_arraytypes() tuple and indicates which type of array module is
    used for the array creation.
    """
    return __arraytype

def get_arraytypes ():
    """pygame2.sndarray.get_arraytypes (): return tuple

    Gets the array system types currently supported.

    Checks, which array system types are available and returns them as a
    tuple of strings. The values of the tuple can be used directly in
    the use_arraytype () method.

    If no supported array system could be found, None will be returned.
    """
    vals = []
    if __hasnumeric:
        vals.append ("numeric")
    if __hasnumpy:
        vals.append ("numpy")
    if len (vals) == 0:
        return None
    return tuple (vals)
