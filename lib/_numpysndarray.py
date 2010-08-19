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
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
##    Library General Public License for more details.
##
##    You should have received a copy of the GNU Library General Public
##    License along with this library; if not, write to the Free
##    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
##
##    Marcus von Appen
##    mva@sysfault.org

"""pygame module for accessing sound sample data using numpy

Functions to convert between numpy arrays and Sound objects. This module
will only be available when pygame can use the external numpy package.

Sound data is made of thousands of samples per second, and each sample
is the amplitude of the wave at a particular moment in time. For
example, in 22-kHz format, element number 5 of the array is the
amplitude of the wave after 5/22000 seconds.

Each sample is an 8-bit or 16-bit integer, depending on the data format.
A stereo sound file has two values per sample, while a mono sound file
only has one.
"""

import pygame
import pygame.mixer as mixer 
import numpy

def _array_samples(sound, raw):
    # Info is a (freq, format, stereo) tuple
    info = mixer.get_init ()
    if not info:
        raise pygame.error("Mixer not initialized")
    fmtbytes = (abs (info[1]) & 0xff) >> 3
    channels = info[2]
    if raw:
        data = sound.get_buffer ().raw
    else:
        data = sound.get_buffer ()

    shape = (len (data) // fmtbytes, )
    if channels > 1:
        shape = (shape[0] // channels, channels)

    # mixer.init () does not support different formats from the ones below,
    # so MSB/LSB stuff is silently ignored.
    typecode = { 8 : numpy.uint8,   # AUDIO_U8
                 16 : numpy.uint16, # AUDIO_U16 / AUDIO_U16SYS
                 -8 : numpy.int8,   # AUDIO_S8
                 -16 : numpy.int16  # AUDUI_S16 / AUDIO_S16SYS
                 }[info[1]]
                 
    array = numpy.fromstring (data, typecode)
    array.shape = shape
    return array

def array (sound):
    """pygame._numpysndarray.array(Sound): return array

    Copy Sound samples into an array.

    Creates a new array for the sound data and copies the samples. The
    array will always be in the format returned from
    pygame.mixer.get_init().
    """
    return _array_samples(sound, True)

def samples (sound):
    """pygame._numpysndarray.samples(Sound): return array

    Reference Sound samples into an array.

    Creates a new array that directly references the samples in a Sound
    object. Modifying the array will change the Sound. The array will
    always be in the format returned from pygame.mixer.get_init().
    """
    # Info is a (freq, format, stereo) tuple
    info = pygame.mixer.get_init ()
    if not info:
        raise pygame.error("Mixer not initialized")
    fmtbytes = (abs (info[1]) & 0xff) >> 3
    channels = info[2]
    data = sound.get_buffer ()

    shape = (data.length // fmtbytes, )
    if channels > 1:
        shape = (shape[0] // channels, channels)
        
    # mixer.init () does not support different formats from the ones below,
    # so MSB/LSB stuff is silently ignored.
    typecode = { 8 : numpy.uint8,   # AUDIO_U8
                 16 : numpy.uint16, # AUDIO_U16
                 -8 : numpy.int8,   # AUDIO_S8
                 -16 : numpy.int16  # AUDUI_S16
                 }[info[1]]

    array = numpy.frombuffer (data, typecode)
    array.shape = shape
    return array

def make_sound (array):
    """pygame._numpysndarray.make_sound(array): return Sound

    Convert an array into a Sound object.
    
    Create a new playable Sound object from an array. The mixer module
    must be initialized and the array format must be similar to the mixer
    audio format.
    """
    # Info is a (freq, format, stereo) tuple
    info = pygame.mixer.get_init ()
    if not info:
        raise pygame.error("Mixer not initialized")
    channels = info[2]

    shape = array.shape
    if channels == 1:
        if len (shape) != 1:
            raise ValueError("Array must be 1-dimensional for mono mixer")
    else:
        if len (shape) != 2:
            raise ValueError("Array must be 2-dimensional for stereo mixer")
        elif shape[1] != channels:
            raise ValueError("Array depth must match number of mixer channels")
    return mixer.Sound (buffer=array)
