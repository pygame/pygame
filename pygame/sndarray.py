#!/usr/bin/env python

'''Pygame module for accessing sound sample data.

Functions to convert between Numeric arrays and Sound objects. This
module will only be available when pygame can use the external Numeric package.

Sound data is made of thousands of samples per second, and each sample is the
amplitude of the wave at a particular moment in time. For example, in
22-kHz format, element number 5 of the array is the amplitude of the wave
after 5/22000 seconds.

Each sample is an 8-bit or 16-bit integer, depending on the data format.
A stereo sound file has two values per sample, while a mono sound file only has one.

By default a Numeric array will be returned.  You can request a numpy
or numarray array using `pygame.set_array_module`.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from SDL import *

import pygame.array
import pygame.base
import pygame.mixer

def _as_array(sound):
    ready, frequency, format, channels = Mix_QuerySpec()
    if not ready:
        raise pygame.base.error, 'Mixer not initialized'

    formatbytes = (format & 0xff) >> 3
    if channels == 1:
        shape = (chunk.alen / formatbytes,)
    else:
        shape = (chunk.alen / formatbytes / 2, 2)

    signed = format in (AUDIO_S8, AUDIO_S16LE, AUDIO_S16BE)

    return pygame.array._array_from_buffer(chunk.abuf, formatbytes,
                                           shape, signed)

def array(sound):
    '''Copy Sound samples into an array.

    Creates a new Numeric array for the sound data and copies the samples. The
    array will always be in the format returned from pygame.mixer.get_init(). 

    :Parameters:
        `sound` : `Sound`
            Sound data to copy.

    :rtype: Numeric, numpy or numarray array
    '''
    return _as_array(sound).copy()

def samples(sound):
    '''Reference Sound samples into an array.

    Creates a new Numeric array that directly references the samples in a 
    Sound object. Modifying the array will change the Sound. The array
    will always be in the format returned from pygame.mixer.get_init().

    :Parameters:
        `sound` : `Sound`
            Sound data to reference.

    :rtype: Numeric, numpy or numarray array
    '''
    # XXX not setting base pointer; up to caller to maintain ref to sound.
    return _as_array(sound)

def make_sound(array):
    '''Convert an array into a Sound object.

    Create a new playable Sound object from a Numeric array. The mixer module
    must be initialized and the array format must be similar to the mixer
    audio format.

    :Parameters:
        `array` : Numeric, numpy or numarray array
            Array to copy.

    :rtype: `Sound`
    ''' 
    ready, frequency, format, channels = Mix_QuerySpec()
    if not ready:
        raise pygame.base.error, 'Mixer not initialized'

    mixerbytes = (format & 0xff) >> 3

    module = pygame.array._get_array_local_module(array)
    shape = module.shape(array)

    if len(shape) != channels or (len(shape) == 2 and shape[1] != channels):
        raise ValueError, \
             'Array must be have same number of dimensions as mixer channels.'

    data = array.tostring()
    chunk = Mix_Chunk()
    chunk.allocated = 1
    chunk.volume = 128
    chunk.alen = len(data)
    chunk._abuf = create_string_buffer(data)

    return pygame.mixer.Sound(None, _chunk=chunk)
            

