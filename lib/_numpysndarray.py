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

"""pygame module for accessing 
"""

import pygame
import pygame.mixer as mixer 
import numpy

def array (sound):
    """
    """
    # Info is a (freq, format, stereo) tuple
    info = mixer.get_init ()
    if not info:
        raise pygame.error, "Mixer not initialized"
    fmtbytes = (abs (info[1]) & 0xff) >> 3
    channels = mixer.get_num_channels ()
    data = sound.get_buffer ().raw

    shape = (len (data) / channels * fmtbytes, )
    if channels > 1:
        shape = (shape[0], 2)
        
    typecode = None
    # Signed or unsigned representation?
    if info[1] in (pygame.AUDIO_S8, pygame.AUDIO_S16LSB, pygame.AUDIO_S16MSB):
        typecode = (numpy.uint8, numpy.uint16, None, numpy.uint32)[fmtbytes - 1]
    else:
        typecode = (numpy.int8, numpy.int16, None, numpy.int32)[fmtbytes - 1]
        
    array = numpy.fromstring (data, typecode)
    array.shape = shape
    return array

def samples (sound):
    """
    """
    # Info is a (freq, format, stereo) tuple
    info = pygame.mixer.get_init ()
    if not info:
        raise pygame.error, "Mixer not initialized"
    fmtbytes = (abs (info[1]) & 0xff) >> 3
    channels = mixer.get_num_channels ()
    data = sound.get_buffer ()

    shape = (data.length / channels * fmtbytes, )
    if channels > 1:
        shape = (shape[0], 2)
        
    typecode = None
    # Signed or unsigned representation?
    if format in (pygame.AUDIO_S8, pygame.AUDIO_S16LSB, pygame.AUDIO_S16MSB):
        typecode = (numpy.uint8, numpy.uint16, None, numpy.uint32)[fmtbytes - 1]
    else:
        typecode = (numpy.int8, numpy.int16, None, numpy.int32)[fmtbytes - 1]
        
    array = numpy.frombuffer (data, typecode)
    array.shape = shape
    return array

def make_sound (array):
    """
    """
    # Info is a (freq, format, stereo) tuple
    info = pygame.mixer.get_init ()
    if not info:
        raise pygame.error, "Mixer not initialized"
    fmtbytes = (abs (info[1]) & 0xff) >> 3
    channels = mixer.get_num_channels ()

    shape = array.shape
    if len (shape) != channels:
        if channels == 1:
            raise ValueError, "Array must be 1-dimensional for mono mixer"
        elif channels == 2:
            raise ValueError, "Array must be 2-dimensional for stereo mixer"
        else:
            raise ValueError, "Array depth must match number of mixer channels"
    return mixer.Sound (array)
