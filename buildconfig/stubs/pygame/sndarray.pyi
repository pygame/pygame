"""
pygame module for accessing sound sample data

Functions to convert between NumPy arrays and Sound objects. This module
will only be functional when pygame can use the external NumPy package.
If NumPy can't be imported, surfarray becomes a MissingModule object.

Sound data is made of thousands of samples per second, and each sample
is the amplitude of the wave at a particular moment in time. For
example, in 22-kHz format, element number 5 of the array is the
amplitude of the wave after 5/22000 seconds.

Each sample is an 8-bit or 16-bit integer, depending on the data format.
A stereo sound file has two values per sample, while a mono sound file
only has one.

Sounds with 16-bit data will be treated as unsigned integers,
if the sound sample type requests this.
"""

from typing import Tuple

import numpy

from pygame.mixer import Sound

def array(sound: Sound) -> numpy.ndarray: ...
def samples(sound: Sound) -> numpy.ndarray: ...
def make_sound(array: numpy.ndarray) -> Sound: ...
def use_arraytype(arraytype: str) -> Sound: ...
def get_arraytype() -> str: ...
def get_arraytypes() -> Tuple[str]: ...
