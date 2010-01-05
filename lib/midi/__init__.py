"""
MIDI input and output interaction module.

The midi module can send output to midi devices, and get input from midi
devices. It can also list midi devices on the system, including real and
virtual ones.

It uses the portmidi library (using PyPortMidi) and is portable to all
platforms portmidi supports (currently windows, OSX, and linux).
"""
from pygame2.midi.base import *
