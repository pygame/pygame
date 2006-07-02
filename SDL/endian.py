#!/usr/bin/env python

'''Functions for converting to native byte order
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys

import SDL.constants

def SDL_Swap16(x):
    return (x << 8 & 0xff00) | \
           (x >> 8 & 0x00ff)

def SDL_Swap32(x):
    return (x << 24 & 0xff000000) | \
           (x << 8  & 0x00ff0000) | \
           (x >> 8  & 0x0000ff00) | \
           (x >> 24 & 0x000000ff) 

def SDL_Swap64(x):
    return (SDL_Swap32(x & 0xffffffff) << 32) | \
           (SDL_Swap32(x >> 32 & 0xffffffff))

def _noop(x):
    return x

if sys.byteorder == 'big':
    SDL_BYTEORDER = SDL.constants.SDL_BIG_ENDIAN
    SDL_SwapLE16 = SDL_Swap16
    SDL_SwapLE32 = SDL_Swap32
    SDL_SwapLE64 = SDL_Swap64
    SDL_SwapBE16 = _noop
    SDL_SwapBE32 = _noop
    SDL_SwapBE64 = _noop
else:
    SDL_BYTEORDER = SDL.constants.SDL_LIL_ENDIAN
    SDL_SwapLE16 = _noop
    SDL_SwapLE32 = _noop
    SDL_SwapLE64 = _noop
    SDL_SwapBE16 = SDL_Swap16
    SDL_SwapBE32 = SDL_Swap32
    SDL_SwapBE64 = SDL_Swap64

