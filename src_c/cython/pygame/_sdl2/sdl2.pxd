# cython: language_level=2
#

from libc.string cimport memset
from libc.stdio cimport *

cdef extern from "SDL.h" nogil:
    # SDL_stdinc.h provides the real ones based on platform.
    ctypedef char Sint8
    ctypedef unsigned char Uint8
    ctypedef signed short Sint16
    ctypedef unsigned short Uint16
    ctypedef signed long Sint32
    ctypedef unsigned long Uint32
    ctypedef unsigned long long Uint64
    ctypedef signed long long Sint64
    ctypedef int SDL_bool

    const char *SDL_GetError()

    # https://wiki.libsdl.org/SDL_InitSubSystem
    # https://wiki.libsdl.org/SDL_QuitSubSystem
    # https://wiki.libsdl.org/SDL_WasInit
    int SDL_InitSubSystem(Uint32 flags)
    void SDL_QuitSubSystem(Uint32 flags)
    Uint32 SDL_WasInit(Uint32 flags)

    cdef int _SDL_INIT_TIMER "SDL_INIT_TIMER"
    cdef int _SDL_INIT_AUDIO "SDL_INIT_AUDIO"
    cdef int _SDL_INIT_VIDEO "SDL_INIT_VIDEO"
    cdef int _SDL_INIT_JOYSTICK "SDL_INIT_JOYSTICK"
    cdef int _SDL_INIT_HAPTIC "SDL_INIT_HAPTIC"
    cdef int _SDL_INIT_GAMECONTROLLER "SDL_INIT_GAMECONTROLLER"
    cdef int _SDL_INIT_EVENTS "SDL_INIT_EVENTS"
    cdef int _SDL_INIT_SENSOR "SDL_INIT_SENSOR"
    cdef int _SDL_INIT_NOPARACHUTE "SDL_INIT_NOPARACHUTE"
    cdef int _SDL_INIT_EVERYTHING "SDL_INIT_EVERYTHING"
