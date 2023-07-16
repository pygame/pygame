# cython: language_level=3str
#

from .sdl2 cimport *

#https://www.libsdl.org/projects/SDL_mixer/docs/SDL_mixer.html#SEC79

ctypedef void (*mixcallback)(void *udata, Uint8 *stream, int len) noexcept nogil

cdef extern from "SDL_mixer.h" nogil:
    ctypedef void (*mix_func)(void *udata, Uint8 *stream, int len)
    void Mix_SetPostMix(void (*mixcallback)(void *udata, Uint8 *stream, int len), void *arg)


cdef class _PostMix:
    cdef mixcallback callback
    cdef void *userdata
    cdef object _callback
