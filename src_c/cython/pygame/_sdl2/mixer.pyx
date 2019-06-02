from . import error
import traceback

#https://www.libsdl.org/projects/SDL_mixer/docs/SDL_mixer.html#SEC79

# void (*mix_func)(void *udata, Uint8 *stream, int len),
# // make a passthru processor function that does nothing...
# void noEffect(void *udata, Uint8 *stream, int len)
# {
#     // you could work with stream here...
# }
# ...
# // register noEffect as a postmix processor
# Mix_SetPostMix(noEffect, NULL);


cdef void recording_cb(void* userdata, Uint8* stream, int len) nogil:
    """ This is called in a thread made by SDL.
        So we need the python GIL to do python stuff.
    """
    cdef Uint8 [:] a_memoryview
    with gil:
        a_memoryview = <Uint8[:len]> stream
        try:
            # The userdata is the audio device.
            # The audio device is needed in some apps
            (<object>userdata).callback(<object>userdata, a_memoryview)
        except:
            traceback.print_exc()
            raise

# ctypedef void (*cfptr)(int)
# cdef cfptr myfunctionptr = &myfunc




cdef class _PostMix:
    # def __cinit__(self):

    def __init__(self, callback):
        self._callback = callback
        self.userdata = <void*>self
        self.callback = <mixcallback>recording_cb;
        Mix_SetPostMix(self.callback, self.userdata)

    def __dealloc__(self):
        Mix_SetPostMix(NULL, NULL)

    @property
    def callback(self):
        """ called in the sound thread with (audiodevice, memoryview)
        """
        return self._callback


_postmix = None
cpdef set_post_mix(mix_func):
    """ Hook a processor function mix_func to the postmix stream for
        post processing effects. You may just be reading the data and displaying
        it, or you may be altering the stream to add an echo.


    """
    global _postmix
    if mix_func is None:
        _postmix = None
        Mix_SetPostMix(NULL, NULL)
    else:
        _postmix = _PostMix(mix_func)
