#!/usr/bin/env python

'''Access to the raw audio mixing buffer.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.array
import SDL.constants
import SDL.dll

_SDL_AudioSpec_fn = \
    CFUNCTYPE(POINTER(c_ubyte), POINTER(c_ubyte), POINTER(c_ubyte), c_int)

class SDL_AudioSpec(Structure):
    _fields_ = [('freq', c_int),
                ('format', c_ushort),
                ('channels', c_ubyte),
                ('silence', c_ubyte),
                ('samples', c_ushort),
                ('padding', c_ushort),
                ('size', c_uint),
                ('_callback', _SDL_AudioSpec_fn),
                ('_userdata', c_void_p)]

# TODO Fix AUDIO_U16SYS and AUDIO_S16SYS in constants.py

_SDL_AudioCVT_p = POINTER('SDL_AudioCVT')

_SDL_AudioCVT_filter_fn = \
    CFUNCTYPE(POINTER(c_ubyte), _SDL_AudioCVT_p, c_ushort)

class SDL_AudioCVT(Structure):
    _fields_ = [('needed', c_int),
                ('src_format', c_ushort),
                ('dst_format', c_ushort),
                ('rate_incr', c_double),
                ('buf', POINTER(c_ubyte)),
                ('len', c_int),
                ('len_cvt', c_int),
                ('len_mult', c_int),
                ('len_ratio', c_double),
                ('filters', _SDL_AudioCVT_filter_fn * 10),
                ('filter_index', c_int)]

SetPointerType(_SDL_AudioCVT_p, SDL_AudioCVT)

# SDL_AudioInit and SDL_AudioQuit marked private

_SDL_AudioDriverName = SDL.dll.private_function('SDL_AudioDriverName',
    arg_types=[c_char_p, c_int],
    return_type=c_char_p)

def SDL_AudioDriverName(maxlen=1024):
    '''
    Returns the name of the audio driver.  Returns None if no driver has
    been initialised.

    :Parameters:
        `maxlen`
            Maximum length of the returned driver name; defaults to 1024.

    :rtype: string
    '''
    buf = create_string_buffer(maxlen)
    if _SDL_AudioDriverName(buf, maxlen):
        return buf.value
    return None

def _ctype_audio_format(fmt):
    if fmt == SDL.constants.AUDIO_U8:
        return c_ubyte
    elif fmt == SDL.constants.AUDIO_S8:
        return c_char
    elif fmt in (SDL.constants.AUDIO_U16LSB, SDL.constants.AUDIO_U16MSB):
        return c_ushort
    elif fmt in (SDL.constants.AUDIO_S16LSB, SDL.constants.AUDIO_S16MSB):
        return c_short
    else:
        raise TypeError, 'Unsupported format %r' % fmt

_SDL_OpenAudio = SDL.dll.private_function('SDL_OpenAudio',
    arg_types=[POINTER(SDL_AudioSpec), POINTER(SDL_AudioSpec)],
    return_type=c_int,
    error_return=-1)

def SDL_OpenAudio(desired, obtained):
    '''Open the audio device with the desired parameters.

    If successful, the actual hardware parameters will be set in the
    instance passed into `obtained`.  If `obtained` is None, the audio
    data passed to the callback function will be guaranteed to be in
    the requested format, and will be automatically converted to the
    hardware audio format if necessary.

    An exception will be raised if the audio device couldn't be opened,
    or the audio thread could not be set up.

    The fields of `desired` are interpreted as follows:

        `desired.freq`
            desired audio frequency in samples per second
        `desired.format`
            desired audio format, i.e., one of AUDIO_U8, AUDIO_S8,
            AUDIO_U16LSB, AUDIO_S16LSB, AUDIO_U16MSB or AUDIO_S16MSB
        `desired.samples`
            size of the audio buffer, in samples.  This number should
            be a power of two, and may be adjusted by the audio driver
            to a value more suitable for the hardware.  Good values seem
            to range between 512 and 8096 inclusive, depending on the
            application and CPU speed.  Smaller values yield faster response
            time, but can lead to underflow if the application is doing
            heavy processing and cannot fill the audio buffer in time.
            A stereo sample consists of both right and left channels in
            LR ordering.  Note that the number of samples is directly
            related to time by the following formula::

                ms = (samples * 1000) / freq

        `desired.size`
            size in bytes of the audio buffer; calculated by SDL_OpenAudio.
        `desired.silence`
            value used to set the buffer to silence; calculated by
            SDL_OpenAudio.
        `desired.callback`
            a function that will be called when the audio device is ready
            for more data.  The signature of the function should be::

                callback(userdata: any, stream: SDL_array) -> None

            The function is called with the userdata you specify (see below),
            and an SDL_array of the obtained format which you must fill
            with audio data.

            This function usually runs in a separate thread, so you should
            protect data structures that it accesses by calling
            `SDL_LockAudio` and `SDL_UnlockAudio` in your code.
        `desired.userdata`
            passed as the first parameter to your callback function.
    
    The audio device starts out playing silence when it's opened, and should
    be enabled for playing by calling ``SDL_PauseAudio(False)`` when you are
    ready for your audio callback function to be called.  Since the audio
    driver may modify the requested size of the audio buffer, you should
    allocate any local mixing buffers after you open the audio device.

    :Parameters:
     - `desired`: `SDL_AudioSpec`
     - `obtained`: `SDL_AudioSpec` or None

    '''
    if not hasattr(desired, 'callback'):
        raise TypeError, 'Attribute "callback" not set on "desired"'
    userdata = getattr(desired, 'userdata', None)
    callback = desired.callback
    ctype = [_ctype_audio_format(desired.format)]  # List, so mutable

    def cb(data, stream, len):
        ar = SDL.array.SDL_array(stream, len, ctype[0])
        callback(userdata, ar)

    desired._callback = _SDL_AudioSpec_fn(cb)
    _SDL_OpenAudio(desired, obtained)
    if obtained:
        obtained.userdata = desired.userdata
        obtained.callback = desired.callback
        ctype[0] = _ctype_audio_format(obtained.format)

# enum SDL_audiostatus
(SDL_AUDIO_STOPPED,
    SDL_AUDIO_PLAYING,
    SDL_AUDIO_PAUSED) = range(3)

SDL_GetAudioStatus = SDL.dll.function('SDL_GetAudioStatus',
    '''Get the current audio state.

    :rtype: int
    :return: one of SDL_AUDIO_STOPPED, SDL_AUDIO_PLAYING, SDL_AUDIO_PAUSED
    ''',
    args=[],
    arg_types=[],
    return_type=c_int)

SDL_PauseAudio = SDL.dll.function('SDL_PauseAudio',
    '''Pause and unpause the audio callback processing.

    It should be called with a parameter of 0 after opening the audio
    device to start playing sound.  This is so you can safely initalize
    data for your callback function after opening the audio device.
    Silence will be written to the audio device during the pause.

    :Parameters:
     - `pause_on`: int

    ''',
    args=['pause_on'],
    arg_types=[c_int],
    return_type=None)

