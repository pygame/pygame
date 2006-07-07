#!/usr/bin/env python

'''Access to the raw audio mixing buffer.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *
import sys

import SDL.array
import SDL.constants
import SDL.dll
import SDL.rwops

_SDL_AudioSpec_fn = \
    CFUNCTYPE(POINTER(c_ubyte), POINTER(c_ubyte), POINTER(c_ubyte), c_int)

class SDL_AudioSpec(Structure):
    '''Audio format structure.

    The calculated values in this structure are calculated by `SDL_OpenAudio`.

    :Ivariables:
        `freq` : int
            DSP frequency, in samples per second
        `format` : int
            Audio data format. One of AUDIO_U8, AUDIO_S8, AUDIO_U16LSB,
            AUDIO_S16LSB, AUDIO_U16MSB or AUDIO_S16MSB
        `channels` : int
            Number of channels; 1 for mono or 2 for stereo.
        `silence` : int
            Audio buffer silence value (calculated)
        `samples` : int
            Audio buffer size in samples (power of 2)
        `size` : int
            Audio buffer size in bytes (calculated)

    '''
    _fields_ = [('freq', c_int),
                ('format', c_ushort),
                ('channels', c_ubyte),
                ('silence', c_ubyte),
                ('samples', c_ushort),
                ('_padding', c_ushort),
                ('size', c_uint),
                ('_callback', _SDL_AudioSpec_fn),
                ('_userdata', c_void_p)]

_SDL_AudioCVT_p = POINTER('SDL_AudioCVT')

_SDL_AudioCVT_filter_fn = \
    CFUNCTYPE(POINTER(c_ubyte), _SDL_AudioCVT_p, c_ushort)

class SDL_AudioCVT(Structure):
    '''Set of audio conversion filters and buffers.

    :Ivariables:
        `needed` : int
            1 if conversion is possible
        `src_format` : int
            Source audio format.  See `SDL_AudioSpec.format`
        `dst_format` : int
            Destination audio format.  See `SDL_AudioSpec.format`
        `rate_incr` : float
            Rate conversion increment
        `len` : int
            Length of original audio buffer
        `len_cvt` : int
            Length of converted audio buffer
        `len_mult` : int
            Buffer must be len * len_mult big
        `len_ratio` : float
            Given len, final size is len * len_ratio
        `filter_index` : int
            Current audio conversion function
        
    '''
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
        ar = SDL.array.SDL_array(stream, len/sizeof(ctype[0]), ctype[0])
        callback(userdata, ar)

    desired._callback = _SDL_AudioSpec_fn(cb)
    _SDL_OpenAudio(desired, obtained)
    if obtained:
        obtained.userdata = desired.userdata
        obtained.callback = desired.callback
        ctype[0] = _ctype_audio_format(obtained.format)

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

_SDL_LoadWAV_RW = SDL.dll.private_function('SDL_LoadWAV_RW',
    arg_types=[POINTER(SDL.rwops.SDL_RWops), 
               c_int, 
               POINTER(SDL_AudioSpec), 
               POINTER(POINTER(c_ubyte)),
               POINTER(c_uint)],
    return_type=POINTER(SDL_AudioSpec),
    require_return=True)

def SDL_LoadWAV_RW(src, freesrc):
    '''Load a WAVE from the data source.

    The source is automatically freed if `freesrc` is non-zero.  For
    example, to load a WAVE file, you could do::

        SDL_LoadWAV_RW(SDL_RWFromFile('sample.wav', 'rb'), 1)

    You need to free the returned buffer with `SDL_FreeWAV` when you
    are done with it.

    :Parameters:
     - `src`: `SDL_RWops`
     - `freesrc`: int

    :rtype: (`SDL_AudioSpec`, `SDL_array`)
    :return: a tuple (`spec`, `audio_buf`) where `spec` describes the data
        format and `audio_buf` is the buffer containing audio data.
    '''
    spec = SDL_AudioSpec()
    audio_buf = POINTER(c_ubyte)()
    audio_len = c_uint()
    _SDL_LoadWAV_RW(src, freesrc, spec, byref(audio_buf), byref(audio_len))
    ctype = _ctype_audio_format(spec.format)
    return (spec, 
            SDL.array.SDL_array(audio_buf, audio_len.value/sizeof(ctype), ctype))

def SDL_LoadWAV(file):
    '''Load a WAVE from a file.

    :Parameters:
     - `file`: str

    :rtype: (`SDL_AudioSpec`, `SDL_array`)
    :see: `SDL_LoadWAV_RW`
    '''
    return SDL_LoadWAV_RW(SDL.rwops.SDL_RWFromFile(file, 'rb'), 1)

_SDL_FreeWAV = SDL.dll.private_function('SDL_FreeWAV',
    arg_types=[POINTER(c_ubyte)],
    return_type=None)

def SDL_FreeWAV(audio_buf):
    '''Free a buffer previously allocated with `SDL_LoadWAV_RW` or
    `SDL_LoadWAV`.

    :Parameters:
     - `audio_buf`: `SDL_array`

    '''
    _SDL_FreeWAV(audio_buf.as_bytes().as_ctypes())

_SDL_BuildAudioCVT = SDL.dll.private_function('SDL_BuildAudioCVT',
    arg_types=[POINTER(SDL_AudioCVT), c_ushort, c_ubyte, c_uint,
               c_ushort, c_ubyte, c_uint],
    return_type=c_int,
    error_return=-1)

def SDL_BuildAudioCVT(src_format, src_channels, src_rate,
                      dst_format, dst_channels, dst_rate):
    '''Take a source format and rate and a destination format and rate,
    and return a `SDL_AudioCVT` structure.

    The `SDL_AudioCVT` structure is used by `SDL_ConvertAudio` to convert
    a buffer of audio data from one format to the other.

    :Parameters:
     - `src_format`: int
     - `src_channels`: int
     - `src_rate`: int
     - `dst_format`: int
     - `dst_channels`: int
     - `dst_rate`: int

    :rtype: `SDL_AudioCVT`
    '''
    cvt = SDL_AudioCVT()
    _SDL_BuildAudioCVT(cvt, src_format, src_channels, src_rate,
                       dst_format, dst_channels, dst_rate)
    return cvt

SDL_ConvertAudio = SDL.dll.function('SDL_ConvertAudio',
    '''Convert audio data in-place.

    Once you have initialized the 'cvt' structure using
    `SDL_BuildAudioCVT`, created an audio buffer ``cvt.buf``, and filled it
    with ``cvt.len`` bytes of audio data in the source format, this
    function will convert it in-place to the desired format.  The data
    conversion may expand the size of the audio data, so the buffer
    ``cvt.buf`` should be allocated after the cvt structure is initialized
    by `SDL_BuildAudioCVT`, and should be ``cvt->len*cvt->len_mult`` bytes
    long.

    Note that you are responsible for allocating the buffer.  The
    recommended way is to construct an `SDL_array` of the correct size,
    and set ``cvt.buf`` to the result of `SDL_array.as_ctypes`.

    :Parameters:
     - `cvt`: `SDL_AudioCVT`

    :rtype: int
    :return: undocumented
    ''',
    args=['cvt'],
    arg_types=[POINTER(SDL_AudioCVT)],
    return_type=c_int)

_SDL_MixAudio = SDL.dll.private_function('SDL_MixAudio',
    arg_types=[POINTER(c_ubyte), POINTER(c_ubyte), c_uint, c_int],
    return_type=None)

def SDL_MixAudio(dst, src, length, volume):
    '''Mix two audio buffers.

    This takes two audio buffers of the playing audio format and mixes
    them, performing addition, volume adjustment, and overflow clipping.
    The volume ranges from 0 - 128, and should be set to SDL_MIX_MAXVOLUME
    for full audio volume.  Note this does not change hardware volume.
    This is provided for convenience -- you can mix your own audio data.

    :note: SDL-ctypes doesn't know the current play format, so you must
        always pass in byte buffers (SDL_array or sequence) to this function,
        rather than of the native data type.

    :Parameters:
     - `dst`: `SDL_array`
     - `src`: `SDL_array`
     - `length`: int
     - `volume`: int

    '''
    dstref, dst = SDL.array.to_ctypes(dst, len(dst), c_ubyte)
    srcref, src = SDL.array.to_ctypes(src, len(src), c_ubyte)
    if len(dst) < length:
        raise TypeError, 'Destination buffer too small'
    elif len(src) < length:
        raise TypeError, 'Source buffer too small'
    _SDL_MixAudio(dst, src, length, volume)

SDL_LockAudio = SDL.dll.function('SDL_LockAudio',
    '''Guarantee the callback function is not running.

    The lock manipulated by these functions protects the callback function.
    During a LockAudio/UnlockAudio pair, you can be guaranteed that the
    callback function is not running.  Do not call these from the callback
    function or you will cause deadlock.
    ''',
    args=[],
    arg_types=[],
    return_type=None)

SDL_UnlockAudio = SDL.dll.function('SDL_UnlockAudio',
    '''Release the audio callback lock.

    :see: `SDL_LockAudio`
    ''',
    args=[],
    arg_types=[],
    return_type=None)

SDL_CloseAudio = SDL.dll.function('SDL_CloseAudio',
    '''Shut down audio processing and close the audio device.
    ''',
    args=[],
    arg_types=[],
    return_type=None)
