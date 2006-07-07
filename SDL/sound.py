#!/usr/bin/env python

'''An abstract sound format decoding API.

The latest version of SDL_sound can be found at: http://icculus.org/SDL_sound/

The basic gist of SDL_sound is that you use an SDL_RWops to get sound data
into this library, and SDL_sound will take that data, in one of several
popular formats, and decode it into raw waveform data in the format of
your choice. This gives you a nice abstraction for getting sound into your
game or application; just feed it to SDL_sound, and it will handle
decoding and converting, so you can just pass it to your SDL audio
callback (or whatever). Since it gets data from an SDL_RWops, you can get
the initial sound data from any number of sources: file, memory buffer,
network connection, etc.

As the name implies, this library depends on SDL: Simple Directmedia Layer,
which is a powerful, free, and cross-platform multimedia library. It can
be found at http://www.libsdl.org/

Support is in place or planned for the following sound formats:
- .WAV  (Microsoft WAVfile RIFF data, internal.)
- .VOC  (Creative Labs' Voice format, internal.)
- .MP3  (MPEG-1 Layer 3 support, via the SMPEG and mpglib libraries.)
- .MID  (MIDI music converted to Waveform data, internal.)
- .MOD  (MOD files, via MikMod and ModPlug.)
- .OGG  (Ogg files, via Ogg Vorbis libraries.)
- .SPX  (Speex files, via libspeex.)
- .SHN  (Shorten files, internal.)
- .RAW  (Raw sound data in any format, internal.)
- .AU   (Sun's Audio format, internal.)
- .AIFF (Audio Interchange format, internal.)
- .FLAC (Lossless audio compression, via libFLAC.)
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.array
import SDL.dll
import SDL.rwops
import SDL.version

_dll = SDL.dll.SDL_DLL('SDL_sound', None)

class Sound_Version(Structure):
    '''Version structure.

    :Ivariables:
        `major` : int
            Major version number
        `minor` : int
            Minor version number
        `patch` : int
            Patch revision number

    '''
    _fields_ = [('major', c_int),
                ('minor', c_int),
                ('patch', c_int)]

    def __repr__(self):
        return '%d.%d.%d' % (self.major, self.minor, self.patch)

_Sound_GetLinkedVersion = _dll.private_function('Sound_GetLinkedVersion',
    arg_types=[POINTER(Sound_Version)],
    return_type=None)

def Sound_GetLinkedVersion():
    '''Get the version of the dynamically linked SDL_sound library

    :rtype: `Sound_Version`
    '''
    version = Sound_Version()
    _Sound_GetLinkedVersion(byref(version))
    return version

# Fill in non-standard linked version now, so "since" declarations can work
_dll._version = SDL.dll._version_parts(Sound_GetLinkedVersion())

class Sound_AudioInfo(Structure):
    '''Information about an existing sample's format.

    :Ivariables:
        `format` : int
            Equivalent to `SDL_AudioSpec.format`
        `channels` : int
            Number of sound channels.  1 == mono, 2 == stereo.
        `rate` : int
            Sample rate, in samples per second.

    :see: `Sound_SampleNew`
    :see: `Sound_SampleNewFromFile`
    '''
    _fields_ = [('format', c_ushort),
                ('channels', c_ubyte),
                ('rate', c_uint)]

class Sound_DecoderInfo(Structure):
    '''Information about a sound decoder.

    Each decoder sets up one of these structures, which can be retrieved
    via the `Sound_AvailableDecoders` function.  Fields in this
    structure are read-only.

    :Ivariables:
        `extensions` : list of str
            List of file extensions
        `description` : str
            Human-readable description of the decoder
        `author` : str
            Author and email address
        `url` : str
            URL specific to this decoder
    '''
    _fields_ = [('_extensions', POINTER(c_char_p)),
                ('description', c_char_p),
                ('author', c_char_p),
                ('url', c_char_p)]

    def __getattr__(self, name):
        if name == 'extensions':
            extensions = []
            ext_p = self._extensions
            i = 0
            while ext_p[i]:
                extensions.append(ext_p[i])
                i += 1
            return extensions
        raise AttributeError, name

class Sound_Sample(Structure):
    '''Represents sound data in the process of being decoded.

    The `Sound_Sample` structure is the heart of SDL_sound.  This holds
    information about a source of sound data as it is beind decoded.  All
    fields in this structure are read-only.

    :Ivariables:
        `decoder` : `Sound_DecoderInfo`
            Decoder used for this sample
        `desired` : `Sound_AudioInfo`
            Desired audio format for conversion
        `actual` : `Sound_AudioInfo`
            Actual audio format of the sample
        `buffer` : `SDL_array`
            Buffer of decoded data, as bytes
        `buffer_size` : int
            Current size of the buffer, in bytes
        `flags` : int
            Bitwise combination of SOUND_SAMPLEFLAG_CANSEEK,
            SOUND_SAMPLEFLAG_EOF, SOUND_SAMPLEFLAG_ERROR,
            SOUND_SAMPLEFLAG_EGAIN

    '''
    _fields_ = [('opaque', c_void_p),
                ('_decoder', POINTER(Sound_DecoderInfo)),
                ('desired', Sound_AudioInfo),
                ('actual', Sound_AudioInfo),
                ('_buffer', POINTER(c_ubyte)),
                ('buffer_size', c_uint),
                ('flags', c_int)]
    
    def __getattr__(self, name):
        if name == 'decoder':
            return self._decoder.contents
        elif name == 'buffer':
            return SDL.array.SDL_array(self._buffer, self.buffer_size, c_ubyte)
        raise AttributeError, name

Sound_Init = _dll.function('Sound_Init',
    '''Initialize SDL_sound.

    This must be called before any other SDL_sound function (except perhaps
    `Sound_GetLinkedVersion`). You should call `SDL_Init` before calling
    this.  `Sound_Init` will attempt to call ``SDL_Init(SDL_INIT_AUDIO)``,
    just in case.  This is a safe behaviour, but it may not configure SDL
    to your liking by itself.
    ''',
    args=[],
    arg_types=[],
    return_type=c_int,
    error_return=0)

Sound_Quit = _dll.function('Sound_Quit',
    '''Shutdown SDL_sound.

    This closes any SDL_RWops that were being used as sound sources, and
    frees any resources in use by SDL_sound.

    All Sound_Sample structures existing will become invalid.

    Once successfully deinitialized, `Sound_Init` can be called again to
    restart the subsystem. All default API states are restored at this
    point.

    You should call this before `SDL_Quit`. This will not call `SDL_Quit`
    for you.
    ''',
    args=[],
    arg_types=[],
    return_type=c_int,
    error_return=0)

_Sound_AvailableDecoders = _dll.private_function('Sound_AvailableDecoders',
    arg_types=[],
    return_type=POINTER(POINTER(Sound_DecoderInfo)))

def Sound_AvailableDecoders():
    '''Get a list of sound formats supported by this version of SDL_sound.

    This is for informational purposes only. Note that the extension listed
    is merely convention: if we list "MP3", you can open an MPEG-1 Layer 3
    audio file with an extension of "XYZ", if you like. The file extensions
    are informational, and only required as a hint to choosing the correct
    decoder, since the sound data may not be coming from a file at all,
    thanks to the abstraction that an SDL_RWops provides.

    :rtype: list of `Sound_DecoderInfo`
    '''
    decoders = []
    decoder_p = _Sound_AvailableDecoders()
    i = 0
    while decoder_p[i]:
        decoders.append(decoder_p[i].contents)
        i += 1
    return decoders
 
Sound_GetError = _dll.function('Sound_GetError',
    '''Get the last SDL_sound error message.

    This will be None if there's been no error since the last call to this
    function.  Each thread has a unique error state associated with it, but
    each time a new error message is set, it will overwrite the previous
    one associated with that thread.  It is safe to call this function at
    any time, even before `Sound_Init`.

    :rtype: str
    ''',
    args=[],
    arg_types=[],
    return_type=c_char_p)

Sound_ClearError = _dll.function('Sound_ClearError',
    '''Clear the current error message.

    The next call to `Sound_GetError` after `Sound_ClearError` will return
    None.
    ''',
    args=[],
    arg_types=[],
    return_type=None)

Sound_NewSample = _dll.function('Sound_NewSample',
    '''Start decoding a new sound sample.

    The data is read via an SDL_RWops structure, so it may be coming from
    memory, disk, network stream, etc. The `ext` parameter is merely a hint
    to determining the correct decoder; if you specify, for example, "mp3"
    for an extension, and one of the decoders lists that as a handled
    extension, then that decoder is given first shot at trying to claim the
    data for decoding. If none of the extensions match (or the extension is
    None), then every decoder examines the data to determine if it can
    handle it, until one accepts it. In such a case your SDL_RWops will
    need to be capable of rewinding to the start of the stream.

    If no decoders can handle the data, an exception is raised.

    Optionally, a desired audio format can be specified. If the incoming data
    is in a different format, SDL_sound will convert it to the desired format
    on the fly. Note that this can be an expensive operation, so it may be
    wise to convert data before you need to play it back, if possible, or
    make sure your data is initially in the format that you need it in.
    If you don't want to convert the data, you can specify None for a desired
    format. The incoming format of the data, preconversion, can be found
    in the `Sound_Sample` structure.

    Note that the raw sound data "decoder" needs you to specify both the
    extension "RAW" and a "desired" format, or it will refuse to handle
    the data. This is to prevent it from catching all formats unsupported
    by the other decoders.

    Finally, specify an initial buffer size; this is the number of bytes that
    will be allocated to store each read from the sound buffer. The more you
    can safely allocate, the more decoding can be done in one block, but the
    more resources you have to use up, and the longer each decoding call will
    take. Note that different data formats require more or less space to
    store. This buffer can be resized via `Sound_SetBufferSize`.

    The buffer size specified must be a multiple of the size of a single
    sample point. So, if you want 16-bit, stereo samples, then your sample
    point size is (2 channels   16 bits), or 32 bits per sample, which is four
    bytes. In such a case, you could specify 128 or 132 bytes for a buffer,
    but not 129, 130, or 131 (although in reality, you'll want to specify a
    MUCH larger buffer).

    When you are done with this `Sound_Sample` instance, you can dispose of
    it via `Sound_FreeSample`.

    You do not have to keep a reference to `rw` around. If this function
    suceeds, it stores `rw` internally (and disposes of it during the call
    to `Sound_FreeSample`). If this function fails, it will dispose of the
    SDL_RWops for you.

    :Parameters:
        `rw` : `SDL_RWops`
            SDL_RWops with sound data
        `ext` : str
            File extension normally associated with a data format.  Can
            usually be None.
        `desired` : `Sound_AudioInfo`
            Format to convert sound data into.  Can usually be None if you
            don't need conversion.
        `bufferSize` : int
            Size, in bytes, to allocate for the decoding buffer
    
    :rtype: `Sound_Sample`
    ''',
    args=['rw', 'ext', 'desired', 'bufferSize'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops), c_char_p,
               POINTER(Sound_AudioInfo), c_uint],
    return_type=POINTER(Sound_Sample),
    dereference_return=True,
    require_return=True)

_Sound_NewSampleFromMem = _dll.private_function('Sound_NewSampleFromMem',
    arg_types=[POINTER(c_ubyte), c_uint, c_char_p,
               POINTER(Sound_AudioInfo), c_uint],
    return_type=POINTER(Sound_Sample),
    dereference_return=True,
    require_return=True,
    since=(9,9,9))  # Appears in header only

def Sound_NewSampleFromMem(data, ext, desired, bufferSize):
    '''Start decoding a new sound sample from a buffer.

    This is identical to `Sound_NewSample`, but it creates an `SDL_RWops`
    for you from the buffer.

    :Parameters:
        `data` : `SDL_array` or sequence
            Buffer holding encoded byte sound data
        `ext` : str
            File extension normally associated with a data format.  Can
            usually be None.
        `desired` : `Sound_AudioInfo`
            Format to convert sound data into.  Can usually be None if you
            don't need conversion.
        `bufferSize` : int
            Size, in bytes, to allocate for the decoding buffer
    
    :rtype: `Sound_Sample`

    :since: Not yet released in SDL_sound
    '''
    ref, data = SDL.array.to_ctypes(data, len(data), c_ubyte)
    return _Sound_NewSampleFromMem(data, len(data), ext, desired, bufferSize)

Sound_NewSampleFromFile = _dll.function('Sound_NewSampleFromFile',
    '''Start decoding a new sound sample from a file on disk.

    This is identical to `Sound_NewSample`, but it creates an `SDL_RWops
    for you from the file located at `filename`.
    ''',
    args=['filename', 'desired', 'bufferSize'],
    arg_types=[c_char_p, POINTER(Sound_AudioInfo), c_uint],
    return_type=POINTER(Sound_Sample),
    dereference_return=True,
    require_return=True)

Sound_FreeSample = _dll.function('Sound_FreeSample',
    '''Dispose of a `Sound_Sample`.

    This will also close/dispose of the `SDL_RWops` that was used at
    creation time.  The `Sound_Sample` structure is invalid after this
    call.

    :Parameters:
        `sample` : `Sound_Sample`
            The sound sample to delete.

    ''',
    args=['sample'],
    arg_types=[POINTER(Sound_Sample)],
    return_type=None)

Sound_GetDuration = _dll.function('Sound_GetDuration',
    '''Retrieve the total play time of a sample, in milliseconds.

    Report total time length of sample, in milliseconds.  This is a fast
    call.  Duration is calculated during `Sound_NewSample`, so this is just
    an accessor into otherwise opaque data.

    Note that not all formats can determine a total time, some can't
    be exact without fully decoding the data, and thus will estimate the
    duration. Many decoders will require the ability to seek in the data
    stream to calculate this, so even if we can tell you how long an .ogg
    file will be, the same data set may fail if it's, say, streamed over an
    HTTP connection. 

    :Parameters:
        `sample` : `Sound_Sample`
            Sample from which to retrieve duration information.

    :rtype: int
    :return: Sample length in milliseconds, or -1 if duration can't be
        determined.

    :since: Not yet released in SDL_sound
    ''',
    args=['sample'],
    arg_types=[POINTER(Sound_Sample)],
    return_type=c_int,
    since=(9,9,9))

Sound_SetBufferSize = _dll.function('Sound_SetBufferSize',
    '''Change the current buffer size for a sample.

    If the buffer size could be changed, then the ``sample.buffer`` and
    ``sample.buffer_size`` fields will reflect that. If they could not be
    changed, then your original sample state is preserved. If the buffer is
    shrinking, the data at the end of buffer is truncated. If the buffer is
    growing, the contents of the new space at the end is undefined until you
    decode more into it or initialize it yourself.

    The buffer size specified must be a multiple of the size of a single
    sample point. So, if you want 16-bit, stereo samples, then your sample
    point size is (2 channels   16 bits), or 32 bits per sample, which is four
    bytes. In such a case, you could specify 128 or 132 bytes for a buffer,
    but not 129, 130, or 131 (although in reality, you'll want to specify a
    MUCH larger buffer).

    :Parameters:
        `sample` : `Sound_Sample`
            Sample to modify
        `new_size` : int
            The desired size, in bytes of the new buffer

    ''',
    args=['sample', 'new_size'],
    arg_types=[POINTER(Sound_Sample), c_uint],
    return_type=c_int,
    error_return=0)

Sound_Decode = _dll.function('Sound_Decode',
    '''Decode more of the sound data in a `Sound_Sample`.

    It will decode at most sample->buffer_size bytes into ``sample.buffer``
    in the desired format, and return the number of decoded bytes.

    If ``sample.buffer_size`` bytes could not be decoded, then refer to
    ``sample.flags`` to determine if this was an end-of-stream or error
    condition.

    :Parameters:
        `sample` : `Sound_Sample`
            Do more decoding to this sample
    
    :rtype: int
    :return: number of bytes decoded into ``sample.buffer``
    ''',
    args=['sample'],
    arg_types=[POINTER(Sound_Sample)],
    return_type=c_uint)

Sound_DecodeAll = _dll.function('Sound_DecodeAll',
    '''Decode the remainder of the sound data in a `Sound_Sample`.

    This will dynamically allocate memory for the entire remaining sample.
    ``sample.buffer_size`` and ``sample.buffer`` will be updated to reflect
    the new buffer.  Refer to ``sample.flags`` to determine if the
    decoding finished due to an End-of-stream or error condition.

    Be aware that sound data can take a large amount of memory, and that
    this function may block for quite awhile while processing. Also note
    that a streaming source (for example, from a SDL_RWops that is getting
    fed from an Internet radio feed that doesn't end) may fill all available
    memory before giving up...be sure to use this on finite sound sources
    only.

    When decoding the sample in its entirety, the work is done one buffer
    at a time. That is, sound is decoded in ``sample.buffer_size`` blocks, and
    appended to a continually-growing buffer until the decoding completes.
    That means that this function will need enough RAM to hold
    approximately ``sample.buffer_size`` bytes plus the complete decoded
    sample at most. The larger your buffer size, the less overhead this
    function needs, but beware the possibility of paging to disk. Best to
    make this user-configurable if the sample isn't specific and small.

    :Parameters:
        `sample` : `Sound_Sample`
            Do all decoding for this sample.
    
    :rtype: int
    :return: number of bytes decoded into ``sample.buffer``
    ''',
    args=['sample'],
    arg_types=[POINTER(Sound_Sample)],
    return_type=c_uint)

Sound_Rewind = _dll.function('Sound_Rewind',
    '''Rewind a sample to the start.

    Restart a sample at the start of its waveform data, as if newly
    created with `Sound_NewSample`. If successful, the next call to
    `Sound_Decode` will give audio data from the earliest point in the
    stream.

    Beware that this function will fail if the SDL_RWops that feeds the
    decoder can not be rewound via it's seek method, but this can
    theoretically be avoided by wrapping it in some sort of buffering
    SDL_RWops.

    This function will raise an exception if the RWops is not seekable, or
    SDL_sound is not initialized.

    If this function fails, the state of the sample is undefined, but it
    is still safe to call `Sound_FreeSample` to dispose of it.

    :Parameters:
        `sample` : `Sound_Sample`
            The sample to rewind

    ''',
    args=['sample'],
    arg_types=[POINTER(Sound_Sample)],
    return_type=c_int,
    error_return=0)

Sound_Seek = _dll.function('Sound_Seek',
    '''Seek to a different point in a sample.

    Reposition a sample's stream. If successful, the next call to
    `Sound_Decode` or `Sound_DecodeAll` will give audio data from the
    offset you specified.

    The offset is specified in milliseconds from the start of the
    sample.

    Beware that this function can fail for several reasons. If the
    SDL_RWops that feeds the decoder can not seek, this call will almost
    certainly fail, but this can theoretically be avoided by wrapping it
    in some sort of buffering SDL_RWops. Some decoders can never seek,
    others can only seek with certain files. The decoders will set a flag
    in the sample at creation time to help you determine this.

    You should check ``sample.flags & SOUND_SAMPLEFLAG_CANSEEK``
    before attempting. `Sound_Seek` reports failure immediately if this
    flag isn't set. This function can still fail for other reasons if the
    flag is set.

    This function can be emulated in the application with `Sound_Rewind`
    and predecoding a specific amount of the sample, but this can be
    extremely inefficient. `Sound_Seek()` accelerates the seek on a
    with decoder-specific code.

    If this function fails, the sample should continue to function as if
    this call was never made. If there was an unrecoverable error,
    ``sample.flags & SOUND_SAMPLEFLAG_ERROR`` will be set, which your
    regular decoding loop can pick up.

    On success, ERROR, EOF, and EAGAIN are cleared from sample->flags.

    :Parameters:
        `sample` : `Sound_Sample`
            The sample to seek
        `ms` : int
            The new position, in milliseconds, from the start of sample

    ''',
    args=['sample', 'ms'],
    arg_types=[POINTER(Sound_Sample), c_uint],
    return_type=c_int,
    error_return=0)

    
