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

    :rtype: `SDL_Version`
    '''
    version = Sound_Version()
    _Sound_GetLinkedVersion(byref(version))
    return version

# Fill in non-standard linked version now, so "since" declarations can work
_dll._version = SDL.dll._version_parts(Sound_GetLinkedVersion())

# enum Sound_SampleFlags
SOUND_SAMPLEFLAG_NONE       = 0
SOUND_SAMPLEFLAG_CANSEEK    = 1
SOUND_SAMPLEFLAG_EOF        = 1 << 29
SOUND_SAMPLEFLAG_ERROR      = 1 << 30
SOUND_SAMPLEFLAG_EGAIN      = 1 << 31
# end enum Sound_SampleFlags

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
                ('buffer', c_void_p),
                ('buffer_size', c_uint),
                ('flags', c_int)]
    
    def __getattr__(self, name):
        if name == 'decoder':
            return self._decoder.contents
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
 
