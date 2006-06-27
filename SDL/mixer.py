#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.array
import SDL.dll
import SDL.rwops
import SDL.version

_dll = SDL.dll.SDL_DLL('SDL_mixer', 'Mix_Linked_Version')

Mix_Linked_Version = _dll.function('Mix_Linked_Version',
    '''Get the version of the dynamically linked SDL_mixer library.
    ''',
    args=[],
    arg_types=[],
    return_type=POINTER(SDL.version.SDL_version),
    dereference_return=True,
    require_return=True)

class Mix_Chunk(Structure):
    _fields_ = [('allocated', c_int),
                ('abuf', POINTER(c_ubyte)),
                ('alen', c_uint),
                ('volume', c_ubyte)]

# begin enum Mix_Fading
(MIX_NO_FADING,
    MIX_FADING_OUT,
    MIX_FADING_IN) = range(3)
# end enum Mix_Fading

# begin enum Mix_MusicType
(MUS_NONE,
    MUS_CMD,
    MUS_WAV,
    MUS_MOD,
    MUS_MID,
    MUS_OGG,
    MUS_MP3) = range(7)
# end enum Mix_MusicType

# opaque type
_Mix_Music = c_void_p

Mix_OpenAudio = _dll.function('Mix_OpenAudio',
    '''Open the mixer with a certain audio format.

    :Parameters:
        `frequency`: int
            Samples per second.  Typical values are 22050, 44100, 44800.
        `format`: int
            Audio format; one of AUDIO_U8, AUDIO_S8, AUDIO_U16LSB, 
            AUDIO_S16LSB, AUDIO_U16MSB or AUDIO_S16MSB
        `channels`: int
            Either 1 for monophonic or 2 for stereo.
        `chunksize`: int
            Size of the audio buffer.  Typical values are 4096, 8192.

    ''',
    args=['frequency', 'format', 'channels', 'chunksize'],
    arg_types=[c_int, c_ushort, c_int, c_int],
    return_type=c_int,
    error_return=-1)

Mix_AllocateChannels = _dll.function('Mix_AllocateChannels',
    '''Dynamically change the number of channels managed by the mixer.

    If decreasing the number of channels, the upper channels
    are stopped.

    :Parameters:
     - `numchans`: int

    :rtype: int
    :return: the new number of allocated channels.
    ''',
    args=['numchans'],
    arg_types=[c_int],
    return_type=c_int)

_Mix_QuerySpec = _dll.private_function('Mix_QuerySpec',
    arg_types=[POINTER(c_int), POINTER(c_ushort), POINTER(c_int)],
    return_type=c_int)

def Mix_QuerySpec():
    '''Find out what the actual audio device parameters are.

    The function returns a tuple giving each parameter value.  The first
    value is 1 if the audio has been opened, 0 otherwise.

    :rtype: (int, int, int, int)
    :return: (opened, frequency, format, channels)
    '''
    frequency, format, channels = c_int(), c_ushort(), c_int()
    opened = _Mix_QuerySpec(byref(frequency), byref(format), byref(channels))
    return opened, frequency.value, format.value, channels.value

Mix_LoadWAV_RW = _dll.function('Mix_LoadWAV_RW',
    '''Load a WAV, RIFF, AIFF, OGG or VOC file from a RWops source.



    :rtype: `Mix_Chunk`
    ''',
    args=['src', 'freesrc'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops), c_int],
    return_type=POINTER(Mix_Chunk),
    dereference_return=True,
    require_return=True)

def Mix_LoadWAV(file):
    '''Load a WAV, RIFF, AIFF, OGG or VOC file.

    :Parameters:
        `file` : string
            Filename to load.

    :rtype: `Mix_Chunk`
    '''
    return Mix_LoadWAV_RW(SDL.rwops.SDL_RWFromFile(file, 'rb'), 1)

Mix_LoadMUS = _dll.function('Mix_LoadMUS',
    '''Load a WAV, MID, OGG, MP3 or MOD file.

    :Parameters:
        `file` : string
            Filename to load.

    :rtype: ``Mix_Music``
    ''',
    args=['file'],
    arg_types=[c_char_p],
    return_type=_Mix_Music)

Mix_LoadMUS_RW = _dll.function('Mix_LoadMUS_RW',
    '''Load a MID, OGG, MP3 or MOD file from a RWops source.

    :Parameters:
        `src` : `SDL_RWops`
            Readable RWops to load from.
        `freesrc` : `int`
            If non-zero, the source will be closed after loading.

    :rtype: ``Mix_Music``
    ''',
    args=['file'],
    arg_types=[c_char_p],
    return_type=_Mix_Music)

_Mix_QuickLoad_WAV = _dll.private_function('Mix_QuickLoad_WAV',
    arg_types=[POINTER(c_ubyte)],
    return_type=POINTER(Mix_Chunk),
    dereference_return=True,
    require_return=True)

def Mix_QuickLoad_WAV(mem):
    '''Load a wave file of the mixer format from a sequence or SDL_array.

    :Parameters:
     - `mem`: sequence or `SDL_array`

    :rtype: `Mix_Chunk`
    '''
    mem = SDL.array.to_ctypes(mem, len(mem), c_ubyte)
    return _Mix_QuickLoad_WAV(mem)

_Mix_QuickLoad_RAW = _dll.private_function('Mix_QuickLoad_RAW',
    arg_types=[POINTER(c_ubyte), c_uint],
    return_type=POINTER(Mix_Chunk),
    dereference_return=True,
    require_return=True)

def Mix_QuickLoad_RAW(mem):
    '''Load raw audio data of the mixer format from a sequence or SDL_array.

    :Parameters:
     - `mem`: sequence or `SDL_array`

    :rtype: `Mix_Chunk`
    '''
    l = len(mem)
    mem = SDL.array.to_ctypes(mem, len(mem), c_ubyte)
    return _Mix_QuickLoad_RAW(mem, l)

Mix_FreeChunk = _dll.function('Mix_FreeChunk',
    '''Free an audio chunk previously loaded.

    :Parameters:
     - `chunk`: `Mix_Chunk`

    ''',
    args=['chunk'],
    arg_types=[POINTER(Mix_Chunk)],
    return_type=None)

Mix_FreeMusic = _dll.function('Mix_FreeMusic',
    '''Free a music chunk previously loaded.

    :Parameters:
     - `music`: ``Mix_Music``

    ''',
    args=['music'],
    arg_types=[_Mix_Music],
    return_type=None)

Mix_GetMusicType = _dll.function('Mix_GetMusicType',
    '''Get the music format of a mixer music.

    Returns the format of the currently playing music if `music` is None.

    :Parameters:
     - `music`: ``Mix_Music``

    :rtype: int
    :return: one of `MUS_NONE`, `MUS_CMD`, `MUS_WAV`, `MUS_MOD`, `MUS_MID`,
        `MUS_OGG` or `MUS_MP3`
    ''',
    args=['music'],
    arg_types=[_Mix_Music],
    return_type=c_int)

# TODO hooks and effects

Mix_ReserveChannels = _dll.function('Mix_ReserveChannels',
    '''Reserve the first channels (0 to n-1) for the application.

    If reserved, a channel will not be allocated dynamically to a sample
    if requested with one of the ``Mix_Play*`` functions.

    :Parameters:
     - `num`: int

    :rtype: int
    :return: the number of reserved channels
    ''',
    args=['num'],
    arg_types=[c_int],
    return_type=c_int)

Mix_GroupChannels = _dll.function('Mix_GroupChannels',
    '''Assign several consecutive channels to a group.

    A tag can be assigned to several mixer channels, to form groups
    of channels.  If `tag` is -1, the tag is removed (actually -1 is the
    tag used to represent the group of all the channels).

    :Parameters:
     - `from`: int
     - `to`: int
     - `tag`: int

    ''',
    args=['from', 'to', 'tag'],
    arg_types=[c_int, c_int, c_int],
    return_type=c_int,
    error_return=0)

Mix_GroupAvailable = _dll.function('Mix_GroupAvailable',
    '''Find the first available channel in a group of channels.

    :Parameters:
     - `tag`: int

    :rtype: int
    :return: a channel, or -1 if none are available.
    ''',
    args=['tag'],
    arg_types=[c_int],
    return_type=c_int)

Mix_GroupCount = _dll.function('Mix_GroupCount',
    '''Get the number of channels in a group.

    If `tag` is -1, returns the total number of channels.

    :Parameters:
     - `tag`: int

    :rtype: int
    ''',
    args=['tag'],
    arg_types=[c_int],
    return_type=c_int)

Mix_GroupOldest = _dll.function('Mix_GroupOldest',
    '''Find the "oldest" sample playing in a group of channels.

    :Parameters:
     - `tag`: int

    :rtype: int
    ''',
    args=['tag'],
    arg_types=[c_int],
    return_type=c_int)

Mix_GroupNewer = _dll.function('Mix_GroupNewer',
    '''Find the "most recent" (i.e., last) sample playing in a group of
    channels.

    :Parameters:
     - `tag`: int

    :rtype: int
    ''',
    args=['tag'],
    arg_types=[c_int],
    return_type=c_int)

def Mix_PlayChannel(channel, chunk, loops):
    '''Play an audio chunk on a specific channel.

    :Parameters:
        `channel`: int
            If -1, play on the first free channel.
        `chunk`: `Mix_Chunk`
            Chunk to play
        `loops`: int
            If greater than zero, the number of times to play the sound;
            if -1, loop infinitely.
    
    :rtype: int
    :return: the channel that was used to play the sound.
    '''
    return Mix_PlayChannelTimed(channel, chunk, loops, -1)

Mix_PlayChannelTimed = _dll.function('Mix_PlayChannelTimed',
    '''Play an audio chunk on a specific channel for a specified amount of
    time.

    :Parameters:
        `channel`: int
            If -1, play on the first free channel.
        `chunk`: `Mix_Chunk`
            Chunk to play
        `loops`: int
            If greater than zero, the number of times to play the sound;
            if -1, loop infinitely.
        `ticks`: int
            Maximum number of milliseconds to play sound for.
    
    :rtype: int
    :return: the channel that was used to play the sound.
    ''',
    args=['channel', 'chunk', 'loops', 'ticks'],
    arg_types=[c_int, POINTER(Mix_Chunk), c_int, c_int],
    return_type=c_int)

Mix_PlayMusic = _dll.function('Mix_PlayMusic',
    '''Play a music chunk.

    :Parameters:
        `music`: ``Mix_Music``
            Chunk to play
        `loops`: int
            If greater than zero, the number of times to play the sound;
            if -1, loop infinitely.
    ''',
    args=['music', 'loops'],
    arg_types=[_Mix_Music, c_int],
    return_type=c_int,
    error_return=-1)

Mix_FadeInMusic = _dll.function('Mix_FadeInMusic',
    '''Fade in music over a period of time.

    :Parameters:
        `music`: ``Mix_Music``
            Chunk to play
        `loops`: int
            If greater than zero, the number of times to play the sound;
            if -1, loop infinitely.
        `ms`: int
            Number of milliseconds to fade up over.
    ''',
    args=['music', 'loops', 'ms'],
    arg_types=[_Mix_Music, c_int, c_int],
    return_type=c_int,
    error_return=-1)

Mix_FadeInMusicPos = _dll.function('Mix_FadeInMusicPos',
    '''Fade in music at an offset over a period of time.

    :Parameters:
        `music`: ``Mix_Music``
            Chunk to play
        `loops`: int
            If greater than zero, the number of times to play the sound;
            if -1, loop infinitely.
        `ms`: int
            Number of milliseconds to fade up over.
        `position`: double
            Position within music to start at.  Currently implemented
            only for MOD, OGG and MP3.
    ''',
    args=['music', 'loops', 'ms', 'position'],
    arg_types=[_Mix_Music, c_int, c_int, c_double],
    return_type=c_int,
    error_return=-1)

def Mix_FadeInChannel(channel, chunk, loops, ms):
    '''Fade in a channel.

    :Parameters:
        `channel`: int
            If -1, play on the first free channel.
        `chunk`: `Mix_Chunk`
            Chunk to play
        `loops`: int
            If greater than zero, the number of times to play the sound;
            if -1, loop infinitely.
        `ms`: int
            Number of milliseconds to fade up over.
    '''
    Mix_FadeInChannelTimed(channel, chunk, loops, -1)

Mix_FadeInChannelTimed = _dll.function('Mix_FadeInChannelTimed',
    '''Fade in a channel and play for a specified amount of time.

    :Parameters:
        `channel`: int
            If -1, play on the first free channel.
        `chunk`: `Mix_Chunk`
            Chunk to play
        `loops`: int
            If greater than zero, the number of times to play the sound;
            if -1, loop infinitely.
        `ms`: int
            Number of milliseconds to fade up over.
        `ticks`: int
            Maximum number of milliseconds to play sound for.
    ''',
    args=['channel', 'music', 'loops', 'ms', 'ticks'],
    arg_types=[c_int, _Mix_Music, c_int, c_int, c_int],
    return_type=c_int,
    error_return=-1)

Mix_Volume = _dll.function('Mix_Volume',
    '''Set the volume in the range of 0-128 of a specific channel.

    :Parameters:
        `channel` : int
            If -1, set the volume for all channels
        `volume` : int
            Volume to set, in the range 0-128, or -1 to just return the
            current volume.

    :rtype: int
    :return: the original volume.
    ''',
    args=['channel', 'volume'],
    arg_types=[c_int, c_int],
    return_type=c_int)

Mix_VolumeChunk = _dll.function('Mix_VolumeChunk',
    '''Set the volume in the range of 0-128 of a chunk.

    :Parameters:
        `chunk` : `Mix_Chunk`
            Chunk to set volume.
        `volume` : int
            Volume to set, in the range 0-128, or -1 to just return the
            current volume.

    :rtype: int
    :return: the original volume.
    ''',
    args=['chunk', 'volume'],
    arg_types=[POINTER(Mix_Chunk), c_int],
    return_type=c_int)

Mix_VolumeMusic = _dll.function('Mix_VolumeMusic',
    '''Set the volume in the range of 0-128 of the music.

    :Parameters:
        `volume` : int
            Volume to set, in the range 0-128, or -1 to just return the
            current volume.

    :rtype: int
    :return: the original volume.
    ''',
    args=['volume'],
    arg_types=[c_int],
    return_type=c_int)

Mix_HaltChannel = _dll.function('Mix_HaltChannel',
    '''Halt playing of a particular channel.

    :Parameters:
     - `channel`: int
    ''',
    args=['channel'],
    arg_types=[c_int],
    return_type=None)

Mix_HaltGroup = _dll.function('Mix_HaltGroup',
    '''Halt playing of a particular group.

    :Parameters:
     - `tag`: int
    ''',
    args=['tag'],
    arg_types=[c_int],
    return_type=None)

Mix_HaltMusic = _dll.function('Mix_HaltMusic',
    '''Halt playing music. 
    ''',
    args=[],
    arg_types=[],
    return_type=None)

Mix_ExpireChannel = _dll.function('Mix_ExpireChannel',
    '''Change the expiration delay for a particular channel.

    The sample will stop playing afte the `ticks` milliseconds have
    elapsed, or remove the expiration if `ticks` is -1.

    :Parameters:
     - `channel`: int
     - `ticks`: int

    :rtype: int
    :return: the number of channels affected.
    ''',
    args=['channel', 'ticks'],
    arg_types=[c_int, c_int],
    return_type=c_int)

Mix_FadeOutChannel = _dll.function('Mix_FadeOutChannel',
    '''Halt a channel, fading it out progressively until it's silent.

    The `ms` parameter indicates the number of milliseconds the fading
    will take.

    :Parameters:
     - `channel`: int
     - `ms`: int
    ''',
    args=['channel', 'ms'],
    arg_types=[c_int, c_int],
    return_type=None)

Mix_FadeOutGroup = _dll.function('Mix_FadeOutGroup',
    '''Halt a group, fading it out progressively until it's silent.

    The `ms` parameter indicates the number of milliseconds the fading
    will take.

    :Parameters:
     - `tag`: int
     - `ms`: int
    ''',
    args=['tag', 'ms'],
    arg_types=[c_int, c_int],
    return_type=None)

Mix_FadeOutMusic = _dll.function('Mix_FadeOutMusic',
    '''Halt playing music, fading it out progressively until it's silent.

    The `ms` parameter indicates the number of milliseconds the fading
    will take.

    :Parameters:
     - `ms`: int
    ''',
    args=['ms'],
    arg_types=[c_int],
    return_type=None)


