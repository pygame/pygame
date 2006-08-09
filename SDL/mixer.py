#!/usr/bin/env python

'''A simple multi-channel audio mixer.

It supports 8 channels of 16 bit stereo audio, plus a single channel
of music, mixed by MidMod MOD, Timidity MIDI or SMPEG MP3 libraries.

The mixer can currently load Microsoft WAVE files and Creative Labs VOC
files as audio samples, and can load MIDI files via Timidity and the
following music formats via MikMod: MOD, S3M, IT, XM.  It can load Ogg
Vorbis streams as music if built with the Ogg Vorbis libraries, and finally
it can load MP3 music using the SMPEG library.

The process of mixing MIDI files to wave output is very CPU intensive, so
if playing regular WAVE files sounds great, but playing MIDI files sounds
choppy, try using 8-bit audio, mono audio, or lower frequencies.

:note: The music stream does not resample to the required audio rate.  You
    must call `Mix_OpenAudio` with the sampling rate of your music track.
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
    '''Internal format for an audio chunk.

    :Ivariables:
        `allocated` : int
            Undocumented.
        `abuf` : `SDL_array`
            Buffer of audio data
        `alen` : int
            Length of audio buffer
        `volume` : int
            Per-sample volume, in range [0, 128]

    '''
    _fields_ = [('allocated', c_int),
                ('_abuf', POINTER(c_ubyte)),
                ('alen', c_uint),
                ('volume', c_ubyte)]

    def __getattr__(self, attr):
        if attr == 'abuf':
            return SDL.array.SDL_array(self._abuf, self.alen, c_ubyte)
        raise AttributeException, attr

# opaque type
_Mix_Music = c_void_p

Mix_OpenAudio = _dll.function('Mix_OpenAudio',
    '''Open the mixer with a certain audio format.

    :Parameters:
        `frequency` : int
            Samples per second.  Typical values are 22050, 44100, 44800.
        `format` : int
            Audio format; one of AUDIO_U8, AUDIO_S8, AUDIO_U16LSB, 
            AUDIO_S16LSB, AUDIO_U16MSB or AUDIO_S16MSB
        `channels` : int
            Either 1 for monophonic or 2 for stereo.
        `chunksize` : int
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
    return_type=_Mix_Music,
    require_return=True)

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
    return_type=_Mix_Music,
    require_return=True)

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
    ref, mem = SDL.array.to_ctypes(mem, len(mem), c_ubyte)
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
    ref, mem = SDL.array.to_ctypes(mem, len(mem), c_ubyte)
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

_Mix_FilterFunc = CFUNCTYPE(None, c_void_p, POINTER(c_ubyte), c_int)
def _make_filter(func, udata):
    if func:
        def f(ignored, stream, len):
            stream = SDL.array.SDL_array(stream, len, c_ubyte)
            func(udata, stream)
        return _Mix_FilterFunc(f)
    else:
        return _Mix_FilterFunc()

_Mix_SetPostMix = _dll.private_function('Mix_SetPostMix',
    arg_types=[_Mix_FilterFunc, c_void_p],
    return_type=None)

_mix_postmix_ref = None

def Mix_SetPostMix(mix_func, udata):
    '''Set a function that is called after all mixing is performed.

    This can be used to provide real-time visual display of the audio
    stream or add a custom mixer filter for the stream data.

    :Parameters
        `mix_func` : function
            The function must have the signature 
            (stream: `SDL_array`, udata: any) -> None.  The first argument
            is the array of audio data that may be modified in place.
            `udata` is the value passed in this function.
        `udata` : any
            A variable that is passed to the `mix_func` function each
            call.

    '''
    global _mix_postmix_ref
    _mix_postmix_ref = _make_filter(mix_func, udata)
    _Mix_SetPostMix(_mix_postmix_ref, None)

_Mix_HookMusic = _dll.private_function('Mix_HookMusic',
    arg_types=[_Mix_FilterFunc, c_void_p],
    return_type=None)

_hookmusic_ref = None

def Mix_HookMusic(mix_func, udata):
    '''Add your own music player or additional mixer function.

    If `mix_func` is None, the default music player is re-enabled.

    :Parameters
        `mix_func` : function
            The function must have the signature 
            (stream: `SDL_array`, udata: any) -> None.  The first argument
            is the array of audio data that may be modified in place.
            `udata` is the value passed in this function.
        `udata` : any
            A variable that is passed to the `mix_func` function each
            call.

    '''
    global _hookmusic_ref
    _hookmusic_ref = _make_filter(mix_func, udata)
    _Mix_HookMusic(_hookmusic_ref, None)

_Mix_HookMusicFinishedFunc = CFUNCTYPE(None)

_Mix_HookMusicFinished = _dll.private_function('Mix_HookMusicFinished',
    arg_types=[_Mix_HookMusicFinishedFunc],
    return_type=None)

def Mix_HookMusicFinished(music_finished):
    '''Add your own callback when the music has finished playing.

    This callback is only called if the music finishes naturally.

    :Parameters:
        `music_finished` : function
            The callback takes no arguments and returns no value.

    '''
    if music_finished:
        _Mix_HookMusicFinished(_Mix_HookMusicFinishedFunc(music_finished))
    else:
        _Mix_HookMusicFinished(_Mix_HookMusicFinishedFunc())

# Mix_GetMusicHookData not implemented (unnecessary)

_Mix_ChannelFinishedFunc = CFUNCTYPE(None, c_int)

_Mix_ChannelFinished = _dll.private_function('Mix_ChannelFinished',
    arg_types=[_Mix_ChannelFinishedFunc],
    return_type=None)

# Keep the ctypes func around
_channelfinished_ref = None

def Mix_ChannelFinished(channel_finished):
    '''Add your own callback when a channel has finished playing.

    The callback may be called from the mixer's audio callback or it
    could be called as a result of `Mix_HaltChannel`, etc.

    Do not call `SDL_LockAudio` from this callback; you will either be
    inside the audio callback, or SDL_mixer will explicitly lock the
    audio before calling your callback.

    :Parameters:
        `channel_finished` : function
            The function takes the channel number as its only argument,
            and returns None.  Pass None here to disable the callback.

    '''
    global _channelfinished_ref
    if channel_finished:
        _channelfinished_ref = _Mix_ChannelFinishedFunc(channel_finished)
    else:
        _channelfinished_ref = _Mix_ChannelFinishedFunc()
    _Mix_ChannelFinished(_channelfinished_ref)

_Mix_EffectFunc = CFUNCTYPE(None, c_int, POINTER(c_ubyte), c_int, c_void_p)
def _make_Mix_EffectFunc(func, udata):
    if func:
        def f(chan, stream, len, ignored):
            stream = SDL.array.SDL_array(stream, len, c_ubyte)
            func(chan, stream, udata)
        return _Mix_EffectFunc(f)
    else:
        return _Mix_EffectFunc()

_Mix_EffectDoneFunc = CFUNCTYPE(None, c_int, c_void_p)
def _make_Mix_EffectDoneFunc(func, udata):
    if func:
        def f(chan, ignored):
            func(chan, udata)
        return _MixEffectDoneFunc(f)
    else:
        return _MixEffectDoneFunc()

_Mix_RegisterEffect = _dll.private_function('Mix_RegisterEffect',
    arg_types=\
     [c_int, _Mix_EffectFunc, _Mix_EffectDoneFunc, c_void_p],
    return_type=c_int,
    error_return=0)

_effect_func_refs = []

def Mix_RegisterEffect(chan, f, d, arg):
    '''Register a special effect function.

    At mixing time, the channel data is copied into a buffer and passed
    through each registered effect function.  After it passes through all
    the functions, it is mixed into the final output stream. The copy to
    buffer is performed once, then each effect function performs on the
    output of the previous effect. Understand that this extra copy to a
    buffer is not performed if there are no effects registered for a given
    chunk, which saves CPU cycles, and any given effect will be extra
    cycles, too, so it is crucial that your code run fast. Also note that
    the data that your function is given is in the format of the sound
    device, and not the format you gave to `Mix_OpenAudio`, although they
    may in reality be the same. This is an unfortunate but necessary speed
    concern. Use `Mix_QuerySpec` to determine if you can handle the data
    before you register your effect, and take appropriate actions.

    You may also specify a callback (`d`) that is called when the channel
    finishes playing. This gives you a more fine-grained control than
    `Mix_ChannelFinished`, in case you need to free effect-specific
    resources, etc. If you don't need this, you can specify None.

    You may set the callbacks before or after calling `Mix_PlayChannel`.

    Things like `Mix_SetPanning` are just internal special effect
    functions, so if you are using that, you've already incurred the
    overhead of a copy to a separate buffer, and that these effects will be
    in the queue with any functions you've registered. The list of
    registered effects for a channel is reset when a chunk finishes
    playing, so you need to explicitly set them with each call to
    ``Mix_PlayChannel*``.

    You may also register a special effect function that is to be run after
    final mixing occurs. The rules for these callbacks are identical to
    those in `Mix_RegisterEffect`, but they are run after all the channels
    and the music have been mixed into a single stream, whereas
    channel-specific effects run on a given channel before any other mixing
    occurs. These global effect callbacks are call "posteffects".
    Posteffects only have their `d` function called when they are
    unregistered (since the main output stream is never "done" in the same
    sense as a channel).  You must unregister them manually when you've had
    enough. Your callback will be told that the channel being mixed is
    (`MIX_CHANNEL_POST`) if the processing is considered a posteffect.

    After all these effects have finished processing, the callback
    registered through `Mix_SetPostMix` runs, and then the stream goes to
    the audio device.

    Do not call `SDL_LockAudio` from your callback function.

    :Parameters:
        `chan` : int
            Channel to set effect on, or `MIX_CHANNEL_POST` for postmix.
        `f` : function
            Callback function for effect.  Must have the signature
            (channel: int, stream: `SDL_array`, udata: any) -> None;
            where channel is the channel being affected, stream contains
            the audio data and udata is the user variable passed in to
            this function.
        `d` : function
            Callback function for when the effect is done.  The function
            must have the signature (channel: int, udata: any) -> None.
        `arg` : any
            User data passed to both callbacks.

    '''
    f = _make_MixEffectFunc(f, arg)
    d = _make_MixEffectDoneFunc(d, arg)
    _effect_func_refs.append(f)
    _effect_func_refs.append(d) 
    # TODO: override EffectDone callback to remove refs and prevent
    # memory leak.  Be careful with MIX_CHANNEL_POST
    _Mix_RegisterEffect(chan, f, d, arg)
            
# Mix_UnregisterEffect cannot be implemented

Mix_UnregisterAllEffects = _dll.function('Mix_UnregisterAllEffects',
    '''Unregister all effects for a channel.

    You may not need to call this explicitly, unless you need to stop all
    effects from processing in the middle of a chunk's playback. Note that
    this will also shut off some internal effect processing, since
    `Mix_SetPanning` and others may use this API under the hood. This is
    called internally when a channel completes playback.
   
    Posteffects are never implicitly unregistered as they are for channels,
    but they may be explicitly unregistered through this function by
    specifying `MIX_CHANNEL_POST` for a channel.

    :Parameters:
     - `channel`: int

    ''',
    args=['channel'],
    arg_types=[c_int],
    return_type=c_int,
    error_return=0)

Mix_SetPanning = _dll.function('Mix_SetPanning',
    '''Set the panning of a channel.

    The left and right channels are specified as integers between 0 and
    255, quietest to loudest, respectively.

    Technically, this is just individual volume control for a sample with
    two (stereo) channels, so it can be used for more than just panning.
    If you want real panning, call it like this::

        Mix_SetPanning(channel, left, 255 - left)

    Setting (channel) to `MIX_CHANNEL_POST` registers this as a posteffect, and
    the panning will be done to the final mixed stream before passing it on
    to the audio device.

    This uses the `Mix_RegisterEffect` API internally, and returns without
    registering the effect function if the audio device is not configured
    for stereo output. Setting both (left) and (right) to 255 causes this
    effect to be unregistered, since that is the data's normal state.

    :Parameters:
     - `channel`: int
     - `left`: int
     - `right`: int

    ''',
    args=['channel', 'left', 'right'],
    arg_types=[c_int, c_ubyte, c_ubyte],
    return_type=c_int,
    error_return=0)

Mix_SetPosition = _dll.function('Mix_SetPosition',
    '''Set the position of a channel.

    `angle` is an integer from 0 to 360, that specifies the location of the
    sound in relation to the listener. `angle` will be reduced as neccesary
    (540 becomes 180 degrees, -100 becomes 260).  Angle 0 is due north, and
    rotates clockwise as the value increases.  For efficiency, the
    precision of this effect may be limited (angles 1 through 7 might all
    produce the same effect, 8 through 15 are equal, etc).  `distance` is
    an integer between 0 and 255 that specifies the space between the sound
    and the listener. The larger the number, the further away the sound is.
    Using 255 does not guarantee that the channel will be culled from the
    mixing process or be completely silent. For efficiency, the precision
    of this effect may be limited (distance 0 through 5 might all produce
    the same effect, 6 through 10 are equal, etc). Setting `angle` and
    `distance` to 0 unregisters this effect, since the data would be
    unchanged.

    If you need more precise positional audio, consider using OpenAL for
    spatialized effects instead of SDL_mixer. This is only meant to be a
    basic effect for simple "3D" games.

    If the audio device is configured for mono output, then you won't get
    any effectiveness from the angle; however, distance attenuation on the
    channel will still occur. While this effect will function with stereo
    voices, it makes more sense to use voices with only one channel of
    sound, so when they are mixed through this effect, the positioning will
    sound correct. You can convert them to mono through SDL before giving
    them to the mixer in the first place if you like.

    Setting `channel` to `MIX_CHANNEL_POST` registers this as a posteffect,
    and the positioning will be done to the final mixed stream before
    passing it on to the audio device.

    This is a convenience wrapper over `Mix_SetDistance` and
    `Mix_SetPanning`.

    :Parameters:
     - `channel`: int
     - `angle`: int
     - `distance`: int

    ''',
    args=['channel', 'angle', 'distance'],
    arg_types=[c_int, c_short, c_ubyte],
    return_type=c_int,
    error_return=0)

Mix_SetDistance = _dll.function('Mix_SetDistance',
    '''Set the "distance" of a channel.

    `distance` is an integer from 0 to 255 that specifies the location of
    the sound in relation to the listener.  Distance 0 is overlapping the
    listener, and 255 is as far away as possible A distance of 255 does not
    guarantee silence; in such a case, you might want to try changing the
    chunk's volume, or just cull the sample from the mixing process with
    `Mix_HaltChannel`.

    For efficiency, the precision of this effect may be limited (distances
    1 through 7 might all produce the same effect, 8 through 15 are equal,
    etc).  `distance` is an integer between 0 and 255 that specifies the
    space between the sound and the listener. The larger the number, the
    further away the sound is.

    Setting `distance` to 0 unregisters this effect, since the data would
    be unchanged.

    If you need more precise positional audio, consider using OpenAL for
    spatialized effects instead of SDL_mixer. This is only meant to be a
    basic effect for simple "3D" games.
    
    Setting `channel` to `MIX_CHANNEL_POST` registers this as a posteffect,
    and the distance attenuation will be done to the final mixed stream
    before passing it on to the audio device.
    
    This uses the `Mix_RegisterEffect` API internally.

    :Parameters:
     - `channel`: int
     - `distance`: distance

    ''',
    args=['channel', 'distance'],
    arg_types=[c_int, c_ubyte],
    return_type=c_int,
    error_return=0)

Mix_SetReverseStereo = _dll.function('Mix_SetReverseStereo',
    '''Causes a channel to reverse its stereo.

    This is handy if the user has his or her speakers hooked up backwards,
    or you would like to have a minor bit of psychedelia in your sound
    code.  Calling this function with `flip` set to non-zero reverses the
    chunks's usual channels. If `flip` is zero, the effect is unregistered.
    
    This uses the `Mix_RegisterEffect` API internally, and thus is probably
    more CPU intensive than having the user just plug in his speakers
    correctly.  `Mix_SetReverseStereo` returns without registering the
    effect function if the audio device is not configured for stereo
    output.
    
    If you specify `MIX_CHANNEL_POST` for `channel`, then this the effect
    is used on the final mixed stream before sending it on to the audio
    device (a posteffect).

    :Parameters:
     - `channel`: int
     - `flip`: int

    ''',
    args=['channel', 'flip'],
    arg_types=[c_int, c_int],
    return_type=c_int,
    error_return=0)

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

Mix_GroupChannel = _dll.function('Mix_GroupChannel',
    '''Assing a channel to a group.

    A tag can be assigned to several mixer channels, to form groups
    of channels.  If `tag` is -1, the tag is removed (actually -1 is the
    tag used to represent the group of all the channels).

    :Parameters:
     - `channel`: int
     - `tag`: int

    ''',
    args=['channel', 'tag'],
    arg_types=[c_int, c_int],
    return_type=c_int,
    error_return=0)

Mix_GroupChannels = _dll.function('Mix_GroupChannels',
    '''Assign several consecutive channels to a group.

    A tag can be assigned to several mixer channels, to form groups
    of channels.  If `tag` is -1, the tag is removed (actually -1 is the
    tag used to represent the group of all the channels).

    :Parameters:
     - `channel_from`: int
     - `channel_to`: int
     - `tag`: int

    ''',
    args=['channel_from', 'channel_to', 'tag'],
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
        `channel` : int
            If -1, play on the first free channel.
        `chunk` : `Mix_Chunk`
            Chunk to play
        `loops` : int
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
        `channel` : int
            If -1, play on the first free channel.
        `chunk` : `Mix_Chunk`
            Chunk to play
        `loops` : int
            If greater than zero, the number of times to play the sound;
            if -1, loop infinitely.
        `ticks` : int
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
        `music` : ``Mix_Music``
            Chunk to play
        `loops` : int
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
        `music` : ``Mix_Music``
            Chunk to play
        `loops` : int
            If greater than zero, the number of times to play the sound;
            if -1, loop infinitely.
        `ms` : int
            Number of milliseconds to fade up over.
    ''',
    args=['music', 'loops', 'ms'],
    arg_types=[_Mix_Music, c_int, c_int],
    return_type=c_int,
    error_return=-1)

Mix_FadeInMusicPos = _dll.function('Mix_FadeInMusicPos',
    '''Fade in music at an offset over a period of time.

    :Parameters:
        `music` : ``Mix_Music``
            Chunk to play
        `loops` : int
            If greater than zero, the number of times to play the sound;
            if -1, loop infinitely.
        `ms` : int
            Number of milliseconds to fade up over.
        `position` : float
            Position within music to start at.  Currently implemented
            only for MOD, OGG and MP3.

    :see: Mix_SetMusicPosition
    ''',
    args=['music', 'loops', 'ms', 'position'],
    arg_types=[_Mix_Music, c_int, c_int, c_double],
    return_type=c_int,
    error_return=-1)

def Mix_FadeInChannel(channel, chunk, loops, ms):
    '''Fade in a channel.

    :Parameters:
        `channel` : int
            If -1, play on the first free channel.
        `chunk` : `Mix_Chunk`
            Chunk to play
        `loops` : int
            If greater than zero, the number of times to play the sound;
            if -1, loop infinitely.
        `ms` : int
            Number of milliseconds to fade up over.
    '''
    Mix_FadeInChannelTimed(channel, chunk, loops, -1)

Mix_FadeInChannelTimed = _dll.function('Mix_FadeInChannelTimed',
    '''Fade in a channel and play for a specified amount of time.

    :Parameters:
        `channel` : int
            If -1, play on the first free channel.
        `chunk` : `Mix_Chunk`
            Chunk to play
        `loops` : int
            If greater than zero, the number of times to play the sound;
            if -1, loop infinitely.
        `ms` : int
            Number of milliseconds to fade up over.
        `ticks` : int
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

Mix_FadingMusic = _dll.function('Mix_FadingMusic',
    '''Query the fading status of the music.

    :rtype: int
    :return: one of MIX_NO_FADING, MIX_FADING_OUT, MIX_FADING_IN.
    ''',
    args=[],
    arg_types=[],
    return_type=c_int)

Mix_FadingChannel = _dll.function('Mix_FadingChannel',
    '''Query the fading status of a channel.

    :Parameters:
     - `channel`: int

    :rtype: int
    :return: one of MIX_NO_FADING, MIX_FADING_OUT, MIX_FADING_IN.
    ''',
    args=['channel'],
    arg_types=[c_int],
    return_type=c_int)

Mix_Pause = _dll.function('Mix_Pause',
    '''Pause a particular channel.

    :Parameters:
     - `channel`: int

    ''',
    args=['channel'],
    arg_types=[c_int],
    return_type=None)

Mix_Resume = _dll.function('Mix_Resume',
    '''Resume a particular channel.

    :Parameters:
     - `channel`: int

    ''',
    args=['channel'],
    arg_types=[c_int],
    return_type=None)

Mix_Paused = _dll.function('Mix_Paused',
    '''Query if a channel is paused.

    :Parameters:
     - `channel`: int

    :rtype: int
    ''',
    args=['channel'],
    arg_types=[c_int],
    return_type=c_int)

Mix_PauseMusic = _dll.function('Mix_PauseMusic',
    '''Pause the music stream.
    ''',
    args=[],
    arg_types=[],
    return_type=None)

Mix_ResumeMusic = _dll.function('Mix_ResumeMusic',
    '''Resume the music stream.
    ''',
    args=[],
    arg_types=[],
    return_type=None)

Mix_RewindMusic = _dll.function('Mix_RewindMusic',
    '''Rewind the music stream.
    ''',
    args=[],
    arg_types=[],
    return_type=None)

Mix_PausedMusic = _dll.function('Mix_PausedMusic',
    '''Query if the music stream is paused.

    :rtype: int
    ''',
    args=[],
    arg_types=[],
    return_type=c_int)

Mix_SetMusicPosition = _dll.function('Mix_SetMusicPosition',
    '''Set the current position in the music stream.

    For MOD files the position represents the pattern order number;
    for OGG and MP3 files the position is in seconds.  Currently no other
    music file formats support positioning.

    :Parameters:
     - `position`: float

    ''',
    args=['position'],
    arg_types=[c_double],
    return_type=c_int,
    error_return=-1)

Mix_Playing = _dll.function('Mix_Playing',
    '''Query the status of a specific channel.

    :Parameters:
     - `channel`: int

    :rtype: int
    :return: the number of queried channels that are playing.
    ''',
    args=['channel'],
    arg_types=[c_int],
    return_type=c_int)

Mix_PlayingMusic = _dll.function('Mix_PlayingMusic',
    '''Query the status of the music stream.

    :rtype: int
    :return: 1 if music is playing, 0 otherwise.
    ''',
    args=[],
    arg_types=[],
    return_type=c_int)

Mix_SetMusicCMD = _dll.function('Mix_SetMusicCMD',
    '''Set the external music playback command.

    Any currently playing music will stop.

    :Parameters:
     - `command`: string

    ''',
    args=['command'],
    arg_types=[c_char_p],
    return_type=c_int,
    error_return=-1)

Mix_SetSynchroValue = _dll.function('Mix_SetSynchroValue',
    '''Set the synchro value for a MOD music stream.

    :Parameters:
     - `value`: int

    ''',
    args=['value'],
    arg_types=[c_int],
    return_type=c_int,
    error_return=-1)

Mix_GetSynchroValue = _dll.function('Mix_GetSynchroValue',
    '''Get the synchro value for a MOD music stream.

    :rtype: int
    ''',
    args=[],
    arg_types=[],
    return_type=c_int)

Mix_GetChunk = _dll.function('Mix_GetChunk',
    '''Get the chunk currently associated with a mixer channel.

    Returns None if the channel is invalid or if there's no chunk associated.

    :Parameters:
     - `channel`: int

    :rtype: `Mix_Chunk`
    ''',
    args=['channel'],
    arg_types=[c_int],
    return_type=POINTER(Mix_Chunk),
    dereference_return=True)

Mix_CloseAudio = _dll.function('Mix_CloseAudio',
    '''Close the mixer, halting all playing audio.

    ''',
    args=[],
    arg_types=[],
    return_type=None)


