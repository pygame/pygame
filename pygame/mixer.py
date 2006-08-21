#!/usr/bin/env python

'''Pygame module for loading and playing sounds.

This module contains classes for loading Sound objects and controlling
playback. The mixer module is optional and depends on SDL_mixer. Your
program should test that pygame.mixer is available and intialized before
using it.

The mixer module has a limited number of channels for playback of sounds.
Usually programs tell pygame to start playing audio and it selects an
available channel automatically. The default is 8 simultaneous channels,
but complex programs can get more precise control over the number of
channels and their use.

All sound playback is mixed in background threads. When you begin
to play a Sound object, it will return immediately while the sound
continues to play. A single Sound object can also be actively played
back multiple times.

The mixer also has a special streaming channel. This is for music
playback and is accessed through the pygame.mixer.music module.

The mixer module must be initialized like other pygame modules, but it has
some extra conditions. The pygame.mixer.init() function takes several
optional arguments to control the playback rate and sample size. Pygame
will default to reasonable values, but pygame cannot perform Sound
resampling, so the mixer should be initialized to match the values of
your audio resources.


NOTE: there is currently a bug on some windows machines which makes
sound play back 'scratchy'.  There is not enough cpu in the sound 
thread to feed the buffer to the sound api.
To get around this you can increase the buffer size.  However this
means that there is more of a delay between the time you ask to play
the sound and when it gets played.  Try calling this before the pygame.init or 
pygame.mixer.init calls.  pygame.mixer.pre_init(44100,-16,2, 1024 * 3)
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from SDL import *
from SDL.mixer import *
from SDL.rwops import *

import pygame.base

try:
    from pygame import music
    _have_music = True
except ImportError:
    _have_music = False

_request_frequency = MIX_DEFAULT_FREQUENCY
_request_size = MIX_DEFAULT_FORMAT
_request_stereo = MIX_DEFAULT_CHANNELS
_request_buffer = 1024

_channels = {}

def __PYGAMEinit__(frequency=None, size=None, stereo=None, buffer=None):
    if not frequency:
        frequency = _request_frequency
    if not size:
        size = _request_size
    if not stereo:
        stereo = _request_stereo
    if not buffer:
        buffer = _request_buffer

    stereo = min(2, stereo)

    if size == 8:
        size = AUDIO_U8
    elif size == -8:
        size = AUDIO_S8
    elif size == 16:
        size ==AUDIO_U16SYS
    elif size == -16:
        size = AUDIO_S16SYS

    # Make buffer power of 2
    i = 256
    while i < buffer:
        i <<= 1
    buffer = i
    
    global _endsound_callback
    if not SDL_WasInit(SDL_INIT_AUDIO):
        pygame.base.register_quit(_autoquit)

        SDL_InitSubSystem(SDL_INIT_AUDIO)
        Mix_OpenAudio(frequency, size, stereo, buffer)
        if Mix_Linked_Version().is_since((1,2,3)):
            Mix_ChannelFinished(_endsound_callback)
        Mix_VolumeMusic(127)

    return 1

def _autoquit():
    global _channels
    if SDL_WasInit(SDL_INIT_AUDIO):
        Mix_HaltMusic()
        _channels = {}
        if _have_music:
            music._free_loaded()
        Mix_CloseAudio()
        SDL_QuitSubSystem(SDL_INIT_AUDIO)

def _endsound_callback(channel):
    channel = _channels.get(channel, None)
    if not channel:
        return

    if channel._endevent and SDL_WasInit(SDL_INIT_VIDEO):
        e = SDL_Event()
        e.type = channel._endevent
        e = e.specialize()
        if isinstance(e, SDL_UserEvent):
            e.code = channel._endevent
        SDL_PushEvent(cast(pointer(e), POINTER(SDL_Event)))
    if channel._queue:
        channel._sound = channel._queue
        channel._queue = None
        channelnum = \
            Mix_PlayChannelTimed(channel._id, channel._sound._chunk, 0, -1)
        if channelnum != -1:
            Mix_GroupChannel(channelnum, id(channel._sound))
    else:
        channel._sound = None

def init(frequency=None, size=None, stereo=None, buffer=None):
    '''Initialize the mixer module.

    Initialize the mixer module for Sound loading and playback. The default
    arguments can be overridden to provide specific audio mixing. The size
    argument represents how many bits are used for each audio sample. If the
    value is negative then signed sample values will be used. Positive values
    mean unsigned audio samples will be used. If the stereo argument is false
    the mixer will use mono sound.

    The buffer argument controls the number of internal samples used in the
    sound mixer. The default value should work for most cases. It can be
    lowered to reduce latency, but sound dropout may occur. It can be raised
    to larger values to ensure playback never skips, but it will impose latency
    on sound playback. The buffer size must be a power of two. 

    Some platforms require the pygame.mixer module to be initialized after the
    display modules have initialized. The top level pygame.init() takes care
    of this automatically, but cannot pass any arguments to the mixer init. To
    solve this, mixer has a function pygame.mixer.pre_init() to set the proper
    defaults before the toplevel init is used.

    It is safe to call this more than once, but after the mixer is initialized
    you cannot change the playback arguments without first calling
    pygame.mixer.quit().

    :Parameters:
        `frequency` : int
            Sample rate, in Hertz; defaults to 22050
        `size` : int
            Bits per sample per channel; defaults to -16.  Positive values for
            unsigned values, negative for signed.
        `stereo` : int
            Number of output channels: 1 for mono, 2 for stereo; defaults to
            2.
        `buffer` : int
            Byte size of each output channel's buffer; a power of two;
            defaults to 1024.

    '''
    __PYGAMEinit__(frequency, size, stereo, buffer)

def pre_init(frequency=0, size=0, stereo=0, buffer=0):
    '''Preset the mixer init arguments.

    Any nonzero arguments change the default values used when the real
    pygame.mixer.init() is called. The best way to set custom mixer playback
    values is to call pygame.mixer.pre_init() before calling the top level
    pygame.init().

    :Parameters:
        `frequency` : int
            Sample rate, in Hertz
        `size` : int
            Bits per sample per channel.  Positive values for unsigned
            values, negative for signed.
        `stereo` : bool
            Number of mixdown channels: False for 1, True for 2.
        `buffer` : int
            Bytes for mixdown buffer size; a power of two.

    '''
    global _request_frequency
    global _request_size
    global _request_stereo
    global _request_buffer
    if frequency:
        _request_frequency = frequency
    if size:
        _request_size = size
    if stereo:
        _request_stereo = stereo
    if buffer:
        _request_buffer = buffer

def quit():
    '''Uninitialize the mixer.

    This will uninitialize pygame.mixer. All playback will stop and any loaded
    Sound objects may not be compatable with the mixer if it is reinitialized
    later.
    '''
    _autoquit()

def get_init():
    '''Determine if the mixer is initialized.

    If the mixer is initialized, this returns the playback arguments it 
    is using. If the mixer has not been initialized this returns None

    The value of `size` follows the same conventions as in `init`.

    :rtype: (int, int, bool) or None
    :return: (frequency, size, stereo)
    '''
    if not SDL_WasInit(SDL_INIT_AUDIO):
        return
    opened, frequency, format, channels = Mix_QuerySpec()
    if format & ~0xff:
        format = -(format & 0xff)
    return frequency, format, channels > 1

def _mixer_init_check():
    if not SDL_WasInit(SDL_INIT_AUDIO):
        raise pygame.base.error, 'mixer system not initialized'

def stop():
    '''Stop playback of all sound channels.

    This will stop all playback of all active mixer channels.
    '''
    _mixer_init_check()
    Mix_HaltChannel(-1)

def pause():
    '''Temporarily stop playback of all sound channels.

    This will temporarily stop all playback on the active mixer channels.
    The playback can later be resumed with pygame.mixer.unpause()
    '''
    _mixer_init_check()
    Mix_Pause(-1)

def unpause():
    '''Resume paused playback of sound channels.

    This will resume all active sound channels after they have been paused.
    '''
    _mixer_init_check()
    Mix_Resume(-1)

def fadeout(time):
    '''Fade out the volume on all sounds before stopping.

    This will fade out the volume on all active channels over the time 
    argument in milliseconds. After the sound is muted the playback will stop.

    :Parameters:
        `time` : int
            Time to fade out, in milliseconds.

    '''
    _mixer_init_check()
    Mix_FadOutChannel(-1, time)

def set_num_channels(channels):
    '''Set the total number of playback channels.

    Sets the number of available channels for the mixer. The default value is
    8. The value can be increased or decreased. If the value is decreased,
    sounds playing on the truncated channels are stopped.

    :Parameters:
        `channels` : int
            Number of channels

    '''
    _mixer_init_check()
    Mix_AllocateChannels(channels)

    for i in _channels.keys()[:]:
        if i >= channels:
            del channels[i]

def get_num_channels():
    '''Get the total number of playback channels.

    Returns the number of currently active playback channels.

    :rtype: int
    '''
    _mixer_init_check()
    return Mix_GroupCount(-1)

def set_reserved(channels):
    '''Reserve channels from being automatically used.

    The mixer can reserve any number of channels that will not be automatically
    selected for playback by Sounds. If sounds are currently playing on the
    reserved channels they will not be stopped.

    This allows the application to reserve a specific number of channels for
    important sounds that must not be dropped or have a guaranteed channel to
    play on.

    :Parameters:
        `channels` : int
            Number of channels to reserve.
    ''' 
    _mixer_init_check()
    Mix_ReserveChannels(channels)

def find_channel(force=False):
    '''Find an unused channel.

    This will find and return an inactive Channel object. If there are no
    inactive Channels this function will return None. If there are no inactive
    channels and the force argument is True, this will find the Channel with
    the longest running Sound and return it.

    If the mixer has reserved channels from pygame.mixer.set_reserved() then
    those channels will not be returned here.

    :Parameters:
        `force` : bool
            If True, a playing channel will be returned if no free ones
            are available.

    :rtype: `Channel`
    '''
    _mixer_init_check()

    chan = Mix_GroupAvailable(-1)
    if chan == -1:
        if not force:
            return
        chan = Mix_GroupOldest(-1)
    return Channel(chan)

def get_busy():
    '''Test if any sound is being mixed.

    Returns True if the mixer is busy mixing any channels. If the mixer is
    idle then this return False.

    :rtype: bool
    '''
    if not SDL_WasInit(SDL_INIT_AUDIO):
        return False

    return bool(Mix_Playing(-1))

class Sound(object):
    '''The Sound object represents actual sound sample data. 
    
    Methods that change the state of the Sound object will the all instances
    of the Sound playback.
    '''

    __slots__ = ['_chunk']

    def __init__(self, file, _chunk=None):
        '''Create a new Sound object from a file.

        Load a new sound buffer from a filename or from a python file object.
        Limited resampling will be performed to help the sample match the
        initialize arguments for the mixer.

        The Sound can be loaded from an OGG audio file or from an uncompressed
        WAV.

        :Parameters:
            `file` : str or file-like object
                The filename or file to load.
            `_chunk` : None
                Internal use only.

        '''
        if _chunk:
            self._chunk = _chunk
            return

        _mixer_init_check()

        if hasattr(file, 'read'):
            rw = SDL_RWFromObject(file)
            # differ from Pygame, no freesrc here.
            self._chunk = Mix_LoadWAV_RW(rw, 0)
        else:
            self._chunk = Mix_LoadWAV(file)

    def __del__(self):
        if self._chunk:
            Mix_FreeChunk(self._chunk)

    def play(self, loops=0, maxtime=-1):
        '''Begin sound playback.

        Begin playback of the Sound (i.e., on the computer's speakers) on an
        available Channel. This will forcibly select a Channel, so playback
        may cut off a currently playing sound if necessary.

        The loops argument controls how many times the sample will be repeated
        after being played the first time. A value of 5 means that the sound
        will be played once, then repeated five times, and so is played a
        total of six times. The default value (zero) means the Sound is not
        repeated, and so is only played once. If loops is set to -1 the Sound
        will loop indefinitely (though you can still call stop() to stop it).

        The maxtime argument can be used to stop playback after a given number
        of milliseconds.

        :Parameters:
            `loops` : int
                Number of times to repeat the sound after the first play.
            `maxtime` : int
                Maximum number of milliseconds to play for.

        :rtype: `Channel`
        :return: The Channel object for the channel that was selected.
        '''
        channelnum = Mix_PlayChannelTimed(-1, self._chunk, loops, maxtime)
        if channelnum == -1:
            return

        Mix_Volume(channelnum, 128)
        Mix_GroupChannel(channelnum, id(self))

        channel = Channel(channelnum)
        channel._queue = None
        channel._sound = self
        return channel
        
    def stop(self):
        '''Stop sound playback.

        This will stop the playback of this Sound on any active Channels.
        '''        
        _mixer_init_check()
        Mix_HaltGroup(id(self))

    def fadeout(self, time):
        '''Stop sound playback after fading out.

        This will stop playback of the sound after fading it out over the 
        time argument in milliseconds. The Sound will fade and stop on all
        actively playing channels.
        
        :Parameters:
            `time` : int
                Time to fade out, in milliseconds.

        '''
        _mixer_init_check()
        Mix_FadeOutGroup(id(self), time)

    def set_volume(self, volume):
        '''Set the playback volume for this Sound.

        This will set the playback volume (loudness) for this Sound. This will
        immediately affect the Sound if it is playing. It will also affect any
        future playback of this Sound. The argument is a value from 0.0 to
        1.0.
        
        :Parameters:
            `volume` : float
                Volume of playback, in range [0.0, 1.0]

        '''
        _mixer_init_check()

        Mix_VolumeChunk(self._chunk, int(volume * 128))

    def get_volume(self):
        '''Get the playback volume.

        Return a value from 0.0 to 1.0 representing the volume for this Sound.

        :rtype: float
        ''' 
        _mixer_init_check()

        return Mix_VolumeChunk(self._chunk, -1) / 128.0

    def get_num_channels(self):
        '''Count how many times this Sound is playing.

        Return the number of active channels this sound is playing on.
        
        :rtype: int
        '''
        _mixer_init_check()

        return Mix_GroupCount(id(self))

    def get_length(self):
        '''Get the length of the Sound.

        Return the length of this Sound in seconds.
        
        :rtype: float 
        '''
        _mixer_init_check()

        opened, freq, format, channels = Mix_QuerySpec()
        if format == AUDIO_S8 or format == AUDIO_U8:
            mixerbytes = 1
        else:
            mixerbytes = 2
        numsamples = self._chunk.alen / mixerbytes / channels

        return numsamples / float(freq)

class Channel(object):
    '''The Channel object can be used to get fine control over the playback of
    Sounds. A channel can only playback a single Sound at time. Using channels
    is entirely optional since pygame can manage them by default.
    '''

    __slots__ = ['_id', '_sound', '_queue', '_endevent']

    def __new__(cls, id):
        _mixer_init_check()

        if id < 0 or id >= Mix_GroupCount(-1):
            raise IndexError, 'invalid channel index'

        if id in _channels:
            return _channels[id]

        inst = super(Channel, cls).__new__(cls, id)

        return inst

    def __init__(self, id):
        '''Create a Channel object for controlling playback.

        Create a Channel object for one of the current channels. The id must
        be a value from 0 to the value of pygame.mixer.get_num_channels().

        :Parameters:
            `id` : int
                ID of existing channel to create object for.
        '''
        self._id = id
        if id not in _channels:
            self._sound = None
            self._queue = None 
            self._endevent = SDL_NOEVENT
            _channels[id] = self

    def play(self, sound, loops=0, time=-1):
        '''Play a Sound on a specific Channel.

        This will begin playback of a Sound on a specific Channel. If the
        Channel is currently playing any other Sound it will be stopped.

        The loops argument has the same meaning as in Sound.play(): it is the
        number of times to repeat the sound after the first time. If it is 3,
        the sound will be played 4 times (the first time, then three more).
        If loops is -1 then the playback will repeat indefinitely.

        As in Sound.play(), the time argument can be used to
        stop playback of the Sound after a given number of milliseconds.

        :Parameters:
            `sound` : `Sound`
                Sound data to play.
            `loops` : int
                Number of times to repeat the sound after the first play.
            `maxtime` : int
                Maximum number of milliseconds to play for.

        '''
        channelnum = Mix_PlayChannelTimed(self._id, sound._chunk, loops, time)
        if channelnum != -1:
            Mix_GroupChannel(channelnum, id(sound))
        self._sound = sound
        self._queue = None

    def stop(self):
        '''Stop playback on a Channel.

        Stop sound playback on a channel. After playback is stopped the
        channel becomes available for new Sounds to play on it.
        '''
        _mixer_init_check()
        Mix_HaltChannel(self._id)

    def pause(self):
        '''Temporarily stop playback of a channel.

        Temporarily stop the playback of sound on a channel. It can be resumed
        at a later time with Channel.unpause()
        '''
        _mixer_init_check()
        Mix_Pause(self._id)

    def unpause(self):
        '''Resume pause playback of a channel.

        Resume the playback on a paused channel.
        '''
        _mixer_init_check()
        Mix_Resume(self._id)

    def fadeout(self, time):
        '''Stop playback after fading channel out.

        Stop playback of a channel after fading out the sound over the given
        time argument in milliseconds.

        :Parameters:
            `time` : int
                Time to fade out, in milliseconds.
        '''
        _mixer_init_check()
        Mix_FadeOutChannel(self._id, time)
                
    def set_volume(self, left, right=None):
        '''Set the volume of a playing channel.

        Set the volume (loudness) of a playing sound. When a channel starts to
        play its volume value is reset. This only affects the current sound.
        Each argument is between 0.0 and 1.0.

        If one argument is passed, it will be the volume of both speakers.
        If two arguments are passed and the mixer is in stereo mode, the
        first argument will be the volume of the left speaker and the second
        will be the volume of the right speaker. (If the second argument is
        None, the first argument will be the volume of both speakers.)

        If the channel is playing a Sound on which set_volume() has also
        been called, both calls are taken into account. For example::

            sound = pygame.mixer.Sound("s.wav")
            channel = s.play()      # Sound plays at full volume by default
            sound.set_volume(0.9)   # Now plays at 90% of full volume.
            sound.set_volume(0.6)   # Now plays at 60% (previous value replaced)
            channel.set_volume(0.5) # Now plays at 30% (0.6 * 0.5).
        
        :Parameters:
            `left` : float
                Volume of left (or mono) channel, in range [0.0, 1.0]
            `right` : float
                Volume of right channel, in range [0.0, 1.0]

        '''
        _mixer_init_check()
        if Mix_Linked_Version().is_since((1,2,1)):
            if right is None:
                Mix_SetPanning(self._id, 255, 255)
            else:
                Mix_SetPanning(self._id, int(left * 255), int(right * 255))
                left = 1.0
        else:
            if right is not None:
                left = (left + right) / 2
        
        Mix_Volume(self._id, int(left * 128))


    def get_volume(self):
        '''Get the volume of the playing channel.

        Return the volume of the channel for the current playing sound. This
        does not take into account stereo separation used by
        Channel.set_volume.  The Sound object also has its own volume which is
        mixed with the channel.

        :rtype: float
        '''
        _mixer_init_check()

        return Mix_Volume(self._id, -1) / 128.0

    def get_busy(self):
        '''Determine if the channel is active.

        Returns true if the channel is activily mixing sound. If the channel
        is idle this returns False.

        :rtype: bool
        '''
        _mixer_init_check()
        return Mix_Playing(self._id)

    def get_sound(self):
        '''Get the currently playing Sound.

        Return the actual Sound object currently playing on this channel. If
        the channel is idle None is returned.

        :rtype: `Sound`
        '''
        return self._sound

    def queue(self, sound):
        '''Queue a Sound object to follow the current.

        When a Sound is queued on a Channel, it will begin playing immediately
        after the current Sound is finished. Each channel can only have a
        single Sound queued at a time. The queued Sound will only play if the
        current playback finished automatically. It is cleared on any other
        call to Channel.stop() or Channel.play().

        If there is no sound actively playing on the Channel then the Sound
        will begin playing immediately.

        :Parameters:
            `sound` : `Sound`
                Sound data to queue.

        '''
        if not self._sound:
            channelnum = Mix_PlayChannelTimed(self._id, sound._chunk, 0, -1)
            if channelnum != -1:
                Mix_GroupChannel(channelnum, id(sound))
            self._sound = sound
        else:
            self._queue = sound

    def get_queue(self):
        '''Return any Sound that is queued.

        If a Sound is already queued on this channel it will be returned. Once
        the queued sound begins playback it will no longer be on the queue.
    
        :rtype: `Sound`
        '''
        return self._queue

    def set_endevent(self, id=None):
        '''Have the channel send an event when playback stops.

        When an endevent is set for a channel, it will send an event to the
        pygame queue every time a sound finishes playing on that channel (not
        just the first time). Use pygame.event.get() to retrieve the endevent
        once it's sent.

        Note that if you called Sound.play(n) or Channel.play(sound,n), the
        end event is sent only once: after the sound has been played "n+1"
        times (see the documentation of Sound.play).

        If Channel.stop() or Channel.play() is called while the sound was
        still playing, the event will be posted immediately.

        The `id` argument will be the event id sent to the queue. This can be
        any valid event type, but a good choice would be a value between
        pygame.locals.USEREVENT and pygame.locals.NUMEVENTS.  If no type
        argument is given then the Channel will stop sending endevents.

        :Parameters:
            `id` : int
                Event ID to send.

        '''
        if id is None:
            id = SDL_NOEVENT

        self._endevent = id

    def get_endevent(self):
        '''Get the event a channel sends when playback stops.

        Returns the event type to be sent every time the Channel finishes
        playback of a Sound. If there is no endevent the function returns
        pygame.NOEVENT.

        :rtype: int
        '''
        return self._endevent
