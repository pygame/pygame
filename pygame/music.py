#!/usr/bin/env python

'''Pygame module for controlling streamed audio

The music module is closely tied to pygame.mixer. Use the music module to
control the playback of music in the sound mixer.

The difference between the music playback and regular Sound playback is that
the music is streamed, and never actually loaded all at once. The mixer system
only supports a single music stream at once.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from SDL import *
from SDL.mixer import *

import pygame.base
import pygame.mixer

_current_music = None
_queue_music = None

_frequency = 0
_format = 0
_channels = 0
_pos = 0
_pos_time = -1

_endmusic_event = SDL_NOEVENT

def _mixmusic_callback(data, stream):
    global _pos, _pos_time
    if not Mix_PausedMusic():
        _pos += len(stream)
        _pos_time = SDL_GetTicks()

def _endmusic_callback():
    global _current_music, _queue_music, _pos, _pos_time

    if _endmusic_event:
        _free_loaded(True, False)
        _current_music = _queue_music
        _queue_music = None
        Mix_HookMusicFinished(_endmusic_callback)
        _pos = 0
        Mix_PlayMusic(_current_music, 0)
    else:
        _pos_time = -1
        Mix_SetPostMix(None, None)

def _free_loaded(current=True, queue=True):
    global _current_music, _queue_music
    if current and _current_music:
        Mix_FreeMusic(_current_music)
        _current_music = None

    if queue and _queue_music:
        Mix_FreeMusic(_queue_music)
        _current_music = None 

def load(filename):
    '''Load a music file for playback.

    This will load a music file and prepare it for playback. If a music stream
    is already playing it will be stopped. This does not start the music
    playing.

    Music can only be loaded from filenames, not python file objects like the
    other pygame loading functions.
    
    :Parameters:
        `filename` : str
            Filename of music to load.

    '''
    global _current_music
    pygame.mixer._mixer_init_check()

    _free_loaded()
    try:
        _current_music = Mix_LoadMUS(filename)
    except SDL.SDL_Exception, e:
        raise pygame.base.error(str(e))

def play(loops=0, start=0.0):
    '''Start the playback of the music stream.

    This will play the loaded music stream. If the music is already playing it
    will be restarted.

    The `loops` argument controls the number of repeats a music will play.
    play(5) will cause the music to played once, then repeated five times, for
    a total of six. If `loops` is -1 then the music will repeat until stopped.

    The `start` argument controls where in the music the song starts playing.
    The starting position is dependent on the format of music playing.
    MP3 and OGG use the position as time (in seconds). MOD music it is the
    pattern order number. Passing a value to `start` will raise a
    NotImplementedError if it cannot set the start position

    :Parameters:
        `loops` : int
            Number of times to repeat music after initial play through.
        `start` : float
            Starting time within music track to play from, in seconds.

    '''    
    global _frequency, _format, _channels

    pygame.mixer._mixer_init_check()

    if not _current_music:
        raise pygame.base.error, 'music not loaded'

    Mix_HookMusicFinished(_endmusic_callback)
    Mix_SetPostMix(_mixmusic_callback, None)
    ready, _frequency, _format, _channels = Mix_QuerySpec()

    if Mix_Linked_Version().is_since((1, 2, 3)):
        volume = Mix_VolumeMusic(-1)
        Mix_FadeInMusicPos(_current_music, loops, 0, start)
        Mix_VolumeMusic(volume)
    else:
        if start:
            raise NotImplementedError, \
                'music start position requires SDL_Mixer 1.2.3 or later'
        Mix_PlayMusic(_current_music, loops)

def rewind():
    '''Restart music.

    Resets playback of the current music to the beginning.
    '''    
    pygame.mixer._mixer_init_check()
    Mix_RewindMusic()

def stop():
    '''Stop the music playback.

    Stops the current music if it is playing.  Any queued music will be
    unqueued.
    '''    
    pygame.mixer._mixer_init_check()
    Mix_HaltMusic()
    _free_loaded(False, True)

def pause():
    '''Temporarily stop music playback.

    Temporarily stop playback of the music stream. It can be resumed
    with the `unpause` function.
    '''
    pygame.mixer._mixer_init_check()
    Mix_PauseMusic()

def unpause():
    '''Resume paused music.

    This will resume the playback of a music stream after it has been paused.
    '''    
    pygame.mixer._mixer_init_check()
    Mix_ResumeMusic()

def fadeout(time):
    '''Stop music playback after fading out.

    This will stop the music playback after it has been faded out over the
    specified time (measured in milliseconds).  Any queued music will be
    unqueued.

    Note, that this function blocks until the music has faded out.
    
    :Parameters:
        `time` : int
            Time to fade out over, in milliseconds.

    '''
    pygame.mixer._mixer_init_check()

    Mix_FadeOutMusic(time)
    _free_loaded(False, True)

def set_volume(volume):
    '''Set the music volume.

    Set the volume of the music playback. The value argument is between
    0.0 and 1.0. When new music is loaded the volume is reset.
    
    :Parameters:
        `volume` : float
            Volume of music playback, in range [0.0, 1.0].

    '''
    pygame.mixer._mixer_init_check()
    Mix_VolumeMusic(int(volume * 128))

def get_volume():
    '''Get the music volume.

    Returns the current volume for the mixer. The value will be between 0.0
    and 1.0.
    
    :rtype: float
    '''
    pygame.mixer._mixer_init_check()
    return Mix_VolumeMusic(-1) / 128.0

def get_busy():
    '''Check if the music stream is playing.

    Returns True when the music stream is actively playing. When the music
    is idle this returns False.
    
    :rtype: bool
    '''
    pygame.mixer._mixer_init_check()
    return Mix_PlayingMusic()

def get_pos():
    '''Get the amount of time music has been playing.

    This gets the number of milliseconds that the music has been playing for.
    The returned time only represents how long the music has been playing; it
    does not take into account any starting position offsets.

    Returns -1 if the position is unknown.
    
    :rtype: int
    '''
    pygame.mixer._mixer_init_check()
    if _pos_time < 0:
        return -1

    ticks = 1000 * _pos / _channels / _frequency / ((_format & 0xff) >> 3)
    if not Mix_PausedMusic():
        ticks += SDL_GetTicks() - _pos_time
    return int(ticks)

def queue(filename):
    '''Queue a music file to follow the current one.

    This will load a music file and queue it. A queued music file will begin
    as soon as the current music naturally ends. If the current music is ever
    stopped or changed, the queued song will be lost.

    The following example will play music by Bach six times, then play
    music by Mozart once::

        pygame.mixer.music.load('bach.ogg')
        pygame.mixer.music.play(5)        # Plays six times, not five
        pygame.mixer.music.queue('mozart.ogg')
    
    :Parameters:
        `filename` : str
            Filename of music file to queue.

    '''
    global _queue_music

    pygame.mixer._mixer_init_check()

    music = Mix_LoadMUS(filename)
    _free_loaded(False, True)

    _queue_music = music

def set_endevent(eventtype=None):
    '''Have the music send an event when playback stops.

    This causes Pygame to signal (by means of the event queue) when
    the music is done playing. The argument determines the type of
    event that will be queued.

    The event will be queued every time the music finishes, not just
    the first time. To stop the event from being queued, call this
    method with no argument.
    
    :Parameters:
        `eventtype` : int
            Type of event to post.  For example, ``SDL_USEREVENT + n``

    '''
    global _endmusic_event
    
    if eventtype is None:
        eventtype = SDL_NOEVENT
    _endmusic_event = eventtype

def get_endevent():
    '''Get the event a channel sends when playback stops.

    Returns the event type to be sent every time the music finishes playback.
    If there is no endevent the function returns pygame.NOEVENT.
    
    :rtype: int 
    '''
    return _endmusic_event
