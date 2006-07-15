#!/usr/bin/env python

'''Pygame module for audio cdrom control.

The cdrom module manages the CD and DVD drives on a computer. It can
also control the playback of audio cd's. This module needs to be initialized
before it can do anything. Each CD object you create represents a cdrom drive
and must also be initialized individually before it can do most things.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from SDL import *

import pygame.base

_cdroms = []

def __PYGAMEinit__():
    if not SDL_WasInit(SDL_INIT_CDROM):
        SDL_InitSubSystem(SDL_INIT_CDROM)
        pygame.base.register_quit(_cdrom_autoquit)
    return 1

def _cdrom_autoquit():
    global _cdroms
    for cdrom in _cdroms:
        SDL_CDClose(cdrom)
    _cdroms = []

    if SDL_WasInit(SDL_INIT_CDROM):
        SDL_QuitSubSystem(SDL_INIT_CDROM)

def _cdrom_init_check():
    if not SDL_WasInit(SDL_INIT_CDROM):
        raise pygame.base.error, 'cdrom system not initialized'

def init():
    '''Initialize the cdrom module.

    Initialize the cdrom module. This will scan the system for all CD devices.
    The module must be initialized before any other functions will work. This
    automatically happens when you call pygame.init().

    It is safe to call this function more than once.
    '''
    __PYGAMEinit__()

def quit():
    '''Uninitialize the cdrom module.

    Uninitialize the cdrom module. After you call this any existing CD objects
    will no longer work.

    It is safe to call this function more than once.
    '''
    _cdrom_autoquit()

def get_init():
    '''Determine if the cdrom module is initialized.

    Test if the cdrom module is initialized or not. This is different than the
    CD.init() since each drive must also be initialized individually.

    :rtype: bool
    '''
    return SDL_WasInit(SDL_INIT_CDROM) != 0

def get_count():
    '''Number of cd drives on the system.

    Return the number of cd drives on the system. When you create CD objects
    you need to pass an integer id that must be lower than this count. The
    count will be 0 if there are no drives on the system.

    :rtype: int
    '''
    _cdrom_init_check()
    
    return SDL_CDNumDrives()

class CD(object):
    '''Class to manage a cdrom drive.

    :see: `__init__`
    '''

    __slots__ = ['_id', '_device']

    def __init__(self, id):
        '''Create a CD object.

        You can create a CD object for each cdrom on the system. Use
        pygame.cdrom.get_count() to determine how many drives actually exist.
        The id argument is an integer of the drive, starting at zero.

        The CD object is not initialized, you can only call CD.get_id() and
        CD.get_name() on an uninitialized drive.

        It is safe to create multiple CD objects for the same drive, they will
        all cooperate normally.

        :Parameters:
            `id` : int
                Device ID
        '''
        _cdrom_init_check()

        if id < 0 or id >= SDL_CDNumDrives():
            raise pygame.base.error, 'Invalid cdrom device number'

        self._id = id
        self._device = None

    def _init_check(self):
        _cdrom_init_check()
        if not self._device:
            raise pygame.base.error, 'CD drive not initialized'

    def init(self):
        '''Initialize a cdrom drive for use.

        Initialize the cdrom drive for use. The drive must be initialized for
        most CD methods to work.  Even if the rest of pygame has been
        initialized.

        There may be a brief pause while the drive is initialized. Avoid
        CD.init() if the program should not stop for a second or two.
        '''
        _cdrom_init_check()

        if not self._device:
            self._device = SDL_CDOpen(self._id)
            _cdroms.append(self._device)

    def quit(self):
        '''Uninitialize a cdrom drive for use.

        Uninitialize a drive for use. Call this when your program will not be
        accessing the drive for awhile.
        '''
        _cdrom_init_check()

        if self._device:
            SDL_CDClose(self._device)
            _cdroms.remove(self._device)
            self._device = None

    def get_init(self):
        '''Deterimine if this cd device initialized.

        Test if this CDROM device is initialized. This is different than the
        pygame.cdrom.init() since each drive must also be initialized
        individually.

        :rtype: bool
        '''
        return self._device is not None

    def play(self, track, start=0.0, end=0.0):
        '''Start playing audio.

        Playback audio from an audio cdrom in the drive. Besides the track
        number argument, you can also pass a starting and ending time for
        playback. The start and end time are in seconds, and can limit the
        section of an audio track played.

        If you pass a start time but no end, the audio will play to the end of
        the track. If you pass a start time and 'None' for the end time, the
        audio will play to the end of the entire disc.

        See the CD.get_numtracks() and CD.get_track_audio() to find tracks to
        playback.

        Note, track 0 is track 1 on the CD.  Track numbers start at zero.
        
        :Parameters:
            `track` : int
                Track number to start from.
            `start` : float
                Starting time offset, in seconds.
            `end` : float
                End time, in seconds.

        '''
        self._init_check()

        SDL_CDStatus(self._device)
        if track < 0 or track >= self._device.numtracks:
            raise IndexError, 'Invalid track number'

        trackdata = self._device.track[track]
        if trackdata.type != SDL_AUDIO_TRACK:
            raise pygame.base.error, 'CD track type is not audio'

        if start == end and start != 0.0:
            # Zero-length playtime
            return

        startframe = int(max(start * CD_FPS, 0))
        if end is None:
            # Entire disc
            numframes = 0
        elif end == 0.0:
            # Remainder of track
            numframes = trackdata.length - startframe
        else:
            # To end time
            numframes = int((end - start) * CD_FPS)

        if numframes < 0 or startframe > trackdata.length * CD_FPS:
            # Outside track time
            return
        
        SDL_CDPlayTracks(self._device, track, startframe, 0, numframes)

    def stop(self):
        '''Stop audio playback.

        Stops playback of audio from the cdrom. This will also lose the
        current playback position. This method does nothing if the drive isn't
        already playing audio.
        '''
        self._init_check()

        SDL_CDStop(self._device)

    def pause(self):
        '''Temporarily stop audio playback.

        Temporarily stop audio playback on the CD. The playback can be resumed
        at the same point with the CD.resume() method. If the CD is not
        playing this method does nothing.
        '''
        self._init_check()

        SDL_CDPause(self._device)

    def resume(self):
        '''Unpause audio playback.

        Unpause a paused CD. If the CD is not paused or already playing, this
        method does nothing.
        '''
        self._init_check()

        SDL_CDResume(self._device)

    def eject(self):
        '''Eject or open the cdrom drive.

        This will open the cdrom drive and eject the cdrom. If the drive is
        playing or paused it will be stopped.
        '''
        self._init_check()

        SDL_CDEject(self._device)

    def get_id(self):
        '''Return the index of the cdrom drive.

        Returns the integer id that was used to create the CD instance. This
        method can work on an uninitialized CD.

        :rtype: int
        '''
        return self._id

    def get_name(self):
        '''Get the system name of the cdrom drive.

        Return the string name of the drive. This is the system name used to
        represent the drive. It is often the drive letter or device name. This
        method can work on an uninitialized CD.

        :rtype: str
        '''
        _cdrom_init_check()

        return SDL_CDName(self._id)

    def get_busy(self):
        '''Determine if the drive is playing audio.

        :rtype: bool
        '''
        self._init_check()

        return SDL_CDStatus(self._device) == CD_PLAYING

    def get_paused(self):
        '''Determine if the drive is paused.

        :rtype: bool
        '''
        self._init_check()

        return SDL_CDStatus(self._device) == CD_PAUSED

    def get_current(self):
        '''Retrieve the current audio playback position.

        Returns both the current track and time of that track. This method
        works when the drive is either playing or paused.

        Note, track 0 is track 1 on the CD.  Track numbers start at zero.

        :rtype: int, float
        :return: track, seconds
        '''
        self._init_check()
        SDL_CDStatus(self._device)
        track = self._device.cur_track
        seconds = self._device.cur_frame / float(CD_FPS)
        return track, seconds

    def get_empty(self):
        '''Determine if a cdrom is in the drive.

        Return True if there is a cdrom currently in the drive. If the drive
        is open this will also be False.

        :rtype: bool
        '''
        self._init_check()

        return SDL_CDStatus(self._device) == CD_TRAYEMPTY

    def get_numtracks(self):
        '''Get the number of tracks on the cdrom.

        Return the number of tracks on the cdrom in the drive. This will
        return zero of the drive is empty or has no tracks.

        :rtype: int
        '''
        self._init_check()
        
        SDL_CDStatus(self._device)
        return self._device.numtracks

    def get_track_audio(self):
        '''Determine if the cdrom track has audio data.

        Determine if a track on a cdrom contains audio data. You can also call
        CD.num_tracks() and CD.get_all() to determine more information about
        the cdrom.

        :rtype: bool
        '''
        self._init_check()

        SDL_CDStatus(self._device)
        if track < 0 or track >= self._device.numtracks:
            raise IndexError, 'Invalid track number'

        return self._device.track[track].type == SDL_AUDIO_TRACK

    def get_all(self):
        '''Get all track information.

        Return a list with information for every track on the cdrom. The
        information consists of a tuple with four values. The audio value is
        True if the track contains audio data. The start, end, and length
        values are floating point numbers in seconds. Start and end represent
        absolute times on the entire disc.

        :rtype: list of (bool, int, float, float, float)
        :return: each element is a tuple of (audio, start, end, length)
        '''
        self._init_check()
        
        tracks = []
        for track in range(self._device.numtracks):
            audio = self._device.track[track].type == SDL_AUDIO_TRACK
            start = self._device.track[track].offset / float(CD_FPS)
            length = self._device.track[track].length / float(CD_FPS)
            end = start + length
            tracks.append((audio, start, end, length))
        return tracks

    def get_track_start(self, track):
        '''Get the start time of a cdrom track.

        Return the absolute time in seconds where at start of the cdrom track.

        Note, track 0 is track 1 on the CD.  Track numbers start at zero.

        :Parameters:
            `track` : int
                Track index.

        :rtype: float
        '''
        self._init_check()

        SDL_CDStatus(self._device)
        if track < 0 or track >= self._device.numtracks:
            raise IndexError, 'Invalid track number'

        return self._device.track[track].offset / float(CD_FPS)

    def get_track_length(self, track):
        '''Get the length of a cdrom track.

        Return a floating point value in seconds of the length of the cdrom
        track.

        Note, track 0 is track 1 on the CD.  Track numbers start at zero.

        :Parameters:
            `track`: int
                Track index.

        :rtype: float
        '''
        self._init_check()

        SDL_CDStatus(self._device)
        if track < 0 or track >= self._device.numtracks:
            raise IndexError, 'Invalid track number'

        if self._device.track[track].type != SDL_AUDIO_TRACK:
            return 0.0

        return self._device.track[track].length / float(CD_FPS)
