#!/usr/bin/env python

'''CD-audio control.

In order to use these functions, `SDL_Init` must have been called with the
`SDL_INIT_CDROM` flag.  This causes SDL to scan the system for CD-ROM
drives, and load appropriate drivers.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.constants
import SDL.dll

class SDL_CDtrack(Structure):
    '''Structure describing a single CD track.

    :Ivariables:
        `id` : int
            Track number
        `type` : int
            One of SDL_AUDIO_TRACK or SDL_DATA_TRACK
        `length` : int
            Length, in frames, of this track
        `offset` : int
            Offset, in frames, from start of disk

    '''
    _fields_ = [('id', c_ubyte),
                ('type', c_ubyte),
                ('_unused', c_ushort),
                ('length', c_uint),
                ('offset', c_uint)]

class SDL_CD(Structure):
    '''Structure describing a CD.

    This structure is only current as of the last call to `SDL_CDStatus`.

    :Ivariables:
        `id` : int
            Private drive identifier
        `status` : int
            Current drive status.  One of CD_TRAYEMPTY, CD_STOPPED,
            CD_PLAYING, CD_PAUSED, CD_ERROR.
        `numtracks` : int
            Number of tracks on disk
        `cur_track` : int
            Current track position
        `cur_frame` : int
            Current frame offset within current track
        `track` : sequence of `SDL_CDtrack`
            Tracks on the disk.

    '''
    _fields_ = [('id', c_uint),
                ('status', c_int),
                ('numtracks', c_int),
                ('cur_track', c_int),
                ('cur_frame', c_int),
                ('track', SDL_CDtrack * (SDL.constants.SDL_MAX_TRACKS + 1))]

def CD_INDRIVE(status):
    '''Given a status, returns True if there's a disk in the drive.

    :Parameters:
     - `status`: int

    :rtype: bool
    '''
    return status > 0

def FRAMES_TO_MSF(frames):
    '''Convert from frames to minute/second/frame

    :Parameters:
     - `frames`: int

    :rtype: (int, int, int)
    :return: tuple of (minutes, seconds, frames)
    '''
    F = frames % SDL.constants.CD_FPS
    frames /= SDL.constants.CD_FPS
    S = frames % 60
    frames /= 60
    M = frames
    return (M, S, F)

def MSF_TO_FRAMES(minutes, seconds, frames):
    '''Convert from minute/second/frame to frames

    :Parameters:
     - `minutes`: int
     - `seconds`: int
     - `frames`: int

    :rtype: int
    '''
    return SDL.constants.CD_FPS(minutes * 60 + seconds) + frames

SDL_CDNumDrives = SDL.dll.function('SDL_CDNumDrives',
    '''Return the number of CD-ROM drives on the system.

    :rtype: int
    ''',
    args=[],
    arg_types=[],
    return_type=c_int,
    error_return=-1)

SDL_CDName = SDL.dll.function('SDL_CDName',
    '''Return a human-readable, system-dependent identifier for the
    CD-ROM.

    Example::

        '/dev/cdrom'
        'E:'
        '/dev/disk/ide/1/master'

    :Parameters:
        `drive` : int
            Drive number, starting with 0.  Drive 0 is the system default
            CD-ROM.

    :rtype: string
    ''',
    args=['drive'],
    arg_types=[c_int],
    return_type=c_char_p,
    require_return=True)

SDL_CDOpen = SDL.dll.function('SDL_CDOpen',
    '''Open a CD-ROM drive for access.

    It returns a drive handle on success, or raises an exception if the
    drive was invalid or busy.  This newly opened CD-ROM becomes the
    default CD used when other CD functions are passed None as the CD-ROM
    handle.

    Drives are numbered starting with 0.  Drive 0 is the system default
    CD-ROM.

    :Parameters:
     - `drive`: int

    :rtype: `SDL_CD`
    ''',
    args=['drive'],
    arg_types=[c_int],
    return_type=POINTER(SDL_CD),
    dereference_return=True,
    require_return=True)

SDL_CDStatus = SDL.dll.function('SDL_CDStatus',
    '''Return the current status of the given drive.

    If the drive has a CD in it, the table of contents of the CD and
    current play position of the CD will be updated in the SDL_CD
    instance.

    Possible return values are
     - `CD_TRAYEMPTY`
     - `CD_STOPPED`
     - `CD_PLAYING`
     - `CD_PAUSED`

    :Parameters:
     - `cdrom`: `SDL_CD`

    :rtype: int
    ''',
    args=['cdrom'],
    arg_types=[POINTER(SDL_CD)],
    return_type=int,
    error_return=-1)

SDL_CDPlayTracks = SDL.dll.function('SDL_CDPlayTracks',
    '''Play the given CD.

    Plays the given CD starting at `start_track` and `start_frame` for
    `ntracks` tracks and `nframes` frames.  If both `ntracks` and `nframes`
    are 0, play until the end of the CD.  This function will skip data
    tracks.  This function should only be called after calling
    `SDL_CDStatus` to get track information about the CD.

    For example::

    	# Play entire CD:
        if CD_INDRIVE(SDL_CDStatus(cdrom)):
            SDL_CDPlayTracks(cdrom, 0, 0, 0, 0)

        # Play last track:
        if CD_INDRIVE(SDL_CDStatus(cdrom)):
            SDL_CDPlayTracks(cdrom, cdrom.numtracks-1, 0, 0, 0)

        #Play first and second track and 10 seconds of third track:
        if CD_INDRIVE(SDL_CDStatus(cdrom)):
            SDL_CDPlayTracks(cdrom, 0, 0, 2, 10)

    :Parameters:
     - `cdrom`: `SDL_CD`
     - `start_track`: int
     - `start_frame`: int
     - `ntracks`: int
     - `nframes`: int

    ''',
    args=['cdrom', 'start_track', 'start_frame', 'ntracks', 'nframes'],
    arg_types=[POINTER(SDL_CD), c_int, c_int, c_int, c_int],
    return_type=c_int,
    error_return=-1)

SDL_CDPlay = SDL.dll.function('SDL_CDPlay',
    '''Play the given CD.

    Plays the given CD starting at `start` frame for `length` frames.

    :Parameters:
     - `cdrom`: `SDL_CD`
     - `start`: int
     - `length`: int

    ''',
    args=['cdrom', 'start', 'length'],
    arg_types=[POINTER(SDL_CD), c_int, c_int],
    return_type=c_int,
    error_return=-1)

SDL_CDPause = SDL.dll.function('SDL_CDPause',
    '''Pause play.

    :Parameters:
     - `cdrom`: `SDL_CD`

    ''',
    args=['cdrom'],
    arg_types=[POINTER(SDL_CD)],
    return_type=c_int,
    error_return=-1)

SDL_CDResume = SDL.dll.function('SDL_CDResume',
    '''Resume play.

    :Parameters:
     - `cdrom`: `SDL_CD`

    ''',
    args=['cdrom'],
    arg_types=[POINTER(SDL_CD)],
    return_type=c_int,
    error_return=-1)

SDL_CDStop = SDL.dll.function('SDL_CDStop',
    '''Stop play.

    :Parameters:
     - `cdrom`: `SDL_CD`

    ''',
    args=['cdrom'],
    arg_types=[POINTER(SDL_CD)],
    return_type=c_int,
    error_return=-1)

SDL_CDEject = SDL.dll.function('SDL_CDEject',
    '''Eject CD-ROM.

    :Parameters:
     - `cdrom`: `SDL_CD`

    ''',
    args=['cdrom'],
    arg_types=[POINTER(SDL_CD)],
    return_type=c_int,
    error_return=-1)

SDL_CDClose = SDL.dll.function('SDL_CDClose',
    '''Close the handle for the CD-ROM drive.

    :Parameters:
     - `cdrom`: `SDL_CD`

    ''',
    args=['cdrom'],
    arg_types=[POINTER(SDL_CD)],
    return_type=c_int,
    error_return=-1)
