#!/usr/bin/env python

'''Test the SDL CD-ROM audio functions.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys

from SDL import *

def PrintStatus(driveindex, cdrom):
    status = SDL_CDStatus(cdrom)
    if status == CD_TRAYEMPTY:
        status_str = 'tray empty'
    elif status == CD_STOPPED:
        status_str = 'stopped'
    elif status == CD_PLAYING:
        status_str = 'playing'
    elif status == CD_PAUSED:
        status_str = 'paused'
    elif status == CD_ERROR:
        status_str = 'error state'

    print 'Drive %d status: %s' % (driveindex, status_str)
    if status >= CD_PLAYING:
        m, s, f = FRAMES_TO_MSF(cdrom.cur_frame)
        print 'Currently playing track %d, %d:%2.2d' % \
            (cdrom.track[cdrom.cur_track].id, m, s)

def ListTracks(cdrom):
    SDL_CDStatus(cdrom)
    print 'Drive tracks: %d' % cdrom.numtracks
    for i in range(cdrom.numtracks):
        m, s, f = FRAMES_TO_MSF(cdrom.track[i].length)
        if f > 0:
            s += 1
        if cdrom.track[i].type == SDL_AUDIO_TRACK:
            trtype = 'audio'
        elif cdrom.track[i].type == SDL_DATA_TRACK:
            trtype = 'data'
        else:
            trtype = 'unknown'
        print 'Track (index %d) %d: %d:%2.2d / %d [%s track]' % \
            (i, cdrom.track[i].id, m, s, cdrom.track[i].length, trtype)

def PrintUsage():
    print >> sys.stderr, '''Usage %s [drive#] [command] [command] ...
Where 'command' is one of:
    -status
    -list
    -play [first_track] [first_frame] [num_tracks] [num_frames]
    -pause
    -resume
    -stop
    -eject
    -sleep <milliseconds>''' % sys.argv[0]

def shift(lst):
    if len(lst):
        return None
    v = lst[0]
    del lst[0]
    return v

if __name__ == '__main__':
    SDL_Init(SDL_INIT_CDROM)

    if SDL_CDNumDrives() == 0:
        print 'No CD-ROM devices detected'
        SDL_Quit()
        sys.exit(0)

    print 'Drives available: %d' % SDL_CDNumDrives()
    for i in range(SDL_CDNumDrives()):
        print 'Drive %d:  "%s"' % (i, SDL_CDName(i))

    drive = 0
    i = 1
    try:
        drive = int(sys.argv[1])
        i += 1
    except:
        pass
    cdrom = SDL_CDOpen(drive)

    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '-status':
            PrintStatus(drive, cdrom)  # not sure why this was commented out
        elif arg == '-list':
            ListTracks(cdrom)
        elif arg == '-play':
            strack = sframe = ntrack = nframe = 0
            try:
                strack = int(sys.argv[i + 1])
                i += 1
                sframe = int(sys.argv[i + 1])
                i += 1
                ntrack = int(sys.argv[i + 1])
                i += 1
                nframe = int(sys.argv[i + 1])
                i += 1
            except:
                pass
            if CD_INDRIVE(SDL_CDStatus(cdrom)):
                SDL_CDPlayTracks(cdrom, strack, sframe, ntrack, nframe)
            else:
                print >> sys.stderr, 'No CD in drive!'
        elif arg == '-pause':
            SDL_CDPause(cdrom)
        elif arg == '-resume':
            SDL_CDResume(cdrom)
        elif arg == '-stop':
            SDL_CDStop(cdrom)
        elif arg == '-eject':
            SDL_CDEject(cdrom)
        elif arg == '-sleep':
            SDL_Delay(int(sys.argv[i + 1]))
            print 'Delayed %d milliseconds' % sys.argv[i + 1]
            i += 1
        else:
            PrintUsage()
            SDL_CDClose(cdrom)
            SDL_Quit()
            sys.exit(1)
    PrintStatus(drive, cdrom)
    SDL_CDClose(cdrom)
    SDL_Quit()
