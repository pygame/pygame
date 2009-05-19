import sys
import pygame2

try:
    import pygame2.sdl.constants as sdlconst
    import pygame2.sdl.cdrom as cdrom
except ImportError:
    print ("No pygame2.sdl support")
    sys.exit ()

def run ():
    cdrom.init ()
    print ("Number of CDROM drives found: %d" % cdrom.num_drives ())
    for index in range (cdrom.num_drives ()):
        # Loop over all drives to get one with a CD inside
        cd = cdrom.CD (index)
        print ("CD system name: %s" % cd.name)
        status = "Empty"
        if status == sdlconst.CD_ERROR:
            status = "Error"
        elif status == sdlconst.CD_PAUSED:
            status = "Paused"
        elif status == sdlconst.CD_PLAYING:
            status = "Playing"
        elif status == sdlconst.CD_STOPPED:
            status = "Stopped"

        print ("CD status:      %s" % status)
        print ("CD tracks:      %d" % cd.num_tracks)
        if cd.status == sdlconst.CD_TRAYEMPTY:
            continue
        print ("Getting track information for CD %s..." % cd.name)
        for track in cd.tracks:
            print ("----------------------------------")
            print ("CD track id:      %d" % track.id)
            print ("CD track length:  %d" % track.length)
            print ("CD track minutes: %d" % track.minutes)
            print ("CD track seconds: %d" % track.seconds)
            print ("CD track time:    %d:%d" % track.time)
            ttype = "Audio"
            if track.type == sdlconst.DATA_TRACK:
                ttype = "Data"
            print ("CD track type:    %s" % ttype)
    cdrom.quit ()

if __name__ == "__main__":
    run ()
