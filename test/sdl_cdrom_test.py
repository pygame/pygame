import sys
try:
    import pygame2.test.pgunittest as unittest
    from pygame2.test.pgunittest import doprint, interactive
except:
    import pgunittest as unittest
    from pgunittest import doprint, interactive

import pygame2
import pygame2.sdl.cdrom as cdrom
import pygame2.sdl.constants as constants

class SDLCDRomTest (unittest.TestCase):

    def todo_test_pygame2_sdl_cdrom_CD_close(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.close:

        # close () -> None
        # 
        # Releases the CD internals. Useful for e.g. switching CDs within
        # the drive without the need to recreate the CD object. open
        # will reinitialize the CD internals. You should not use any other
        # method or attribute until a call to open.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CD_cur_frame(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.cur_frame:

        # The current frame offset within the curent track.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CD_cur_track(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.cur_track:

        # The current track.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CD_eject(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.eject:

        # eject () -> None
        # 
        # Ejects the CD or DVD.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CD_index(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.index:

        # The drive index as specified in the constructor.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CD_name(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.name:

        # The system-dependent drive name (e.g. "/dev/cdrom" or "D:").

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CD_num_tracks(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.num_tracks:

        # The total number of tracks on the CD or DVD.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CD_open(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.open:

        # open () -> None
        # 
        # (Re-)Opens the CD and initialises the CD internals
        # after a close call.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CD_pause(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.pause:

        # pause () -> None
        # 
        # Pauses the actual CD playback.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CD_play(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.play:

        # play (start, length[, asfps]) -> None
        # 
        # Starts playing the current CD beginning at the give *start*
        # time for a maximum of *length* seconds. The *start* and *length*
        # arguments are handled as seconds by default. To use an exact frame
        # offset instead ofseconds, pass True as third *asfps* parameter.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CD_play_tracks(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.play_tracks:

        # play_tracks ([starttrack, ntracks, start, length, asfps]) -> None
        # 
        # Plays a certain number of tracks beginning at the passed start
        # track. If *start* and *length* are not 0, *start* determines the
        # offset of *starttrack* to begin the playback at and *length*
        # specifies the amount of seconds to play from the last track
        # within the track list. To use an exact frame offset instead of
        # seconds for the *start* and *length* parameters, pass True as *asfps*
        # parameter.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CD_resume(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.resume:

        # resume () -> None
        # 
        # Resumes a previously paused playback.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CD_status(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.status:

        # Gets the current CD status.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CD_stop(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.stop:

        # stop () -> None
        # 
        # Stops the current playback.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CD_tracks(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CD.tracks:

        # Gets a list of CDTrack objects with the CD track
        # information.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CDTrack_id(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CDTrack.id:

        # Gets the CD track id.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CDTrack_length(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CDTrack.length:

        # Gets the track length in frames.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CDTrack_minutes(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CDTrack.minutes:

        # Gets the approximate track length in minutes.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CDTrack_offset(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CDTrack.offset:

        # Gets the frame offset of the track on the CD.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CDTrack_seconds(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CDTrack.seconds:

        # Gets the approximate track length in seconds.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CDTrack_time(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CDTrack.time:

        # Gets the approximate track length in minutes and seconds as
        # tuple.

        self.fail() 

    def todo_test_pygame2_sdl_cdrom_CDTrack_type(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.CDTrack.type:

        # Gets the track type (data or audio).

        self.fail() 

    @interactive ("Are the shown CD/DVD drive names correct?")
    def test_pygame2_sdl_cdrom_get_name(self):
        #
        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.get_name:

        # get_name (index) -> str
        # 
        # Gets the name of the specified CD- or DVD-ROM drive.
        # 
        # Gets the system-dependent drive name (e.g. "/dev/cdrom" or "D:")
        # for the CD- or DVD-Rom specified by the passed *index*.
        cdrom.init ()
        self.assertRaises (ValueError, cdrom.get_name, -4)
        self.assertRaises (ValueError, cdrom.get_name, cdrom.num_drives ())
        
        for i in range (cdrom.num_drives ()):
            doprint ("Drive %d: %s" % (i, cdrom.get_name (i)))
        cdrom.quit ()

    def test_pygame2_sdl_cdrom_init(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.init:

        # init () -> None
        # 
        # Initializes the CD-ROM subsystem of the SDL library.
        self.assertEqual (cdrom.init (), None)
        self.assertTrue (cdrom.was_init ())
        self.assertEqual (cdrom.quit (), None)
        self.assertFalse (cdrom.was_init ())

    @interactive ("Does the shown number match your CD and DVD drives?")
    def test_pygame2_sdl_cdrom_num_drives(self):
        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.num_drives:

        # num_drives () -> int
        # 
        # Gets the number of accessible CD- and DVD-ROM drives for the system.
        cdrom.init ()
        doprint ("Found CD/DVD drives: %d" % cdrom.num_drives ())
        cdrom.quit ()

    def test_pygame2_sdl_cdrom_quit(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.quit:

        # quit () -> None
        # 
        # Shuts down the CD-ROM subsystem of the SDL library.
        # 
        # After calling this function, you should not invoke any class,
        # method or function related to the CD-ROM subsystem as they are
        # likely to fail or might give unpredictable results.
        self.assertEqual (cdrom.quit (), None)

    def test_pygame2_sdl_cdrom_was_init(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.cdrom.was_init:

        # was_init () -> bool
        # 
        # Returns, whether the CD-ROM subsystem of the SDL library is initialized.
        self.assertFalse (cdrom.was_init ())
        self.assertEqual (cdrom.init (), None)
        self.assertTrue (cdrom.was_init ())
        self.assertEqual (cdrom.quit (), None)
        self.assertFalse (cdrom.was_init ())

if __name__ == "__main__":
    unittest.main ()

