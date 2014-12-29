#################################### IMPORTS ###################################

if __name__ == '__main__':
    import sys
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

if is_pygame_pkg:
    from pygame.tests.test_utils \
         import test_not_implemented, question, prompt, unittest
else:
    from test.test_utils \
         import test_not_implemented, question, prompt, unittest

import pygame

################################################################################

class CdromModuleTest(unittest.TestCase):
    def todo_test_CD(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD:

          # pygame.cdrom.CD(id): return CD
          # class to manage a cdrom drive
          # 
          # You can create a CD object for each cdrom on the system. Use
          # pygame.cdrom.get_count() to determine how many drives actually
          # exist. The id argument is an integer of the drive, starting at zero.
          # 
          # The CD object is not initialized, you can only call CD.get_id() and
          # CD.get_name() on an uninitialized drive.
          # 
          # It is safe to create multiple CD objects for the same drive, they
          # will all cooperate normally.
          # 

        self.fail() 

    def todo_test_get_count(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.get_count:

          # pygame.cdrom.get_count(): return count
          # number of cd drives on the system
          # 
          # Return the number of cd drives on the system. When you create CD
          # objects you need to pass an integer id that must be lower than this
          # count. The count will be 0 if there are no drives on the system.
          # 

        self.fail() 

    def todo_test_get_init(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.get_init:

          # pygame.cdrom.get_init(): return bool
          # true if the cdrom module is initialized
          # 
          # Test if the cdrom module is initialized or not. This is different
          # than the CD.init() since each drive must also be initialized
          # individually.
          # 

        self.fail() 

    def todo_test_init(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.init:

          # pygame.cdrom.init(): return None
          # initialize the cdrom module
          # 
          # Initialize the cdrom module. This will scan the system for all CD
          # devices. The module must be initialized before any other functions
          # will work. This automatically happens when you call pygame.init().
          # 
          # It is safe to call this function more than once. 

        self.fail() 

    def todo_test_quit(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.quit:

          # pygame.cdrom.quit(): return None
          # uninitialize the cdrom module
          # 
          # Uninitialize the cdrom module. After you call this any existing CD
          # objects will no longer work.
          # 
          # It is safe to call this function more than once. 

        self.fail() 

class CDTypeTest(unittest.TestCase):
    def setUp(self):
        pygame.cdrom.init()

        #TODO:
        try:
            self.cd = pygame.cdrom.CD(0)
        except pygame.error:
            self.cd = None

    def tearDown(self):
        pygame.cdrom.quit()

    def test_1_eject(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.eject:

          # CD.eject(): return None
          # eject or open the cdrom drive
        
        # should raise if cd object not initialized
        if self.cd:
            self.cd.init()
            self.cd.eject()

            self.assert_(question('Did the cd eject?'))
    
            prompt("Please close the cd drive")

    def test_2_get_name(self):

        # __doc__ (as of 2008-07-02) for pygame.cdrom.CD.get_name:

          # CD.get_name(): return name
          # the system name of the cdrom drive

        if self.cd:
            cd_name = self.cd.get_name()
    
            self.assert_ (
                question('Is %s the correct name for the cd drive?' % cd_name)
            )

    def todo_test_get_all(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_all:

          # CD.get_all(): return [(audio, start, end, lenth), ...]
          # get all track information
          # 
          # Return a list with information for every track on the cdrom. The
          # information consists of a tuple with four values. The audio value is
          # True if the track contains audio data. The start, end, and length
          # values are floating point numbers in seconds. Start and end
          # represent absolute times on the entire disc.
          # 

        self.fail() 

    def todo_test_get_busy(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_busy:

          # CD.get_busy(): return bool
          # true if the drive is playing audio
          # 
          # Returns True if the drive busy playing back audio. 

        self.fail() 

    def todo_test_get_current(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_current:

          # CD.get_current(): return track, seconds
          # the current audio playback position
          # 
          # Returns both the current track and time of that track. This method
          # works when the drive is either playing or paused.
          # 
          # Note, track 0 is the first track on the CD.  Track numbers start at zero. 

        self.fail() 

    def todo_test_get_empty(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_empty:

          # CD.get_empty(): return bool
          # False if a cdrom is in the drive
          # 
          # Return False if there is a cdrom currently in the drive. If the
          # drive is empty this will return True.
          # 

        self.fail() 

    def todo_test_get_id(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_id:

          # CD.get_init(): return bool
          # true if this cd device initialized
          # 
          # Returns the integer id that was used to create the CD instance. This
          # method can work on an uninitialized CD.
          # 

        self.fail() 

    def todo_test_get_init(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_init:

          # CD.get_init(): return bool
          # true if this cd device initialized
          # 
          # Test if this CDROM device is initialized. This is different than the
          # pygame.cdrom.init() since each drive must also be initialized
          # individually.
          # 

        self.fail() 

    def todo_test_get_numtracks(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_numtracks:

          # CD.get_numtracks(): return count
          # the number of tracks on the cdrom
          # 
          # Return the number of tracks on the cdrom in the drive. This will
          # return zero of the drive is empty or has no tracks.
          # 

        self.fail() 

    def todo_test_get_paused(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_paused:

          # CD.get_paused(): return bool
          # true if the drive is paused
          # 
          # Returns True if the drive is currently paused. 

        self.fail() 

    def todo_test_get_track_audio(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_track_audio:

          # CD.get_track_audio(track): return bool
          # true if the cdrom track has audio data
          # 
          # Determine if a track on a cdrom contains audio data. You can also
          # call CD.num_tracks() and CD.get_all() to determine more information
          # about the cdrom.
          # 
          # Note, track 0 is the first track on the CD.  Track numbers start at zero. 

        self.fail() 

    def todo_test_get_track_length(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_track_length:

          # CD.get_track_length(track): return seconds
          # length of a cdrom track
          # 
          # Return a floating point value in seconds of the length of the cdrom track. 
          # Note, track 0 is the first track on the CD.  Track numbers start at zero. 

        self.fail() 

    def todo_test_get_track_start(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.get_track_start:

          # CD.get_track_start(track): return seconds
          # start time of a cdrom track
          # 
          # Return the absolute time in seconds where at start of the cdrom track. 
          # Note, track 0 is the first track on the CD.  Track numbers start at zero. 

        self.fail() 

    def todo_test_init(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.init:

          # CD.init(): return None
          # initialize a cdrom drive for use
          # 
          # Initialize the cdrom drive for use. The drive must be initialized
          # for most CD methods to work.  Even if the rest of pygame has been
          # initialized.
          # 
          # There may be a brief pause while the drive is initialized. Avoid
          # CD.init() if the program should not stop for a second or two.
          # 

        self.fail() 

    def todo_test_pause(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.pause:

          # CD.pause(): return None
          # temporarily stop audio playback
          # 
          # Temporarily stop audio playback on the CD. The playback can be
          # resumed at the same point with the CD.resume() method. If the CD is
          # not playing this method does nothing.
          # 
          # Note, track 0 is the first track on the CD.  Track numbers start at zero. 

        self.fail() 

    def todo_test_play(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.play:

          # CD.init(): return None
          # initialize a cdrom drive for use
          # 
          # Playback audio from an audio cdrom in the drive. Besides the track
          # number argument, you can also pass a starting and ending time for
          # playback. The start and end time are in seconds, and can limit the
          # section of an audio track played.
          # 
          # If you pass a start time but no end, the audio will play to the end
          # of the track. If you pass a start time and 'None' for the end time,
          # the audio will play to the end of the entire disc.
          # 
          # See the CD.get_numtracks() and CD.get_track_audio() to find tracks to playback. 
          # Note, track 0 is the first track on the CD.  Track numbers start at zero. 

        self.fail() 

    def todo_test_quit(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.quit:

          # CD.quit(): return None
          # uninitialize a cdrom drive for use
          # 
          # Uninitialize a drive for use. Call this when your program will not
          # be accessing the drive for awhile.
          # 

        self.fail() 

    def todo_test_resume(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.resume:

          # CD.resume(): return None
          # unpause audio playback
          # 
          # Unpause a paused CD. If the CD is not paused or already playing,
          # this method does nothing.
          # 

        self.fail() 

    def todo_test_stop(self):

        # __doc__ (as of 2008-08-02) for pygame.cdrom.CD.stop:

          # CD.stop(): return None
          # stop audio playback
          # 
          # Stops playback of audio from the cdrom. This will also lose the
          # current playback position. This method does nothing if the drive
          # isn't already playing audio.
          # 

        self.fail() 

################################################################################

if __name__ == '__main__':
    unittest.main()
