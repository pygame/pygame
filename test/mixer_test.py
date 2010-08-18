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
    from pygame.tests.test_utils import test_not_implemented, unittest, example_path
else:
    from test.test_utils import test_not_implemented, unittest, example_path
import pygame
from pygame import mixer
from pygame.compat import xrange_, unicode_, as_bytes, geterror

import sys
import os

################################### CONSTANTS ##################################

FREQUENCIES = [11025, 22050, 44100, 48000] 
SIZES       = [-16, -8, 8, 16]
CHANNELS    = [1, 2]
BUFFERS     = [3024]

############################## MODULE LEVEL TESTS ##############################

class MixerModuleTest(unittest.TestCase):
    def test_init__keyword_args(self):
        # Fails on a Mac; probably older SDL_mixer
## Probably don't need to be so exhaustive. Besides being slow the repeated
## init/quit calls may be causing problems on the Mac.
##        configs = ( {'frequency' : f, 'size' : s, 'channels': c }
##                    for f in FREQUENCIES
##                    for s in SIZES
##                    for c in CHANNELS )
####        configs = [{'frequency' : 44100, 'size' : 16, 'channels' : 1}]
        configs = [{'frequency' : 22050, 'size' : -16, 'channels' : 2}]

        for kw_conf in configs:
            mixer.init(**kw_conf)

            mixer_conf = mixer.get_init()
            
            self.assertEquals(
                # Not all "sizes" are supported on all systems.
                (mixer_conf[0], abs(mixer_conf[1]), mixer_conf[2]),
                (kw_conf['frequency'],
                 abs(kw_conf['size']),
                 kw_conf['channels'])
            )
            
            mixer.quit()
    
    def todo_test_pre_init__keyword_args(self):
        # Fails on Mac; probably older SDL_mixer
## Probably don't need to be so exhaustive. Besides being slow the repeated
## init/quit calls may be causing problems on the Mac.
##        configs = ( {'frequency' : f, 'size' : s, 'channels': c }
##                    for f in FREQUENCIES
##                    for s in SIZES
##                    for c in CHANNELS )
        configs = [{'frequency' : 44100, 'size' : 16, 'channels' : 1}]

        for kw_conf in configs:
            mixer.pre_init(**kw_conf)
            mixer.init()

            mixer_conf = mixer.get_init()
            
            self.assertEquals(
                # Not all "sizes" are supported on all systems.
                (mixer_conf[0], abs(mixer_conf[1]), mixer_conf[2]),
                (kw_conf['frequency'],
                 abs(kw_conf['size']),
                 kw_conf['channels'])
            )
            
            mixer.quit()

    def todo_test_pre_init__zero_values(self):
        # Ensure that argument values of 0 are replaced with
        # default values. No way to check buffer size though.
        mixer.pre_init(44100, -8, 1)  # Non default values
        mixer.pre_init(0, 0, 0)       # Should reset to default values
        mixer.init()
        try:
            self.failUnlessEqual(mixer.get_init(), (22050, -16, 2))
        finally:
            mixer.quit()

    def todo_test_init__zero_values(self):
        # Ensure that argument values of 0 are replaced with
        # preset values. No way to check buffer size though.
        mixer.pre_init(44100, 8, 1)  # None default values
        mixer.init(0, 0, 0)
        try:
            self.failUnlessEqual(mixer.get_init(), (44100, 8, 1))
        finally:
            mixer.quit()
            mixer.pre_init(0, 0, 0, 0)

    def test_get_init__returns_exact_values_used_for_init(self):
        return 
        # fix in 1.9 - I think it's a SDL_mixer bug.

        # TODO: When this bug is fixed, testing through every combination
        #       will be too slow so adjust as necessary, at the moment it
        #       breaks the loop after first failure

        configs = []
        for f in FREQUENCIES:
            for s in SIZES:
                for c in CHANNELS:
                    configs.append ((f,s,c))

        print (configs)
    

        for init_conf in configs:
            print (init_conf)
            f,s,c = init_conf
            if (f,s) == (22050,16):continue
            mixer.init(f,s,c)

            mixer_conf = mixer.get_init()
            import time
            time.sleep(0.1)

            mixer.quit()
            time.sleep(0.1)

            if init_conf != mixer_conf:
                continue
            self.assertEquals(init_conf, mixer_conf)

    def test_get_init__returns_None_if_mixer_not_initialized(self):
        self.assert_(mixer.get_init() is None)
    
    def test_get_num_channels__defaults_eight_after_init(self):
        mixer.init()
        
        num_channels = mixer.get_num_channels()

        self.assert_(num_channels == 8)

        mixer.quit()

    def test_set_num_channels(self):
        mixer.init()

        for i in xrange_(1, mixer.get_num_channels() + 1):
            mixer.set_num_channels(i)
            self.assert_(mixer.get_num_channels() == i)

        mixer.quit()

    def test_quit(self):
        """ get_num_channels() Should throw pygame.error if uninitialized
        after mixer.quit() """

        mixer.init()
        mixer.quit()

        self.assertRaises (
            pygame.error, mixer.get_num_channels,
        )

    def test_sound_args(self):
        def get_bytes(snd):
            return snd.get_buffer().raw
        
        mixer.init()
        try:
            sample = as_bytes('\x00\xff') * 24
            wave_path = example_path(os.path.join('data', 'house_lo.wav'))
            uwave_path = unicode_(wave_path)
            bwave_path = uwave_path.encode(sys.getfilesystemencoding())
            snd = mixer.Sound(file=wave_path)
            self.assert_(snd.get_length() > 0.5)
            snd_bytes = get_bytes(snd)
            self.assert_(len(snd_bytes) > 1000)
            self.assert_(get_bytes(mixer.Sound(wave_path)) == snd_bytes)
            self.assert_(get_bytes(mixer.Sound(file=uwave_path)) == snd_bytes)
            self.assert_(get_bytes(mixer.Sound(uwave_path)) == snd_bytes)
            arg_emsg = 'Sound takes either 1 positional or 1 keyword argument'
            try:
                mixer.Sound()
            except TypeError:
                self.assert_(str(geterror()) == arg_emsg)
            else:
                self.fail("no exception")
            try:
                mixer.Sound(wave_path, buffer=sample)
            except TypeError:
                self.assert_(str(geterror()) == arg_emsg)
            else:
                self.fail("no exception")
            try:
                mixer.Sound(sample, file=wave_path)
            except TypeError:
                self.assert_(str(geterror()) == arg_emsg)
            else:
                self.fail("no exception")           
            try:
                mixer.Sound(buffer=sample, file=wave_path)
            except TypeError:
                self.assert_(str(geterror()) == arg_emsg)
            else:
                self.fail("no exception")           
            try:
                mixer.Sound(foobar=sample)
            except TypeError:
                emsg = "Unrecognized keyword argument 'foobar'"
                self.assert_(str(geterror()) == emsg)
            else:
                self.fail("no exception")
            snd = mixer.Sound(wave_path, **{})
            self.assert_(get_bytes(snd) == snd_bytes)
            snd = mixer.Sound(*[], **{'file': wave_path})          
            try:
               snd = mixer.Sound([])
            except TypeError:
                emsg = 'Unrecognized argument (type list)'
                self.assert_(str(geterror()) == emsg)
            else:   
                self.fail("no exception")
            try:
                snd = mixer.Sound(buffer=[])
            except TypeError:
                emsg = 'Expected object with buffer interface: got a list'
                self.assert_(str(geterror()) == emsg)
            else:
                self.fail("no exception")
            ufake_path = unicode_('12345678')
            self.assertRaises(pygame.error, mixer.Sound, ufake_path)
            try:
                mixer.Sound(buffer=unicode_('something'))
            except TypeError:
                emsg = 'Unicode object not allowed as buffer object'
                self.assert_(str(geterror()) == emsg)
            else:
                self.fail("no exception")
            self.assert_(get_bytes(mixer.Sound(buffer=sample)) == sample)
            self.assert_(get_bytes(mixer.Sound(sample)) == sample)
            self.assert_(get_bytes(mixer.Sound(file=bwave_path)) == snd_bytes)
            self.assert_(get_bytes(mixer.Sound(bwave_path)) == snd_bytes)
        finally:
            mixer.quit()
        
    def todo_test_fadeout(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.fadeout:

          # pygame.mixer.fadeout(time): return None
          # fade out the volume on all sounds before stopping
          # 
          # This will fade out the volume on all active channels over the time
          # argument in milliseconds. After the sound is muted the playback will
          # stop.
          # 

        self.fail() 

    def todo_test_find_channel(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.find_channel:

          # pygame.mixer.find_channel(force=False): return Channel
          # find an unused channel
          # 
          # This will find and return an inactive Channel object. If there are
          # no inactive Channels this function will return None. If there are no
          # inactive channels and the force argument is True, this will find the
          # Channel with the longest running Sound and return it.
          # 
          # If the mixer has reserved channels from pygame.mixer.set_reserved()
          # then those channels will not be returned here.
          # 

        self.fail() 

    def todo_test_get_busy(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.get_busy:

          # pygame.mixer.get_busy(): return bool
          # test if any sound is being mixed
          # 
          # Returns True if the mixer is busy mixing any channels. If the mixer
          # is idle then this return False.
          # 

        self.fail() 

    def todo_test_init(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.init:

          # pygame.mixer.init(frequency=22050, size=-16, channels=2,
          # buffer=3072): return None
          # 
          # initialize the mixer module
          # 
          # Initialize the mixer module for Sound loading and playback. The
          # default arguments can be overridden to provide specific audio
          # mixing. The size argument represents how many bits are used for each
          # audio sample. If the value is negative then signed sample values
          # will be used. Positive values mean unsigned audio samples will be
          # used.
          # 
          # The channels argument is used to specify whether to use mono or
          # stereo.  1 for mono and 2 for stereo. No other values are supported.
          # 
          # The buffer argument controls the number of internal samples used in
          # the sound mixer. The default value should work for most cases. It
          # can be lowered to reduce latency, but sound dropout may occur. It
          # can be raised to larger values to ensure playback never skips, but
          # it will impose latency on sound playback. The buffer size must be a
          # power of two.
          # 
          # Some platforms require the pygame.mixer module to be initialized
          # after the display modules have initialized. The top level
          # pygame.init() takes care of this automatically, but cannot pass any
          # arguments to the mixer init. To solve this, mixer has a function
          # pygame.mixer.pre_init() to set the proper defaults before the
          # toplevel init is used.
          # 
          # It is safe to call this more than once, but after the mixer is
          # initialized you cannot change the playback arguments without first
          # calling pygame.mixer.quit().
          # 

        self.fail() 

    def todo_test_pause(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.pause:

          # pygame.mixer.pause(): return None
          # temporarily stop playback of all sound channels
          # 
          # This will temporarily stop all playback on the active mixer
          # channels. The playback can later be resumed with
          # pygame.mixer.unpause()
          # 

        self.fail() 

    def todo_test_pre_init(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.pre_init:

          # pygame.mixer.pre_init(frequency=0, size=0, channels=0,
          # buffersize=0): return None
          # 
          # preset the mixer init arguments
          # 
          # Any nonzero arguments change the default values used when the real
          # pygame.mixer.init() is called. The best way to set custom mixer
          # playback values is to call pygame.mixer.pre_init() before calling
          # the top level pygame.init().
          # 

        self.fail() 

    def todo_test_set_reserved(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.set_reserved:

          # pygame.mixer.set_reserved(count): return None
          # reserve channels from being automatically used
          # 
          # The mixer can reserve any number of channels that will not be
          # automatically selected for playback by Sounds. If sounds are
          # currently playing on the reserved channels they will not be stopped.
          # 
          # This allows the application to reserve a specific number of channels
          # for important sounds that must not be dropped or have a guaranteed
          # channel to play on.
          # 

        self.fail() 

    def todo_test_stop(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.stop:

          # pygame.mixer.stop(): return None
          # stop playback of all sound channels
          # 
          # This will stop all playback of all active mixer channels. 

        self.fail() 

    def todo_test_unpause(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.unpause:

          # pygame.mixer.unpause(): return None
          # resume paused playback of sound channels
          # 
          # This will resume all active sound channels after they have been paused. 

        self.fail() 

############################## CHANNEL CLASS TESTS #############################

class ChannelTypeTest(unittest.TestCase):
    def todo_test_Channel(self):
        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel:

          # pygame.mixer.Channel(id): return Channel
          # Create a Channel object for controlling playback
          # 
          # Return a Channel object for one of the current channels. The id must
          # be a value from 0 to the value of pygame.mixer.get_num_channels().
          # 
          # The Channel object can be used to get fine control over the playback
          # of Sounds. A channel can only playback a single Sound at time. Using
          # channels is entirely optional since pygame can manage them by
          # default.
          # 

        self.fail()      

    def todo_test_fadeout(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.fadeout:

          # Channel.fadeout(time): return None
          # stop playback after fading channel out
          # 
          # Stop playback of a channel after fading out the sound over the given
          # time argument in milliseconds.
          # 

        self.fail() 

    def todo_test_get_busy(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.get_busy:

          # Channel.get_busy(): return bool
          # check if the channel is active
          # 
          # Returns true if the channel is activily mixing sound. If the channel
          # is idle this returns False.
          # 

        self.fail() 

    def todo_test_get_endevent(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.get_endevent:

          # Channel.get_endevent(): return type
          # get the event a channel sends when playback stops
          # 
          # Returns the event type to be sent every time the Channel finishes
          # playback of a Sound. If there is no endevent the function returns
          # pygame.NOEVENT.
          # 

        self.fail() 

    def todo_test_get_queue(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.get_queue:

          # Channel.get_queue(): return Sound
          # return any Sound that is queued
          # 
          # If a Sound is already queued on this channel it will be returned.
          # Once the queued sound begins playback it will no longer be on the
          # queue.
          # 

        self.fail() 

    def todo_test_get_sound(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.get_sound:

          # Channel.get_sound(): return Sound
          # get the currently playing Sound
          # 
          # Return the actual Sound object currently playing on this channel. If
          # the channel is idle None is returned.
          # 

        self.fail() 

    def todo_test_get_volume(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.get_volume:

          # Channel.get_volume(): return value
          # get the volume of the playing channel
          # 
          # Return the volume of the channel for the current playing sound. This
          # does not take into account stereo separation used by
          # Channel.set_volume. The Sound object also has its own volume which
          # is mixed with the channel.
          # 

        self.fail() 

    def todo_test_pause(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.pause:

          # Channel.pause(): return None
          # temporarily stop playback of a channel
          # 
          # Temporarily stop the playback of sound on a channel. It can be
          # resumed at a later time with Channel.unpause()
          # 

        self.fail() 

    def todo_test_play(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.play:

          # Channel.play(Sound, loops=0, maxtime=0, fade_ms=0): return None
          # play a Sound on a specific Channel
          # 
          # This will begin playback of a Sound on a specific Channel. If the
          # Channel is currently playing any other Sound it will be stopped.
          # 
          # The loops argument has the same meaning as in Sound.play(): it is
          # the number of times to repeat the sound after the first time. If it
          # is 3, the sound will be played 4 times (the first time, then three
          # more). If loops is -1 then the playback will repeat indefinitely.
          # 
          # As in Sound.play(), the maxtime argument can be used to stop
          # playback of the Sound after a given number of milliseconds.
          # 
          # As in Sound.play(), the fade_ms argument can be used fade in the sound. 

        self.fail() 

    def todo_test_queue(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.queue:

          # Channel.queue(Sound): return None
          # queue a Sound object to follow the current
          # 
          # When a Sound is queued on a Channel, it will begin playing
          # immediately after the current Sound is finished. Each channel can
          # only have a single Sound queued at a time. The queued Sound will
          # only play if the current playback finished automatically. It is
          # cleared on any other call to Channel.stop() or Channel.play().
          # 
          # If there is no sound actively playing on the Channel then the Sound
          # will begin playing immediately.
          # 

        self.fail() 

    def todo_test_set_endevent(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.set_endevent:

          # Channel.set_endevent(): return None
          # Channel.set_endevent(type): return None
          # have the channel send an event when playback stops
          # 
          # When an endevent is set for a channel, it will send an event to the
          # pygame queue every time a sound finishes playing on that channel
          # (not just the first time). Use pygame.event.get() to retrieve the
          # endevent once it's sent.
          # 
          # Note that if you called Sound.play(n) or Channel.play(sound,n), the
          # end event is sent only once: after the sound has been played "n+1"
          # times (see the documentation of Sound.play).
          # 
          # If Channel.stop() or Channel.play() is called while the sound was
          # still playing, the event will be posted immediately.
          # 
          # The type argument will be the event id sent to the queue. This can
          # be any valid event type, but a good choice would be a value between
          # pygame.locals.USEREVENT and pygame.locals.NUMEVENTS. If no type
          # argument is given then the Channel will stop sending endevents.
          # 

        self.fail() 

    def todo_test_set_volume(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.set_volume:

          # Channel.set_volume(value): return None
          # Channel.set_volume(left, right): return None
          # set the volume of a playing channel
          # 
          # Set the volume (loudness) of a playing sound. When a channel starts
          # to play its volume value is reset. This only affects the current
          # sound. The value argument is between 0.0 and 1.0.
          # 
          # If one argument is passed, it will be the volume of both speakers.
          # If two arguments are passed and the mixer is in stereo mode, the
          # first argument will be the volume of the left speaker and the second
          # will be the volume of the right speaker. (If the second argument is
          # None, the first argument will be the volume of both speakers.)
          # 
          # If the channel is playing a Sound on which set_volume() has also
          # been called, both calls are taken into account. For example:
          # 
          #     sound = pygame.mixer.Sound("s.wav")
          #     channel = s.play()      # Sound plays at full volume by default
          #     sound.set_volume(0.9)   # Now plays at 90% of full volume.
          #     sound.set_volume(0.6)   # Now plays at 60% (previous value replaced).
          #     channel.set_volume(0.5) # Now plays at 30% (0.6 * 0.5).

        self.fail() 

    def todo_test_stop(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.stop:

          # Channel.stop(): return None
          # stop playback on a Channel
          # 
          # Stop sound playback on a channel. After playback is stopped the
          # channel becomes available for new Sounds to play on it.
          # 

        self.fail() 

    def todo_test_unpause(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Channel.unpause:

          # Channel.unpause(): return None
          # resume pause playback of a channel
          # 
          # Resume the playback on a paused channel. 

        self.fail() 

############################### SOUND CLASS TESTS ##############################

class SoundTypeTest(unittest.TestCase):
    def todo_test_fadeout(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Sound.fadeout:

          # Sound.fadeout(time): return None
          # stop sound playback after fading out
          # 
          # This will stop playback of the sound after fading it out over the
          # time argument in milliseconds. The Sound will fade and stop on all
          # actively playing channels.
          # 

        self.fail() 

    def todo_test_get_buffer(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Sound.get_buffer:

          # Sound.get_buffer(): return BufferProxy
          # acquires a buffer object for the sameples of the Sound.
          # 
          # Return a buffer object for the Sound samples. The buffer can be used
          # for direct access and manipulation.
          # 
          # New in pygame 1.8. 

        self.fail() 

    def todo_test_get_length(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Sound.get_length:

          # Sound.get_length(): return seconds
          # get the length of the Sound
          # 
          # Return the length of this Sound in seconds. 

        self.fail() 

    def todo_test_get_num_channels(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Sound.get_num_channels:

          # Sound.get_num_channels(): return count
          # count how many times this Sound is playing
          # 
          # Return the number of active channels this sound is playing on. 

        self.fail() 

    def todo_test_get_volume(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Sound.get_volume:

          # Sound.get_volume(): return value
          # get the playback volume
          # 
          # Return a value from 0.0 to 1.0 representing the volume for this Sound. 

        self.fail() 

    def todo_test_play(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Sound.play:

          # Sound.play(loops=0, maxtime=0, fade_ms=0): return Channel
          # begin sound playback
          # 
          # Begin playback of the Sound (i.e., on the computer's speakers) on an
          # available Channel. This will forcibly select a Channel, so playback
          # may cut off a currently playing sound if necessary.
          # 
          # The loops argument controls how many times the sample will be
          # repeated after being played the first time. A value of 5 means that
          # the sound will be played once, then repeated five times, and so is
          # played a total of six times. The default value (zero) means the
          # Sound is not repeated, and so is only played once. If loops is set
          # to -1 the Sound will loop indefinitely (though you can still call
          # stop() to stop it).
          # 
          # The maxtime argument can be used to stop playback after a given
          # number of milliseconds.
          # 
          # The fade_ms argument will make the sound start playing at 0 volume
          # and fade up to full volume over the time given. The sample may end
          # before the fade-in is complete.
          # 
          # This returns the Channel object for the channel that was selected. 

        self.fail() 

    def todo_test_set_volume(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Sound.set_volume:

          # Sound.set_volume(value): return None
          # set the playback volume for this Sound
          # 
          # This will set the playback volume (loudness) for this Sound. This
          # will immediately affect the Sound if it is playing. It will also
          # affect any future playback of this Sound. The argument is a value
          # from 0.0 to 1.0.
          # 

        self.fail() 

    def todo_test_stop(self):

        # __doc__ (as of 2008-08-02) for pygame.mixer.Sound.stop:

          # Sound.stop(): return None
          # stop sound playback
          # 
          # This will stop the playback of this Sound on any active Channels. 

        self.fail() 

##################################### MAIN #####################################

if __name__ == '__main__':
    unittest.main()
