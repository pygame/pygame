#################################### IMPORTS ###################################

import test_utils
import test.unittest as unittest

import pygame

from pygame import mixer
import os

from test_utils import test_not_implemented

################################### CONSTANTS ##################################

FREQUENCIES = [11025, 22050, 44100, 48000] 
SIZES       = [-16, -8, 8, 16]
CHANNELS    = [1, 2]
BUFFERS     = [3024]

############################## MODULE LEVEL TESTS ##############################

class MixerModuleTest(unittest.TestCase):
    # def test_init__keyword_args(self):
    #     configs = ( {'frequency' : f, 'size' : s, 'channels': c }
    #                 for f in FREQUENCIES
    #                 for s in SIZES
    #                 for c in CHANNELS )

    #     for kw_conf in configs:
    #         mixer.init(*kw_conf)

    #         mixer_conf = mixer.get_init()
            
    #         self.assertEquals(
    #             mixer_conf,
    #             (kw_conf['frequency'], kw_conf['size'] , kw_conf['channels'])
    #         )
            
    #         mixer.quit()
    
    # Documentation makes it seem as though init() takes kw args
    # TypeError: init() takes no keyword arguments
    
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

        print configs
    

        for init_conf in configs:
            print init_conf
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

        for i in xrange(1, mixer.get_num_channels() + 1):
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

    def test_pre_init(self):
    
        # Doc string for pygame.mixer.pre_init:
    
          # pygame.mixer.pre_init(frequency=0, size=0, channels=0, buffersize=0): return None
          # preset the mixer init arguments
    
        self.assert_(test_not_implemented())
    
    
    def test_fadeout(self):
    
        # Doc string for pygame.mixer.fadeout:
    
          # pygame.mixer.fadeout(time): return None
          # fade out the volume on all sounds before stopping
    
        self.assert_(test_not_implemented())
    
    def test_find_channel(self):
    
        # Doc string for pygame.mixer.find_channel:
    
          # pygame.mixer.find_channel(force=False): return Channel
          # find an unused channel
    
        self.assert_(test_not_implemented())
    
    def test_get_busy(self):
    
        # Doc string for pygame.mixer.get_busy:
    
          # pygame.mixer.get_busy(): return bool
          # test if any sound is being mixed
    
        self.assert_(test_not_implemented())
            
    def test_init(self):
    
        # Doc string for pygame.mixer.init:
    
          # pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=3072): return None
          # initialize the mixer module
    
        self.assert_(test_not_implemented())
    
    def test_pause(self):
    
        # Doc string for pygame.mixer.pause:
    
          # pygame.mixer.pause(): return None
          # temporarily stop playback of all sound channels
    
        self.assert_(test_not_implemented())
    
    def test_unpause(self):
    
        # Doc string for pygame.mixer.unpause:
    
          # pygame.mixer.unpause(): return None
          # resume paused playback of sound channels
    
        self.assert_(test_not_implemented())
        
        
    def test_set_reserved(self):
    
        # Doc string for pygame.mixer.set_reserved:
    
          # pygame.mixer.set_reserved(count): return None
          # reserve channels from being automatically used

        self.assert_(test_not_implemented())
    
    def test_stop(self):
        # Doc string for pygame.mixer.stop:
    
          # pygame.mixer.stop(): return None
          # stop playback of all sound channels
    
        self.assert_(test_not_implemented())

############################## CHANNEL CLASS TESTS #############################

class ChannelTypeTest(unittest.TestCase):
    
    def test_Channel(self):
      self.assert_(test_not_implemented())
      
    def test_fadeout(self):

        # __doc__ (as of 2008-07-02) for pygame.mixer.Channel.fadeout:

          # Channel.fadeout(time): return None
          # stop playback after fading channel out

        self.assert_(test_not_implemented()) 

    def test_get_busy(self):

        # __doc__ (as of 2008-07-02) for pygame.mixer.Channel.get_busy:

          # Channel.get_busy(): return bool
          # check if the channel is active

        self.assert_(test_not_implemented()) 

    def test_get_endevent(self):

        # __doc__ (as of 2008-07-02) for pygame.mixer.Channel.get_endevent:

          # Channel.get_endevent(): return type
          # get the event a channel sends when playback stops

        self.assert_(test_not_implemented()) 

    def test_get_queue(self):

        # __doc__ (as of 2008-07-02) for pygame.mixer.Channel.get_queue:

          # Channel.get_queue(): return Sound
          # return any Sound that is queued

        self.assert_(test_not_implemented()) 

    def test_get_sound(self):

        # __doc__ (as of 2008-07-02) for pygame.mixer.Channel.get_sound:

          # Channel.get_sound(): return Sound
          # get the currently playing Sound

        self.assert_(test_not_implemented()) 

    def test_get_volume(self):

        # __doc__ (as of 2008-07-02) for pygame.mixer.Channel.get_volume:

          # Channel.get_volume(): return value
          # get the volume of the playing channel

        self.assert_(test_not_implemented()) 

    def test_pause(self):

        # __doc__ (as of 2008-07-02) for pygame.mixer.Channel.pause:

          # Channel.pause(): return None
          # temporarily stop playback of a channel

        self.assert_(test_not_implemented()) 

    def test_play(self):

        # __doc__ (as of 2008-07-02) for pygame.mixer.Channel.play:

          # Channel.play(Sound, loops=0, maxtime=0, fade_ms=0): return None
          # play a Sound on a specific Channel

        self.assert_(test_not_implemented()) 

    def test_queue(self):

        # __doc__ (as of 2008-07-02) for pygame.mixer.Channel.queue:

          # Channel.queue(Sound): return None
          # queue a Sound object to follow the current

        self.assert_(test_not_implemented()) 

    def test_set_endevent(self):

        # __doc__ (as of 2008-07-02) for pygame.mixer.Channel.set_endevent:

          # Channel.set_endevent(): return None
          # Channel.set_endevent(type): return None
          # have the channel send an event when playback stops

        self.assert_(test_not_implemented()) 

    def test_set_volume(self):

        # __doc__ (as of 2008-07-02) for pygame.mixer.Channel.set_volume:

          # Channel.set_volume(value): return None
          # Channel.set_volume(left, right): return None
          # set the volume of a playing channel

        self.assert_(test_not_implemented()) 

    def test_stop(self):

        # __doc__ (as of 2008-07-02) for pygame.mixer.Channel.stop:

          # Channel.stop(): return None
          # stop playback on a Channel

        self.assert_(test_not_implemented()) 

    def test_unpause(self):

        # __doc__ (as of 2008-07-02) for pygame.mixer.Channel.unpause:

          # Channel.unpause(): return None
          # resume pause playback of a channel

        self.assert_(test_not_implemented()) 


############################### SOUND CLASS TESTS ##############################

class SoundTypeTest(unittest.TestCase):
    def test_fadeout(self):
    
        # Doc string for pygame.mixer.Sound.fadeout:
    
          # Sound.fadeout(time): return None
          # stop sound playback after fading out
    
        self.assert_(test_not_implemented())
    
    def test_get_buffer(self):
    
        # Doc string for pygame.mixer.Sound.get_buffer:
    
          # Sound.get_buffer(): return BufferProxy
          # acquires a buffer object for the sameples of the Sound.
    
        self.assert_(test_not_implemented())
    
    def test_get_length(self):
    
        # Doc string for pygame.mixer.Sound.get_length:
    
          # Sound.get_length(): return seconds
          # get the length of the Sound
    
        self.assert_(test_not_implemented())
    
    def test_get_num_channels(self):
    
        # Doc string for pygame.mixer.Sound.get_num_channels:
    
          # Sound.get_num_channels(): return count
          # count how many times this Sound is playing
    
        self.assert_(test_not_implemented())
    
    def test_get_volume(self):
    
        # Doc string for pygame.mixer.Sound.get_volume:
    
          # Sound.get_volume(): return value
          # get the playback volume
    
        self.assert_(test_not_implemented())
    
    def test_play(self):
    
        # Doc string for pygame.mixer.Sound.play:
    
          # Sound.play(loops=0, maxtime=0, fade_ms=0): return Channel
          # begin sound playback
    
        self.assert_(test_not_implemented())
    
    def test_set_volume(self):
    
        # Doc string for pygame.mixer.Sound.set_volume:
    
          # Sound.set_volume(value): return None
          # set the playback volume for this Sound
    
        self.assert_(test_not_implemented())
    
    def test_stop(self):
    
        # Doc string for pygame.mixer.Sound.stop:
    
          # Sound.stop(): return None
          # stop sound playback
    
        self.assert_(test_not_implemented())

##################################### MAIN #####################################

if __name__ == '__main__':
    unittest.main()
