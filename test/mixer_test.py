#################################### IMPORTS ###################################

import pygame, unittest, test_utils

from pygame import mixer

from test_utils import test_not_implemented

################################### CONSTANTS ##################################

FREQUENCIES = [11025, 22050, 44100, 48000] 
SIZES       = [-16, -8, 8, 16]
CHANNELS    = [1, 2]
BUFFERS     = [3024]

# "+16 (ie unsigned 16 bit samples) are not supported."

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

        # TODO: When this bug is fixed, testing through every combination
        #       will be too slow so adjust as necessary, at the moment it
        #       breaks the loop after first failure

        configs = ((f,s,c) for f in FREQUENCIES
                           for s in SIZES
                           for c in CHANNELS)

        for init_conf in configs:
            mixer.init(*init_conf)

            mixer_conf = mixer.get_init()

            mixer.quit()

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
        mixer.init()
        mixer.quit()

        # assertRaises does not work here
        # self.assertRaises(pygame.error, mixer.get_num_channels())

        try:
            chans = mixer.get_num_channels()
        except Exception, e:
            self.assert_(type(e) == pygame.error)
        else:
            self.assert_( chans is
                'get_num_channels() Should throw pygame.error if uninitialized '
                'after mixer.quit()' )

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

class ChannelTest(unittest.TestCase):
    pass

############################### SOUND CLASS TESTS ##############################

class SoundTest(unittest.TestCase):
    pass

##################################### MAIN #####################################

if __name__ == '__main__':
    test_utils.get_fail_incomplete_tests_option()
    unittest.main()