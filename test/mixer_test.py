#################################### IMPORTS ###################################

import pygame, unittest, test_utils

from pygame import mixer

############################## MODULE LEVEL TESTS ##############################

class MixerModuleTest(unittest.TestCase):
    def test_get_init__returns_exact_values_used_for_init(self):

        # TODO: When this bug is fixed, testing through every combination
        #       will be too slow so adjust as necessary, at the moment it
        #       breaks the loop after first failure

        configs = ((r,b,c) for r in [11025, 22050, 44100, 48000] 
                           for b in [-16, -8, 8]    # + [16]
                           for c in [1, 2])

        for init_conf in configs:
            mixer.init(*init_conf)

            mixer_conf = mixer.get_init()

            mixer.quit()
            
            self.assertEquals(init_conf, mixer_conf)

    # TODO: "+16 (ie unsigned 16 bit samples) are not supported."

############################## CHANNEL CLASS TESTS #############################

class ChannelTest(unittest.TestCase):
    pass

############################### SOUND CLASS TESTS ##############################

class SoundTest(unittest.TestCase):
    pass

##################################### MAIN #####################################

if __name__ == '__main__':
    if test_utils.get_cl_fail_incomplete_opt():
        test_utils.fail_incomplete_tests = 1
        
    unittest.main()