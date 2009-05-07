#!/usr/bin/env python

"""extremely simple demonstration playing a soundfile
and waiting for it to finish. you'll need the pygame.mixer
module for this to work. Note how in this simple example we
don't even bother loading all of the pygame package. Just
pick the mixer for sound and time for the delay function.

Optional command line argument:
  the name of an audio file.
  

"""

import os.path, sys
import pygame.mixer, pygame.time
mixer = pygame.mixer
time = pygame.time

main_dir = os.path.split(os.path.abspath(__file__))[0]

def main(file_path=None):
    """Play an audio file as a buffered sound sample

    Option argument:
        the name of an audio file (default data/secosmic_low.wav

    """
    if file_path is None:
        file_path = os.path.join(main_dir,
                                 'data',
                                 'secosmic_lo.wav')

    #choose a desired audio format
    mixer.init(11025) #raises exception on fail


    #load the sound    
    sound = mixer.Sound(file_path)


    #start playing
    print ('Playing Sound...')
    channel = sound.play()


    #poll until finished
    while channel.get_busy(): #still playing
        print ('  ...still going...')
        time.wait(1000)
    print ('...Finished')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
