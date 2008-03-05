#!/usr/bin/env python

"""extremely simple demonstration playing a soundfile
and waiting for it to finish. you'll need the pygame.mixer
module for this to work. Note how in this simple example we
don't even bother loading all of the pygame package. Just
pick the mixer for sound and time for the delay function."""

import os.path, sys
import pygame.mixer, pygame.time
mixer = pygame.mixer
time = pygame.time

#choose a desired audio format
mixer.init(11025) #raises exception on fail


#load the sound    

if len(sys.argv) > 1 and "wav" in sys.argv[1]:
    file = sys.argv[1]
else:
    file = os.path.join('data', 'secosmic_lo.wav')
sound = mixer.Sound(file)


#start playing
print 'Playing Sound...'
channel = sound.play()


#poll until finished
while channel.get_busy(): #still playing
    print '  ...still going...'
    time.wait(1000)
print '...Finished'



