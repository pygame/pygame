#!/usr/bin/env python

"""extremely simple demonstration playing a soundfile
and waiting for it to finish. you'll need the pygame.mixer
module for this to work. Note how in this simple example we
don't even bother loading all of the pygame package. Just
pick the mixer for sound and time for the delay function."""

import os.path
import pygame.mixer as mixer
import pygame.time as time

#choose a desired audio format
mixer.init(11025)
if not mixer.get_init():
    raise SystemExit, 'Cannot Initialize Mixer'


#load the sound    
file = os.path.join('data', 'secosmic_lo.wav')
sound = mixer.Sound(file)


#start playing
print 'Playing Sound...'
channel = sound.play()


#poll until finished
while channel.get_busy(): #still playing
    print '  ...still going...'
    time.delay(1000)
print '...Finished'



