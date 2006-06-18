#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import os
import sys

from SDL import *

class Wave:
    pass
wave = Wave()

def fillerup(unused, stream):
    # It is not possible to do pointer arithmetic in Python (even with
    # ctypes, so SDL_MixAudio cannot be used.  Instead, standard slice
    # assignment is used for the same outcome.

    waveleft = wave.soundlen - wave.soundpos
    offset = 0
    length = len(stream)
    
    while waveleft <= length:
        stream[offset:waveleft + offset] = wave.sound[wave.soundpos:]
        offset += waveleft
        length -= waveleft
        waveleft = wave.soundlen
        wave.soundpos = 0
    stream[offset:] = wave.sound[wave.soundpos:wave.soundpos + length]
    wave.soundpos += length

if __name__ == '__main__':
    SDL_Init(SDL_INIT_AUDIO)

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = os.path.join(os.path.dirname(sys.argv[0]), 'sample.wav')
    
    wave.spec, wave.sound = SDL_LoadWAV(filename)
    wave.soundlen = len(wave.sound)
    wave.spec.callback = fillerup
    wave.soundpos = 0

    SDL_OpenAudio(wave.spec, None)
    SDL_PauseAudio(False)

    print 'Using audio driver: %s' % SDL_AudioDriverName()
    while SDL_GetAudioStatus() == SDL_AUDIO_PLAYING:
        try:
            SDL_Delay(1000)
        except KeyboardInterrupt:
            break

    SDL_CloseAudio()
    SDL_FreeWAV(wave.sound)
    SDL_Quit()
