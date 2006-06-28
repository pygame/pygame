#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import os
import sys

from SDL import *
from SDL.mixer import *

def usage():
    print >> sys.stderr, 'Usage: %s [-i] [-l] [-8] [-r rate] [-c channels] \
[-b buffers] [-v N] [-rwops] <musicfile>' % sys.argv[0]

def Menu():
    print 'Available commands: (p)ause (r)esume (h)alt > ',
    buf = raw_input()[0].lower()
    if buf == 'p':
        Mix_PauseMusic()
    elif buf == 'r':
        Mix_ResumeMusic()
    elif buf == 'h':
        Mix_HaltMusic()
    print 'Music playing: %s Paused: %s' % \
        (Mix_PlayingMusic(), Mix_PausedMusic())

if __name__ == '__main__':
    audio_rate = MIX_DEFAULT_FREQUENCY
    audio_format = MIX_DEFAULT_FORMAT
    audio_volume = MIX_MAX_VOLUME
    audio_channels = 2
    looping = 0
    reverse_stereo = 0
    reverse_sample = 0
    interactive = 0
    rwops = 0
    audio_buffers = 4096

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg[0] != '-':
            break
        elif arg == '-r':
            i += 1
            audio_rate = int(sys.argv[i])
        elif arg == '-m':
            audio_channels = 1
        elif arg == '-b':
            i += 1
            audio_buffers = int(sys.argv[i])
        elif arg == '-v':
            i += 1
            audio_volume = int(sys.argv[i])
        elif arg == '-c':
            i += 1
            audio_channels = int(sys.argv[i])
        elif arg == '-l':
            looping = -1
        elif arg == '-i':
            interactive = 1
        elif arg == '-8':
            audio_format = AUDIO_U8
        elif arg == '-rwops':
            rwops = 1
        else:
            usage()
            sys.exit(1)
        i += 1

    if i >= len(sys.argv):
        usage()
        sys.exit(1)

    SDL_Init(SDL_INIT_AUDIO)
    Mix_OpenAudio(audio_rate, audio_format, audio_channels, 4096)
    opened, audio_rate, audio_format, audio_channels = Mix_QuerySpec()
    channels_s = 'mono'
    if audio_channels == 2:
        channels_s = 'stereo'
    elif audio_channels > 2:
        channels_s = 'surround'
    endian = 'LE'
    if audio_format & 0x1000:
        endian = 'BE'
    print 'Opened audio at %d Hz %d bit %s (%s), %d bytes audio buffer' % \
        (audio_rate, audio_format & 0xff, channels_s, endian, audio_buffers)

    Mix_VolumeMusic(audio_volume)
    Mix_SetMusicCMD(os.getenv('MUSIC_CMD'))

    while i < len(sys.argv):
        if rwops:
            rwfp = SDL_RWFromFile(sys.argv[i], 'rb')
            music = Mix_LoadMUS_RW(rwfp)
        else:
            music = Mix_LoadMUS(sys.argv[i])
            
        try:
            print 'Playing %s' % sys.argv[i]
            Mix_FadeInMusic(music, looping, 2000)
            while Mix_PlayingMusic() or Mix_PausedMusic():
                if interactive:
                    Menu()
                else:
                    SDL_Delay(100)
        except KeyboardInterrupt:
            if Mix_PlayingMusic():
                Mix_FadeOutMusic(1500)
                SDL_Delay(1500)

        Mix_FreeMusic(music)
        music = None
        if rwops:
            SDL_FreeRW(rwfp)

        # If the user presses Ctrl-C more than once, exit.
        SDL_Delay(500)
        i += 1
    
    if music:
        Mix_FreeMusic(music)
    Mix_CloseAudio()
    SDL_Quit()
