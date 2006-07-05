#!/usr/bin/env python

'''A simple "decode sound, play it through SDL" example.

The much more complex, fancy and robust code is ``playsound.py``
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys

from SDL import *
from SDL.sound import *

class PlaysoundAudioCallbackData:
    pass

global_done_flag = 0

def audio_callback(data, stream):
    global global_done_flag

    sample = data.sample
    bw = 0 # bytes written to stream this time through the callback
    stream = stream.as_bytes()
    length = len(stream)

    while bw < length:
        if data.decoded_bytes == 0:     # need more data
            # if there wasn't previously an error or EOF, read more
            if (sample.flags & SOUND_SAMPLEFLAG_ERROR) == 0 and \
               (sample.flags & SOUND_SAMPLEFLAG_EOF) == 0:
                data.decoded_bytes = Sound_Decode(sample)
                data.decoded_offset = 0

            if data.decoded_bytes == 0:
                # ... there isn't any more data to read
                stream[bw:] = [0] * (length - bw)  # write silence
                global_done_flag = 1
                return
        
        # we have data decoded and read to write to the device
        cpysize = length - bw  # amount device still wants
        cpysize = min(cpysize, data.decoded_bytes) # clamp what we have left

        # if it's 0, next iteration will decode more or decide we're done
        if cpysize > 0:
            # write this iteration's data to the device
            stream[bw:bw+cpysize] = \
                sample.buffer[data.decoded_offset:data.decoded_offset+cpysize]

            # update state for next iteration or callback
            bw += cpysize
            data.decoded_offset += cpysize
            data.decoded_bytes -= cpysize

def playOneSoundFile(fname):
    global global_done_flag

    data = PlaysoundAudioCallbackData()

    data.decoded_bytes = 0
    data.decoded_offset = 0
    data.sample = Sound_NewSampleFromFile(fname, None, 65536)
    
    # Open device in format of the sound to be played.
    data.devformat = SDL_AudioSpec()
    data.devformat.freq = data.sample.actual.rate
    data.devformat.format = data.sample.actual.format
    data.devformat.channels = data.sample.actual.channels
    data.devformat.samples = 256 
    data.devformat.callback = audio_callback
    data.devformat.userdata = data

    SDL_OpenAudio(data.devformat, None)

    print 'Now playing [%s]...' % fname
    SDL_PauseAudio(0)  # SDL audio device is "paused" right after opening

    global_done_flag = 0     # the audio callback will flip this flag
    while not global_done_flag:
        SDL_Delay(10)  # just wait for the audio callback to finish

    # at this point, we've played the entire audio file.
    SDL_PauseAudio(1)   # so stop the device

    # Sleep two buffers' worth of audio before closing, in order to allow
    # playback to finish.  This isn't always enough; perhaps SDL needs a way
    # to explicitly wait for device drain?  Most apps don't have this issue,
    # since they aren't explicitly closing the device as soon as a sound file
    # is done playback.  As an alternative for this app, you could also change
    # the callback to write silence for a call or two before flipping
    # global_done_flag.

    SDL_Delay(2 * 1000 * data.devformat.samples / data.devformat.freq)

    # if there was an error, tell the user
    if data.sample.flags & SOUND_SAMPLEFLAG_ERROR:
        print >> sys.stderr, 'Error decoding file: %s' % Sound_GetError()

    Sound_FreeSample(data.sample)
    SDL_CloseAudio()

if __name__ == '__main__':
    Sound_Init()    # this calls SDL_Init(SDL_Init_Audio)

    for arg in sys.argv[1:]:
        playOneSoundFile(arg)

    Sound_Quit()
    SDL_Quit()
