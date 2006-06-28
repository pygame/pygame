#!/usr/bin/env python

'''Test application for SDL.mixer
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys

from SDL import *
from SDL.mixer import *

# Various mixer tests; enable the ones you want.
TEST_MIX_VERSIONS = True
TEST_MIX_CHANNELFINISHED = True
TEST_MIX_PANNING = False
TEST_MIX_DISTANCE = False
TEST_MIX_POSITION = True

if TEST_MIX_POSITION and (TEST_MIX_DISTANCE or TEST_MIX_PANNING):
    raise 'TEST_MIX_POSITION cannot be used with TEST_MIX_DISTANCE or PANNING'

channel_is_done = 0

def test_versions():
    print >> sys.stderr, 'Dyanamically linked against SDL %r and SDL_mixer %r' \
        % (SDL_Linked_Version(), Mix_Linked_Version())

def channel_complete_callback(chan):
    global channel_is_done
    done_chunk = Mix_GetChunk(chan)
    print >> sys.stderr, 'We were just alerted that Mixer channel %d is done' \
        % chan
    channel_is_done = 1

def still_playing():
    if TEST_MIX_CHANNELFINISHED:
        return not channel_is_done
    else:
        return Mix_Playing(0)

leftvol = 128
rightvol = 128
leftincr = -1
rightincr = 1
next_panning_update = 0
def do_panning_update():
    global leftvol, rightvol, leftincr, rightincr, next_panning_update
    if SDL_GetTicks() >= next_panning_update:
        Mix_SetPanning(0, leftvol, rightvol)
        if leftvol == 255 or leftvol == 0:
            if leftvol == 255:
                print 'All the way in the left speaker.'
            leftincr *= -1

        if rightvol == 255 or rightvol == 0:
            if rightvol == 255:
                print 'All the way in the right speaker.'
            rightincr *= -1

        leftvol += leftincr
        rightvol += rightincr
        next_panning_update = SDL_GetTicks() + 10

distance = 1
distincr = 1
next_distance_update = 0
def do_distance_update():
    global distance, distincr, next_distance_update
    if SDL_GetTicks() >= next_distance_update:
        Mix_SetDistance(0, distance)
        if distance == 0:
            print 'Distance at nearest point'
            distincr *= -1
        elif distance == 255:
            print 'Distance at furthest point'
            distincr *= -1

        distance += distincr
        next_distance_update = SDL_GetTicks() + 15

angle = 0
angleincr = 1
next_position_update = 0
def do_position_update():
    global distance, distincr, angle, angleincr, next_position_update

    if SDL_GetTicks() >= next_position_update:
        Mix_SetPosition(0, angle, distance)
        if angle == 0:
            print 'Due north; now rotating clockwise...'
            angleincr = 1
        elif angle == 360:
            print 'Due north; now rotating counter-clockwise...'
            angleincr = -1

        distance += distincr

        if distance < 0:
            distance = 0
            distincr = 3
            print 'Distance is very, very near.  Stepping away by threes...'
        elif distance > 255:
            distance = 255
            distincr = -3
            print 'Distance is very, very far.  Stepping towards by threes...'

        angle += angleincr

        next_position_update = SDL_GetTicks() + 30

def flip_sample(wave):
    opened, rate, format, channels = Mix_QuerySpec()
    incr = (format & 0xff) * channels
    if incr == 8:
        buf = wave.abuf.as_bytes()
    elif incr == 16:
        buf = wave.abuf.as_int16()
    elif incr == 32:
        buf = wave.abuf.as_int32()
    else:
        raise 'Unhandled format in sample flipping'

    # SDL_array doesn't have a reverse method, but list does.  Create a
    # list of the array by slicing.  Reverse the list.  Assign it to the
    # buffer with another slice.
    reversed = buf[:]
    reversed.reverse()
    buf[:] = reversed

def usage():
    print >> sys.stderr, 'Usage: %s [-8] [-r rate] [-c channels] [-f] [-F]\
 [-l] [-m] <wavefile>' % sys.argv[0]

if __name__ == '__main__':
    audio_rate = MIX_DEFAULT_FREQUENCY
    audio_format = MIX_DEFAULT_FORMAT
    audio_channels = 2
    loops = 0
    reverse_stereo = 0
    reverse_sample = 0

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
        elif arg == '-c':
            i += 1
            audio_channels = int(sys.argv[i])
        elif arg == '-l':
            loops = -1
        elif arg == '-8':
            audio_format = AUDIO_U8
        elif arg == '-f':
            reverse_stereo = 1
        elif arg == '-F':
            reverse_sample = 1
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
    print 'Opened audio at %d Hz %d bit %s' % \
        (audio_rate, audio_format & 0xff, channels_s),
    if loops:
        print ' (looping)',
    print

    if TEST_MIX_VERSIONS:
        test_versions()

    wave = Mix_LoadWAV(sys.argv[i])
    if reverse_sample:
        flip_sample(wave)
        pass

    if TEST_MIX_CHANNELFINISHED:
        Mix_ChannelFinished(channel_complete_callback)

    if reverse_stereo:
        Mix_SetReverseStereo(MIX_CHANNEL_POST, reverse_stereo)

    Mix_PlayChannel(0, wave, loops)

    while still_playing():
        if TEST_MIX_PANNING:
            do_panning_update()
        if TEST_MIX_DISTANCE:
            do_distance_update()
        if TEST_MIX_POSITION:
            do_position_update()

        SDL_Delay(1)

    Mix_FreeChunk(wave)
    Mix_CloseAudio()
    SDL_Quit()
