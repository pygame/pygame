import pygame as pg
import time

if pg.get_sdl_version()[0] < 2:
    raise SystemExit('This example requires pygame 2 and SDL2.')

from pygame._sdl2 import (
    get_audio_device_name,
    get_num_audio_devices,
    AudioDevice,
    AUDIO_F32,
    AUDIO_ALLOW_FORMAT_CHANGE
)

pg.mixer.pre_init(44100, 32, 2, 512)
pg.init()

# init_subsystem(INIT_AUDIO)
names = [get_audio_device_name(x, 1) for x in range(get_num_audio_devices(1))]
print(names)

iscapture = 1
sounds = []
sound_chunks = []

def callback(audiodevice, audiomemoryview):
    """ This is called in the sound thread.

    Note, that the frequency and such you request may not be what you get.
    """
    # print(type(audiomemoryview), len(audiomemoryview))
    # print(audiodevice)
    sound_chunks.append(bytes(audiomemoryview))


audio = AudioDevice(
    devicename=names[0],
    iscapture=1,
    frequency=44100,
    audioformat=AUDIO_F32,
    numchannels=2,
    chunksize=512,
    allowed_changes=AUDIO_ALLOW_FORMAT_CHANGE,
    callback=callback,
)
# start recording.
audio.pause(0)

print('recording with :%s:' % names[0])
time.sleep(5)


print('Turning data into a pygame.mixer.Sound')
sound = pg.mixer.Sound(buffer=b''.join(sound_chunks))

print('playing back recorded sound')
sound.play()
time.sleep(5)
