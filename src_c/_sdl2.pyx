# cython: language_level=2
#
from libc.stdint cimport *
from libc.string cimport memset

from libc.stdio cimport *

cdef extern from "SDL.h" nogil:

    ctypedef int8_t Sint8
    ctypedef uint8_t Uint8
    ctypedef int16_t Sint16
    ctypedef uint16_t Uint16
    ctypedef int32_t Sint32
    ctypedef uint32_t Uint32
    ctypedef int64_t Sint64
    ctypedef uint64_t Uint64

    # https://wiki.libsdl.org/SDL_OpenAudioDevice
    # https://wiki.libsdl.org/SDL_CloseAudioDevice
    # https://wiki.libsdl.org/SDL_AudioSpec
    # https://wiki.libsdl.org/SDL_AudioFormat

    ctypedef Uint32 SDL_AudioDeviceID
    ctypedef Uint16 SDL_AudioFormat
    ctypedef void (*SDL_AudioCallback)(void *userdata, Uint8 *stream, int len)

    ctypedef struct SDL_AudioSpec:
        int freq
        SDL_AudioFormat format
        Uint8 channels
        Uint8 silence
        Uint16 samples
        Uint16 padding
        Uint32 size
        SDL_AudioCallback callback
        void *userdata

    int SDL_OpenAudio(SDL_AudioSpec *desired, SDL_AudioSpec *obtained)

    int SDL_GetNumAudioDevices(int iscapture)

    const char *SDL_GetAudioDeviceName(int index, int iscapture)

    SDL_AudioDeviceID SDL_OpenAudioDevice(const char *device, int iscapture, const SDL_AudioSpec *desired, SDL_AudioSpec *obtained, int allowed_changes)

    cdef int _SDL_AUDIO_ALLOW_FREQUENCY_CHANGE "SDL_AUDIO_ALLOW_FREQUENCY_CHANGE"
    cdef int _SDL_AUDIO_ALLOW_FORMAT_CHANGE "SDL_AUDIO_ALLOW_FORMAT_CHANGE"
    cdef int _SDL_AUDIO_ALLOW_CHANNELS_CHANGE "SDL_AUDIO_ALLOW_CHANNELS_CHANGE"
    cdef int _SDL_AUDIO_ALLOW_ANY_CHANGE "SDL_AUDIO_ALLOW_ANY_CHANGE"


    # https://wiki.libsdl.org/SDL_PauseAudioDevice
    void SDL_PauseAudioDevice(SDL_AudioDeviceID dev, int pause_on)
    void SDL_CloseAudioDevice(SDL_AudioDeviceID dev)

    cdef Uint16 _AUDIO_U8 "AUDIO_U8"
    cdef Uint16 _AUDIO_S8 "AUDIO_S8"
    cdef Uint16 _AUDIO_U16LSB "AUDIO_U16LSB"
    cdef Uint16 _AUDIO_S16LSB "AUDIO_S16LSB"
    cdef Uint16 _AUDIO_U16MSB "AUDIO_U16MSB"
    cdef Uint16 _AUDIO_S16MSB "AUDIO_S16MSB"
    cdef Uint16 _AUDIO_U16 "AUDIO_U16"
    cdef Uint16 _AUDIO_S16 "AUDIO_S16"
    cdef Uint16 _AUDIO_S32LSB "AUDIO_S32LSB"
    cdef Uint16 _AUDIO_S32MSB "AUDIO_S32MSB"
    cdef Uint16 _AUDIO_S32 "AUDIO_S32"
    cdef Uint16 _AUDIO_F32LSB "AUDIO_F32LSB"
    cdef Uint16 _AUDIO_F32MSB "AUDIO_F32MSB"
    cdef Uint16 _AUDIO_F32 "AUDIO_F32"


    const char *SDL_GetError()


    # https://wiki.libsdl.org/SDL_InitSubSystem
    # https://wiki.libsdl.org/SDL_QuitSubSystem
    # https://wiki.libsdl.org/SDL_WasInit
    int SDL_InitSubSystem(Uint32 flags)
    void SDL_QuitSubSystem(Uint32 flags)
    Uint32 SDL_WasInit(Uint32 flags)


    cdef int _SDL_INIT_TIMER "SDL_INIT_TIMER"
    cdef int _SDL_INIT_AUDIO "SDL_INIT_AUDIO"
    cdef int _SDL_INIT_VIDEO "SDL_INIT_VIDEO"
    cdef int _SDL_INIT_JOYSTICK "SDL_INIT_JOYSTICK"
    cdef int _SDL_INIT_HAPTIC "SDL_INIT_HAPTIC"
    cdef int _SDL_INIT_GAMECONTROLLER "SDL_INIT_GAMECONTROLLER"
    cdef int _SDL_INIT_EVENTS "SDL_INIT_EVENTS"
    cdef int _SDL_INIT_SENSOR "SDL_INIT_SENSOR"
    cdef int _SDL_INIT_NOPARACHUTE "SDL_INIT_NOPARACHUTE"
    cdef int _SDL_INIT_EVERYTHING "SDL_INIT_EVERYTHING"


# expose constants to python.
AUDIO_U8 = _AUDIO_U8
AUDIO_S8 = _AUDIO_S8
AUDIO_U16LSB = _AUDIO_U16LSB
AUDIO_S16LSB = _AUDIO_S16LSB
AUDIO_U16MSB = _AUDIO_U16MSB
AUDIO_S16MSB = _AUDIO_S16MSB
AUDIO_U16 = _AUDIO_U16
AUDIO_S16 = _AUDIO_S16
AUDIO_S32LSB = _AUDIO_S32LSB
AUDIO_S32MSB = _AUDIO_S32MSB
AUDIO_S32 = _AUDIO_S32
AUDIO_F32LSB = _AUDIO_F32LSB
AUDIO_F32MSB = _AUDIO_F32MSB
AUDIO_F32 = _AUDIO_F32
# So we can get the audio formats as string.
_audio_format_str = {
    AUDIO_U8: "AUDIO_U8",
    AUDIO_S8: "AUDIO_S8",
    AUDIO_U16LSB: "AUDIO_U16LSB",
    AUDIO_S16LSB: "AUDIO_S16LSB",
    AUDIO_U16MSB: "AUDIO_U16MSB",
    AUDIO_S16MSB: "AUDIO_S16MSB",
    AUDIO_U16: "AUDIO_U16",
    AUDIO_S16: "AUDIO_S16",
    AUDIO_S32LSB: "AUDIO_S32LSB",
    AUDIO_S32MSB: "AUDIO_S32MSB",
    AUDIO_S32: "AUDIO_S32",
    AUDIO_F32LSB: "AUDIO_F32LSB",
    AUDIO_F32MSB: "AUDIO_F32MSB",
    AUDIO_F32: "AUDIO_F32",
}


# for SDL_OpenAudioDevice.
AUDIO_ALLOW_FREQUENCY_CHANGE = _SDL_AUDIO_ALLOW_FREQUENCY_CHANGE
AUDIO_ALLOW_FORMAT_CHANGE = _SDL_AUDIO_ALLOW_FORMAT_CHANGE
AUDIO_ALLOW_CHANNELS_CHANGE = _SDL_AUDIO_ALLOW_CHANNELS_CHANGE
AUDIO_ALLOW_ANY_CHANGE = _SDL_AUDIO_ALLOW_ANY_CHANGE





# pygame.error
class error(RuntimeError):
    def __init__(self, message=None):
        if message is None:
            message = SDL_GetError().decode('utf8')
        RuntimeError.__init__(self, message)




# for init_subsystem. Expose variables to python.
INIT_TIMER = _SDL_INIT_TIMER
INIT_AUDIO = _SDL_INIT_AUDIO
INIT_VIDEO = _SDL_INIT_VIDEO
INIT_JOYSTICK = _SDL_INIT_JOYSTICK
INIT_HAPTIC = _SDL_INIT_HAPTIC
INIT_GAMECONTROLLER = _SDL_INIT_GAMECONTROLLER
INIT_EVENTS = _SDL_INIT_EVENTS
# INIT_SENSOR = _SDL_INIT_SENSOR
INIT_NOPARACHUTE = _SDL_INIT_NOPARACHUTE
INIT_EVERYTHING = _SDL_INIT_EVERYTHING


# TODO: Not sure about exposing init_subsystem in pygame.
#       It would be useful if you wanted to use audio without SDL_mixer.

# https://wiki.libsdl.org/SDL_InitSubSystem
def init_subsystem(flags):
    """ Use this function to initialize specific subsystems.

    :param int flags: any of the flags used by.

        * INIT_TIMER timer subsystem
        * INIT_AUDIO audio subsystem
        * INIT_VIDEO video subsystem; automatically initializes the events subsystem
        * INIT_JOYSTICK joystick subsystem; automatically initializes the events subsystem
        * INIT_HAPTIC haptic (force feedback) subsystem
        * INIT_GAMECONTROLLER controller subsystem; automatically initializes the joystick subsystem
        * INIT_EVENTS events subsystem
        * INIT_EVERYTHING all of the above subsystems
        * INIT_NOPARACHUTE compatibility; this flag is ignored
    """
    if (SDL_InitSubSystem(flags) == -1):
        raise error()


# https://wiki.libsdl.org/SDL_GetNumAudioDevices
def get_num_audio_devices(iscapture):
    """ return the number of audio devices for playback or capture.

    :param int iscapture: if 0 return devices available for playback of audio.
                          If 1 return devices available for capture of audio.
    :return: the number of devices available.
    :rtype: int
    """
    devcount = SDL_GetNumAudioDevices(iscapture);
    if devcount == -1:
        raise error('Audio system not initialised')
    return devcount

# https://wiki.libsdl.org/SDL_GetAudioDeviceName
def get_audio_device_name(index, iscapture):
    """ A unique devicename is available for each available audio device.

    :param int index: index of the devices from 0 to get_num_audio_devices(iscapture)
    :param int iscapture: if 0 return devices available for playback of audio.
                          If 1 return devices available for capture of audio.

    :return: the devicename.
    :rtype: bytes
    """
    cdef const char * name
    name = SDL_GetAudioDeviceName(index, iscapture)
    if not name:
        raise error()
    return name


import traceback
cdef void recording_cb(void* userdata, Uint8* stream, int len) nogil:
    """ This is called in a thread made by SDL.
        So we need the python GIL to do python stuff.
    """
    cdef Uint8 [:] a_memoryview
    with gil:
        a_memoryview = <Uint8[:len]> stream
        try:
            # The userdata is the audio device.
            # The audio device is needed in some apps
            (<object>userdata).callback(<object>userdata, a_memoryview)
        except:
            traceback.print_exc()
            raise


cdef class AudioDevice:
    cdef SDL_AudioDeviceID _deviceid
    cdef SDL_AudioSpec desired
    cdef SDL_AudioSpec obtained
    cdef int _iscapture
    cdef object _callback
    cdef object _devicename

    def __cinit__(self):
        self._deviceid = 0
        self._iscapture = 0

    def __dealloc__(self):
        if self._deviceid:
            SDL_CloseAudioDevice(self._deviceid)

    def __init__(self,
                 devicename,
                 iscapture,
                 frequency,
                 audioformat,
                 numchannels,
                 chunksize,
                 allowed_changes,
                 callback):
        """ An AudioDevice is for sound playback and capture of 'sound cards'.

        :param bytes devicename: One of the device names from get_audio_device_name.
                                 If None is passed in, it uses the default audio device.
        :param int frequency: Number of samples per second. 44100, 22050, ...
        :param int audioformat: AUDIO_F32SYS, AUDIO_F32SYS, AUDIO_U16SYS, AUDIO_S16SYS, ...
        :param int numchannels: 2 if stereo, 1 if mono.
        :param int chunksize: number of samples buffered.

        :param allowed_changes: some drivers don't support all possible requested formats.
                                So you can tell it which ones yours support.
            * AUDIO_ALLOW_FREQUENCY_CHANGE
            * AUDIO_ALLOW_FORMAT_CHANGE
            * AUDIO_ALLOW_CHANNELS_CHANGE
            * AUDIO_ALLOW_ANY_CHANGE

            If your application can only handle one specific data format,
            pass a zero for allowed_changes and let SDL transparently handle any differences.

        :callback: a function which gets called with (audiodevice, memoryview).
                   memoryview is the audio data.
                   Use audiodevice.iscapture to see if it is incoming audio or outgoing.
                   The audiodevice also has the format of the memory.
        """
        memset(&self.desired, 0, sizeof(SDL_AudioSpec))
        self._iscapture = iscapture
        self._callback = callback
        self._devicename = devicename

        self.desired.freq = frequency;
        self.desired.format = audioformat;
        self.desired.channels = numchannels;
        self.desired.samples = chunksize;
        self.desired.callback = <SDL_AudioCallback>recording_cb;
        self.desired.userdata = <void*>self

        self._deviceid = SDL_OpenAudioDevice(
            devicename,
            self._iscapture,
            &self.desired,
            &self.obtained,
            allowed_changes
        )

        if self._deviceid == 0:
            raise error()

    @property
    def iscapture(self):
        """ is the AudioDevice for capturing audio?
        """
        return self._iscapture

    @property
    def deviceid(self):
        """ deviceid of the audio device relative to the devicename list.
        """
        return self._deviceid

    @property
    def devicename(self):
        """ devicename of the audio device from the devicename list.
        """
        return self._devicename

    @property
    def callback(self):
        """ called in the sound thread with (audiodevice, memoryview)
        """
        return self._callback

    @property
    def frequency(self):
        """ Number of samples per second. 44100, 22050, ...
        """
        return self.obtained.freq

    @property
    def audioformat(self):
        """ AUDIO_F32SYS, AUDIO_F32SYS, AUDIO_U16SYS, AUDIO_S16SYS, ...
        """
        return self.obtained.format

    @property
    def numchannels(self):
        """ 2 if stereo, 1 if mono.
        """
        return self.obtained.channels

    @property
    def chunksize(self):
        """ number of samples buffered.
        """
        return self.obtained.samples

    def __repr__(self):
        ret = "<AudioDevice("
        ret += "devicename=%s, " % self.devicename
        ret += "iscapture=%s, " % self.iscapture
        ret += "frequency=%s, " % self.frequency
        ret += "audioformat=%s, " % _audio_format_str[self.audioformat]
        ret += "numchannels=%s, " % self.numchannels
        ret += "chunksize=%s, " % self.chunksize
        ret += ")>"
        return ret

    def pause(self, int pause_on):
        """ Use this to pause and unpause audio playback on this device.

        :param int pause_on:
        """
        # https://wiki.libsdl.org/SDL_PauseAudioDevice
        if self._deviceid:
            SDL_PauseAudioDevice(self._deviceid, pause_on)

    def close(self):
        """ Use this to pause and unpause audio playback on this device.
        """
        # https://wiki.libsdl.org/SDL_CloseAudioDevice
        if self._deviceid:
            SDL_CloseAudioDevice(self._deviceid)
            self._deviceid = 0
