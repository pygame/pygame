from pygame._sdl2.sdl2 import error


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

# https://wiki.libsdl.org/SDL_GetNumAudioDevices
# https://wiki.libsdl.org/SDL_GetAudioDeviceName
def get_audio_device_names(iscapture = False):
    """ Returns a list of unique devicenames for each available audio device.

    :param bool iscapture: If False return devices available for playback.
                           If True return devices available for capture.

    :return: list of devicenames.
    :rtype: List[string]
    """

    cdef int count = SDL_GetNumAudioDevices(iscapture)
    if count == -1:
        raise error('Audio system not initialised')
    
    names = []
    for i in range(count):
        name = SDL_GetAudioDeviceName(i, iscapture)
        if not name:
            raise error()
        names.append(name.decode('utf8'))

    return names

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

        :param string devicename: One of the device names from get_audio_device_names.
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
        if not isinstance(devicename, str):
            raise TypeError("devicename must be a string")
        self._devicename = devicename

        self.desired.freq = frequency;
        self.desired.format = audioformat;
        self.desired.channels = numchannels;
        self.desired.samples = chunksize;
        self.desired.callback = <SDL_AudioCallback>recording_cb;
        self.desired.userdata = <void*>self

        self._deviceid = SDL_OpenAudioDevice(
            self._devicename.encode("utf-8"),
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
