# cython: language_level=2
#

from sdl2 cimport *

cdef extern from "SDL.h" nogil:
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

cdef class AudioDevice:
    cdef SDL_AudioDeviceID _deviceid
    cdef SDL_AudioSpec desired
    cdef SDL_AudioSpec obtained
    cdef int _iscapture
    cdef object _callback
    cdef object _devicename
