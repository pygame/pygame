/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public
  License along with this library; if not, write to the Free
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

  Pete Shinners
  pete@shinners.org
*/

/*
 *  mixer module for pygame
 */
#define PYGAMEAPI_MIXER_INTERNAL
#include "pygame.h"

#include "pgcompat.h"

#include "doc/mixer_doc.h"

#include "mixer.h"

#define PyBUF_HAS_FLAG(f, F) (((f) & (F)) == (F))

#define CHECK_CHUNK_VALID(CHUNK, RET)                                      \
    if ((CHUNK) == NULL) {                                                 \
        PyErr_SetString(PyExc_RuntimeError,                                \
                        "__init__() was not called on Sound object so it " \
                        "failed to setup correctly.");                     \
        return (RET);                                                      \
    }

/* The SDL audio format constants are not defined for anything larger
   than 2 byte samples. Define our own. Low two bytes gives sample
   size in bytes. Higher bytes are flags.
*/
typedef Uint32 PG_sample_format_t;
const PG_sample_format_t PG_SAMPLE_SIGNED = 0x10000u;
const PG_sample_format_t PG_SAMPLE_NATIVE_ENDIAN = 0x20000u;
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
const PG_sample_format_t PG_SAMPLE_LITTLE_ENDIAN = 0x20000u;
const PG_sample_format_t PG_SAMPLE_BIG_ENDIAN = 0;
#else
const PG_sample_format_t PG_SAMPLE_LITTLE_ENDIAN = 0;
const PG_sample_format_t PG_SAMPLE_BIG_ENDIAN = 0x20000u;
#endif
const PG_sample_format_t PG_SAMPLE_CHAR_SIGN = (char)0xff > 0 ? 0 : 0x10000u;
#define PG_SAMPLE_SIZE(sf) ((sf)&0x0ffffu)
#define PG_IS_SAMPLE_SIGNED(sf) ((sf)&PG_SAMPLE_SIGNED != 0)
#define PG_IS_SAMPLE_NATIVE_ENDIAN(sf) ((sf)&PG_SAMPLE_NATIVE_ENDIAN != 0)
#define PG_IS_SAMPLE_LITTLE_ENDIAN(sf) \
    ((sf)&PG_SAMPLE_LITTLE_ENDIAN == PG_SAMPLE_LITTLE_ENDIAN)
#define PG_IS_SAMPLE_BIG_ENDIAN(sf) \
    ((sf)&PG_SAMPLE_BIG_ENDIAN == PG_SAMPLE_BIG_ENDIAN)

/* Since they are documented, the default init values are defined here
   rather than taken from SDL_mixer. It also means that the default
   size is defined in Pygame, rather than SDL AUDIO_xxx, terms.
 */
#define PYGAME_MIXER_DEFAULT_FREQUENCY 44100
#define PYGAME_MIXER_DEFAULT_SIZE -16
#define PYGAME_MIXER_DEFAULT_CHANNELS 2
#define PYGAME_MIXER_DEFAULT_CHUNKSIZE 512
#define PYGAME_MIXER_DEFAULT_ALLOWEDCHANGES \
    SDL_AUDIO_ALLOW_FREQUENCY_CHANGE | SDL_AUDIO_ALLOW_CHANNELS_CHANGE

static int
sound_init(PyObject *, PyObject *, PyObject *);

static PyTypeObject pgSound_Type;
static PyTypeObject pgChannel_Type;
static PyObject *
pgSound_New(Mix_Chunk *);
static PyObject *
pgChannel_New(int);
#define pgSound_Check(x) (Py_TYPE(x) == &pgSound_Type)
#define pgChannel_Check(x) (Py_TYPE(x) == &pgChannel_Type)

static int
snd_getbuffer(PyObject *, Py_buffer *, int);
static void
snd_releasebuffer(PyObject *, Py_buffer *);

static int request_frequency = PYGAME_MIXER_DEFAULT_FREQUENCY;
static int request_size = PYGAME_MIXER_DEFAULT_SIZE;
static int request_channels = PYGAME_MIXER_DEFAULT_CHANNELS;
static int request_chunksize = PYGAME_MIXER_DEFAULT_CHUNKSIZE;
static int request_allowedchanges = PYGAME_MIXER_DEFAULT_ALLOWEDCHANGES;
static char *request_devicename = NULL;

struct ChannelData {
    PyObject *sound;
    PyObject *queue;
    int endevent;
};
static struct ChannelData *channeldata = NULL;
static int numchanneldata = 0;

Mix_Music **mx_current_music;
Mix_Music **mx_queue_music;

static int
_format_itemsize(Uint16 format)
{
    int size = -1;
    switch (format) {
        case AUDIO_U8:
        case AUDIO_S8:
            size = 1;
            break;

        case AUDIO_U16LSB:
        case AUDIO_U16MSB:
        case AUDIO_S16LSB:
        case AUDIO_S16MSB:
            size = 2;
            break;
        case AUDIO_S32LSB:
        case AUDIO_S32MSB:
        case AUDIO_F32LSB:
        case AUDIO_F32MSB:
            size = 4;
            break;
        default:
            PyErr_Format(PyExc_SystemError,
                         "Pygame bug (mixer.Sound): unknown mixer format %d",
                         (int)format);
    }
    return size;
}

static PG_sample_format_t
_format_view_to_audio(Py_buffer *view)
{
    size_t fstr_len;
    int native_size = 0;
    int index = 0;
    PG_sample_format_t format = 0;

    if (!view->format) {
        /* Assume unsigned byte */
        return (PG_sample_format_t)sizeof(unsigned char);
    }
    fstr_len = strlen(view->format);
    if (fstr_len < 1 || fstr_len > 2) {
        PyErr_SetString(PyExc_ValueError, "Array has unsupported item format");
        return 0;
    }
    if (fstr_len == 1) {
        format |= PG_SAMPLE_NATIVE_ENDIAN;
        native_size = 1;
    }
    else {
        switch (view->format[index]) {
            case '@':
                native_size = 1;
                format |= PG_SAMPLE_NATIVE_ENDIAN;
                break;

            case '=':
                format |= PG_SAMPLE_NATIVE_ENDIAN;
                break;

            case '<':
                format |= PG_SAMPLE_LITTLE_ENDIAN;
                break;

            case '>':
            case '!':
                format |= PG_SAMPLE_BIG_ENDIAN;
                break;

            default:
                PyErr_SetString(PyExc_ValueError,
                                "Array has unsupported item format");
                return 0;
        }
        ++index;
    }
    switch (view->format[index]) {
        case 'c':
            format |= PG_SAMPLE_CHAR_SIGN;
            format += native_size ? sizeof(char) : 1;
            break;

        case 'b':
            format |= PG_SAMPLE_SIGNED;
            format += native_size ? sizeof(signed char) : 1;
            break;

        case 'B':
            format += native_size ? sizeof(unsigned char) : 1;
            break;

        case 'h':
            format |= PG_SAMPLE_SIGNED;
            format += native_size ? sizeof(short int) : 2;
            break;

        case 'H':
            format += native_size ? sizeof(unsigned short int) : 2;
            break;

        case 'i':
            format |= PG_SAMPLE_SIGNED;
            format += native_size ? sizeof(int) : 4;
            break;

        case 'I':
            format += native_size ? sizeof(unsigned int) : 4;
            break;

        case 'l':
            format |= PG_SAMPLE_SIGNED;
            format += native_size ? sizeof(long int) : 4;
            break;

        case 'L':
            format += native_size ? sizeof(unsigned long int) : 4;
            break;

        case 'f':
            format += native_size ? sizeof(float) : 4;
            break;

        case 'd':
            format += native_size ? sizeof(double) : 8;
            break;

        case 'q':
            format |= PG_SAMPLE_SIGNED;
            format += native_size ? sizeof(long long int) : 8;
            break;

        case 'Q':
            format += native_size ? sizeof(unsigned long long int) : 8;
            break;

        default:
            PyErr_Format(PyExc_ValueError,
                         "Array has unsupported item format '%s'",
                         view->format);
            return 0;
    }
    if (view->itemsize && PG_SAMPLE_SIZE(format) != view->itemsize) {
        PyErr_Format(PyExc_ValueError,
                     "Array item size %d does not match format '%s'",
                     (int)view->itemsize, view->format);
        return 0;
    }
    return format;
}

static void
_pg_push_mixer_event(int type, int code)
{
    pgEventObject *e;
    PyObject *dict, *dictcode;
    SDL_Event event;
    PyGILState_STATE gstate = PyGILState_Ensure();

    dict = PyDict_New();
    if (dict) {
        if (type >= PGE_USEREVENT && type < PG_NUMEVENTS) {
            dictcode = PyLong_FromLong(code);
            PyDict_SetItemString(dict, "code", dictcode);
            Py_DECREF(dictcode);
        }
        e = (pgEventObject *)pgEvent_New2(type, dict);
        Py_DECREF(dict);

        if (e) {
            pgEvent_FillUserEvent(e, &event);
            if (SDL_PushEvent(&event) <= 0)
                Py_DECREF(dict);
            Py_DECREF(e);
        }
    }
    PyGILState_Release(gstate);
}

static void
endsound_callback(int channel)
{
    if (channeldata) {
        if (channeldata[channel].endevent && SDL_WasInit(SDL_INIT_VIDEO))
            _pg_push_mixer_event(channeldata[channel].endevent, channel);

        if (channeldata[channel].queue) {
            PyGILState_STATE gstate = PyGILState_Ensure();
            int channelnum;
            Mix_Chunk *sound = pgSound_AsChunk(channeldata[channel].queue);
            Py_XDECREF(channeldata[channel].sound);
            channeldata[channel].sound = channeldata[channel].queue;
            channeldata[channel].queue = NULL;
            PyGILState_Release(gstate);
            channelnum = Mix_PlayChannelTimed(channel, sound, 0, -1);
            if (channelnum != -1)
                Mix_GroupChannel(channelnum, (int)(intptr_t)sound);
        }
        else {
            PyGILState_STATE gstate = PyGILState_Ensure();
            Py_XDECREF(channeldata[channel].sound);
            channeldata[channel].sound = NULL;
            PyGILState_Release(gstate);
            Mix_GroupChannel(channel, -1);
        }
    }
}

static PyObject *
import_music(void)
{
    PyObject *music = PyImport_ImportModule(IMPPREFIX "mixer_music");
    if (music == NULL) {
        PyErr_Clear();
        music = PyImport_ImportModule(RELATIVE_MODULE("mixer_music"));
    }
    return music;
}

static PyObject *
_init(int freq, int size, int channels, int chunk, char *devicename,
      int allowedchanges)
{
    Uint16 fmt = 0;
    int i;
    PyObject *music;
    char *drivername;

    if (!freq) {
        freq = request_frequency;
    }
    if (!size) {
        size = request_size;
    }

    if (allowedchanges == -1) {
        allowedchanges = request_allowedchanges;
    }

    if (!channels) {
        channels = request_channels;
    }
    if (allowedchanges & SDL_AUDIO_ALLOW_CHANNELS_CHANGE) {
        if (channels <= 1)
            channels = 1;
        else if (channels <= 3)
            channels = 2;
        else if (channels <= 5)
            channels = 4;
        else
            channels = 6;
    }
    else {
        switch (channels) {
            case 1:
            case 2:
            case 4:
            case 6:
                break;
            default:
                return RAISE(PyExc_ValueError,
                             "'channels' must be 1, 2, 4, or 6");
        }
    }

    if (!chunk) {
        chunk = request_chunksize;
    }

    if (!devicename) {
        devicename = request_devicename;
    }

    /* printf("size:%d:\n", size); */

    switch (size) {
        case 8:
            fmt = AUDIO_U8;
            break;
        case -8:
            fmt = AUDIO_S8;
            break;
        case 16:
            fmt = AUDIO_U16SYS;
            break;
        case -16:
            fmt = AUDIO_S16SYS;
            break;
        case 32:
            fmt = AUDIO_F32SYS;
            break;
        default:
            PyErr_Format(PyExc_ValueError, "unsupported size %i", size);
            return NULL;
    }

    /* printf("size:%d:\n", size); */

    /*make chunk a power of 2*/
    for (i = 0; 1 << i < chunk; ++i)
        ;                     /*yes, semicolon on for loop*/
    chunk = MAX(1 << i, 256); /*do this after for loop exits*/

    if (!SDL_WasInit(SDL_INIT_AUDIO)) {
        if (!channeldata) { /*should always be null*/
            channeldata = (struct ChannelData *)malloc(
                sizeof(struct ChannelData) * MIX_CHANNELS);
            if (!channeldata) {
                return PyErr_NoMemory();
            }
            numchanneldata = MIX_CHANNELS;
            for (i = 0; i < numchanneldata; ++i) {
                channeldata[i].sound = NULL;
                channeldata[i].queue = NULL;
                channeldata[i].endevent = 0;
            }
        }

        /* Compatibility:
            pulse and dsound audio drivers were renamed in SDL2,
            and we don't want it to fail.
        */
        drivername = SDL_getenv("SDL_AUDIODRIVER");
        if (drivername && SDL_strncasecmp("pulse", drivername,
                                          SDL_strlen(drivername)) == 0) {
            SDL_setenv("SDL_AUDIODRIVER", "pulseaudio", 1);
        }
        else if (drivername && SDL_strncasecmp("dsound", drivername,
                                               SDL_strlen(drivername)) == 0) {
            SDL_setenv("SDL_AUDIODRIVER", "directsound", 1);
        }

        if (SDL_InitSubSystem(SDL_INIT_AUDIO))
            return RAISE(pgExc_SDLError, SDL_GetError());

/* This scary looking block is the expansion of
 * SDL_MIXER_VERSION_ATLEAST(2, 0, 2), but SDL_MIXER_VERSION_ATLEAST is new in
 * 2.0.2, and we currently aim to support down to 2.0.0 */
#if ((SDL_MIXER_MAJOR_VERSION >= 2) &&                                \
     (SDL_MIXER_MAJOR_VERSION > 2 || SDL_MIXER_MINOR_VERSION >= 0) && \
     (SDL_MIXER_MAJOR_VERSION > 2 || SDL_MIXER_MINOR_VERSION > 0 ||   \
      SDL_MIXER_PATCHLEVEL >= 2))
        if (Mix_OpenAudioDevice(freq, fmt, channels, chunk, devicename,
                                allowedchanges) == -1) {
#else
        if (Mix_OpenAudio(freq, fmt, channels, chunk) == -1) {
#endif
            SDL_QuitSubSystem(SDL_INIT_AUDIO);
            return RAISE(pgExc_SDLError, SDL_GetError());
            ;
        }
        Mix_ChannelFinished(endsound_callback);
        Mix_VolumeMusic(127);
    }

    mx_current_music = NULL;
    mx_queue_music = NULL;

    music = import_music();
    if (music) {
        PyObject *ptr;
        ptr = PyObject_GetAttrString(music, "_MUSIC_POINTER");
        if (ptr) {
            mx_current_music =
                (Mix_Music **)PyCapsule_GetPointer(ptr,
                                                   "pygame.music_mixer."
                                                   "_MUSIC_POINTER");
            if (!mx_current_music) {
                PyErr_Clear();
            }
        }
        else {
            PyErr_Clear();
        }

        ptr = PyObject_GetAttrString(music, "_QUEUE_POINTER");
        if (ptr) {
            mx_queue_music =
                (Mix_Music **)PyCapsule_GetPointer(ptr,
                                                   "pygame.music_mixer."
                                                   "_QUEUE_POINTER");
            if (!mx_queue_music) {
                PyErr_Clear();
            }
        }
        else {
            PyErr_Clear();
        }

        Py_DECREF(music);
    }
    else {
        PyErr_Clear();
    }
    Py_RETURN_NONE;
}

static PyObject *
pgMixer_AutoInit(PyObject *self, PyObject *_null)
{
    /* Return init with defaults */
    return _init(0, 0, 0, 0, NULL, -1);
}

static PyObject *
mixer_quit(PyObject *self, PyObject *_null)
{
    int i;
    if (SDL_WasInit(SDL_INIT_AUDIO)) {
        Py_BEGIN_ALLOW_THREADS;
        Mix_HaltMusic();
        Py_END_ALLOW_THREADS;

        if (channeldata) {
            for (i = 0; i < numchanneldata; ++i) {
                Py_XDECREF(channeldata[i].sound);
                Py_XDECREF(channeldata[i].queue);
            }
            free(channeldata);
            channeldata = NULL;
            numchanneldata = 0;
        }

        if (mx_current_music) {
            if (*mx_current_music) {
                Py_BEGIN_ALLOW_THREADS;
                Mix_FreeMusic(*mx_current_music);
                Py_END_ALLOW_THREADS;
                *mx_current_music = NULL;
            }
            mx_current_music = NULL;
        }
        if (mx_queue_music) {
            if (*mx_queue_music) {
                Py_BEGIN_ALLOW_THREADS;
                Mix_FreeMusic(*mx_queue_music);
                Py_END_ALLOW_THREADS;
                *mx_queue_music = NULL;
            }
            mx_queue_music = NULL;
        }

        Py_BEGIN_ALLOW_THREADS;
        Mix_CloseAudio();
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        Py_END_ALLOW_THREADS;
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_mixer_init(PyObject *self, PyObject *args, PyObject *keywds)
{
    int freq = 0, size = 0, channels = 0, chunk = 0, allowedchanges = -1;
    char *devicename = NULL;

    static char *kwids[] = {"frequency", "size",       "channels",
                            "buffer",    "devicename", "allowedchanges",
                            NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|iiiizi", kwids, &freq,
                                     &size, &channels, &chunk, &devicename,
                                     &allowedchanges)) {
        return NULL;
    }
    return _init(freq, size, channels, chunk, devicename, allowedchanges);
}

static PyObject *
pg_mixer_get_init(PyObject *self, PyObject *_null)
{
    int freq, channels, realform;
    Uint16 format;

    if (!SDL_WasInit(SDL_INIT_AUDIO))
        Py_RETURN_NONE;

    if (!Mix_QuerySpec(&freq, &format, &channels))
        Py_RETURN_NONE;

    // create a signed or unsigned number of bits per sample
    realform = SDL_AUDIO_BITSIZE(format);
    if (SDL_AUDIO_ISSIGNED(format)) {
        realform = -realform;
    }
    return Py_BuildValue("(iii)", freq, realform, channels);
}

static PyObject *
pre_init(PyObject *self, PyObject *args, PyObject *keywds)
{
    static char *kwids[] = {"frequency", "size",       "channels",
                            "buffer",    "devicename", "allowedchanges",
                            NULL};

    request_frequency = 0;
    request_size = 0;
    request_channels = 0;
    request_chunksize = 0;
    request_devicename = NULL;
    request_allowedchanges = -1;
    if (!PyArg_ParseTupleAndKeywords(
            args, keywds, "|iiiizi", kwids, &request_frequency, &request_size,
            &request_channels, &request_chunksize, &request_devicename,
            &request_allowedchanges))
        return NULL;
    if (!request_frequency) {
        request_frequency = PYGAME_MIXER_DEFAULT_FREQUENCY;
    }
    if (!request_size) {
        request_size = PYGAME_MIXER_DEFAULT_SIZE;
    }
    if (!request_channels) {
        request_channels = PYGAME_MIXER_DEFAULT_CHANNELS;
    }
    if (!request_chunksize) {
        request_chunksize = PYGAME_MIXER_DEFAULT_CHUNKSIZE;
    }
    if (request_allowedchanges == -1) {
        request_allowedchanges = PYGAME_MIXER_DEFAULT_ALLOWEDCHANGES;
    }
    Py_RETURN_NONE;
}

/* sound object methods */

static PyObject *
pgSound_Play(PyObject *self, PyObject *args, PyObject *kwargs)
{
    Mix_Chunk *chunk = pgSound_AsChunk(self);
    int channelnum = -1;
    int loops = 0, playtime = -1, fade_ms = 0;

    CHECK_CHUNK_VALID(chunk, NULL);

    char *kwids[] = {"loops", "maxtime", "fade_ms", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iii", kwids, &loops,
                                     &playtime, &fade_ms))
        return NULL;

    Py_BEGIN_ALLOW_THREADS;
    if (fade_ms > 0) {
        channelnum =
            Mix_FadeInChannelTimed(-1, chunk, loops, fade_ms, playtime);
    }
    else {
        channelnum = Mix_PlayChannelTimed(-1, chunk, loops, playtime);
    }
    Py_END_ALLOW_THREADS;
    if (channelnum == -1)
        Py_RETURN_NONE;

    Py_XDECREF(channeldata[channelnum].sound);
    Py_XDECREF(channeldata[channelnum].queue);
    channeldata[channelnum].queue = NULL;
    channeldata[channelnum].sound = self;
    Py_INCREF(self);

    // make sure volume on this arbitrary channel is set to full
    Mix_Volume(channelnum, 128);

    Py_BEGIN_ALLOW_THREADS;
    Mix_GroupChannel(channelnum, (int)(intptr_t)chunk);
    Py_END_ALLOW_THREADS;

    return pgChannel_New(channelnum);
}

static PyObject *
snd_get_num_channels(PyObject *self, PyObject *_null)
{
    Mix_Chunk *chunk = pgSound_AsChunk(self);

    CHECK_CHUNK_VALID(chunk, NULL);

    MIXER_INIT_CHECK();
    return PyLong_FromLong(Mix_GroupCount((int)(intptr_t)chunk));
}

static PyObject *
snd_fadeout(PyObject *self, PyObject *args)
{
    Mix_Chunk *chunk = pgSound_AsChunk(self);
    int _time;

    CHECK_CHUNK_VALID(chunk, NULL);

    if (!PyArg_ParseTuple(args, "i", &_time))
        return NULL;

    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    Mix_FadeOutGroup((int)(intptr_t)chunk, _time);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject *
snd_stop(PyObject *self, PyObject *_null)
{
    Mix_Chunk *chunk = pgSound_AsChunk(self);

    CHECK_CHUNK_VALID(chunk, NULL);

    MIXER_INIT_CHECK();
    Py_BEGIN_ALLOW_THREADS;
    Mix_HaltGroup((int)(intptr_t)chunk);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject *
snd_set_volume(PyObject *self, PyObject *args)
{
    Mix_Chunk *chunk = pgSound_AsChunk(self);
    float volume;

    CHECK_CHUNK_VALID(chunk, NULL);

    if (!PyArg_ParseTuple(args, "f", &volume))
        return NULL;

    MIXER_INIT_CHECK();

    Mix_VolumeChunk(chunk, (int)(volume * 128));
    Py_RETURN_NONE;
}

static PyObject *
snd_get_volume(PyObject *self, PyObject *_null)
{
    Mix_Chunk *chunk = pgSound_AsChunk(self);

    CHECK_CHUNK_VALID(chunk, NULL);

    int volume;
    MIXER_INIT_CHECK();

    volume = Mix_VolumeChunk(chunk, -1);
    return PyFloat_FromDouble(volume / 128.0);
}

static PyObject *
snd_get_length(PyObject *self, PyObject *_null)
{
    Mix_Chunk *chunk = pgSound_AsChunk(self);

    CHECK_CHUNK_VALID(chunk, NULL);

    int freq, channels, mixerbytes, numsamples;
    Uint16 format;
    MIXER_INIT_CHECK();

    Mix_QuerySpec(&freq, &format, &channels);
    if (format == AUDIO_S8 || format == AUDIO_U8)
        mixerbytes = 1;
    else if (format == AUDIO_F32LSB || format == AUDIO_F32MSB) {
        mixerbytes = 4;
    }
    else
        mixerbytes = 2;
    numsamples = chunk->alen / mixerbytes / channels;

    return PyFloat_FromDouble((float)numsamples / (float)freq);
}

static PyObject *
snd_get_raw(PyObject *self, PyObject *_null)
{
    Mix_Chunk *chunk = pgSound_AsChunk(self);

    CHECK_CHUNK_VALID(chunk, NULL);
    MIXER_INIT_CHECK();

    return PyBytes_FromStringAndSize((const char *)chunk->abuf,
                                     (Py_ssize_t)chunk->alen);
}

static PyObject *
snd_get_arraystruct(PyObject *self, void *closure)
{
    Py_buffer view;
    PyObject *cobj;

    if (snd_getbuffer(self, &view, PyBUF_RECORDS)) {
        return 0;
    }
    cobj = pgBuffer_AsArrayStruct(&view);
    snd_releasebuffer(view.obj, &view);
    Py_XDECREF(view.obj);
    return cobj;
}

static PyObject *
snd_get_arrayinterface(PyObject *self, void *closure)
{
    Py_buffer view;
    PyObject *dict;

    if (snd_getbuffer(self, &view, PyBUF_RECORDS)) {
        return 0;
    }
    dict = pgBuffer_AsArrayInterface(&view);
    snd_releasebuffer(self, &view);
    Py_DECREF(self);
    return dict;
}

static PyObject *
snd_get_samples_address(PyObject *self, PyObject *closure)
{
    Mix_Chunk *chunk = pgSound_AsChunk(self);

    CHECK_CHUNK_VALID(chunk, NULL);

    MIXER_INIT_CHECK();

#if SIZEOF_VOID_P > SIZEOF_LONG
    return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG)chunk->abuf);
#else
    return PyLong_FromUnsignedLong((unsigned long)chunk->abuf);
#endif
}

PyMethodDef sound_methods[] = {
    {"play", (PyCFunction)pgSound_Play, METH_VARARGS | METH_KEYWORDS,
     DOC_SOUNDPLAY},
    {"get_num_channels", snd_get_num_channels, METH_NOARGS,
     DOC_SOUNDGETNUMCHANNELS},
    {"fadeout", snd_fadeout, METH_VARARGS, DOC_SOUNDFADEOUT},
    {"stop", snd_stop, METH_NOARGS, DOC_SOUNDSTOP},
    {"set_volume", snd_set_volume, METH_VARARGS, DOC_SOUNDSETVOLUME},
    {"get_volume", snd_get_volume, METH_NOARGS, DOC_SOUNDGETVOLUME},
    {"get_length", snd_get_length, METH_NOARGS, DOC_SOUNDGETLENGTH},
    {"get_raw", snd_get_raw, METH_NOARGS, DOC_SOUNDGETRAW},
    {NULL, NULL, 0, NULL}};

static PyGetSetDef sound_getset[] = {
    {"__array_struct__", snd_get_arraystruct, NULL, "Version 3", NULL},
    {"__array_interface__", snd_get_arrayinterface, NULL, "Version 3", NULL},
    {"_samples_address", (getter)snd_get_samples_address, NULL,
     "samples buffer address (readonly)", NULL},
    {NULL, NULL, NULL, NULL, NULL}};

/*buffer protocol*/

/*
snd_buffer_iteminfo converts between SDL and python format constants.

https://wiki.libsdl.org/SDL_AudioSpec
https://docs.python.org/3/library/struct.html#format-characters

returns:
    -1 on error, else 0.
    format: buffer string showing the format.
    itemsize: bytes for each item.
*/
static int
snd_buffer_iteminfo(char **format, Py_ssize_t *itemsize, int *channels)
{
    static char fmt_AUDIO_U8[] = "B";
    static char fmt_AUDIO_S8[] = "b";
    static char fmt_AUDIO_U16SYS[] = "=H";
    static char fmt_AUDIO_S16SYS[] = "=h";

    static char fmt_AUDIO_S32LSB[] = "<i";
    static char fmt_AUDIO_S32MSB[] = ">i";
    static char fmt_AUDIO_F32LSB[] = "<f";
    static char fmt_AUDIO_F32MSB[] = ">f";

    int freq = 0;
    Uint16 mixer_format = 0;

    Mix_QuerySpec(&freq, &mixer_format, channels);

    switch (mixer_format) {
        case AUDIO_U8:
            *format = fmt_AUDIO_U8;
            *itemsize = 1;
            return 0;

        case AUDIO_S8:
            *format = fmt_AUDIO_S8;
            *itemsize = 1;
            return 0;

        case AUDIO_U16SYS:
            *format = fmt_AUDIO_U16SYS;
            *itemsize = 2;
            return 0;

        case AUDIO_S16SYS:
            *format = fmt_AUDIO_S16SYS;
            *itemsize = 2;
            return 0;

        case AUDIO_S32LSB:
            *format = fmt_AUDIO_S32LSB;
            *itemsize = 4;
            return 0;

        case AUDIO_S32MSB:
            *format = fmt_AUDIO_S32MSB;
            *itemsize = 4;
            return 0;

        case AUDIO_F32LSB:
            *format = fmt_AUDIO_F32LSB;
            *itemsize = 4;
            return 0;

        case AUDIO_F32MSB:
            *format = fmt_AUDIO_F32MSB;
            *itemsize = 4;
            return 0;
    }

    PyErr_Format(PyExc_SystemError,
                 "Pygame bug (mixer.Sound): unknown mixer format %d",
                 (int)mixer_format);
    return -1;
}

static int
snd_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    Mix_Chunk *chunk = pgSound_AsChunk(obj);
    int channels;
    char *format;
    int ndim = 0;
    Py_ssize_t *shape = 0;
    Py_ssize_t *strides = 0;
    Py_ssize_t itemsize;
    Py_ssize_t samples;

    CHECK_CHUNK_VALID(chunk, -1);

    view->obj = 0;
    if (snd_buffer_iteminfo(&format, &itemsize, &channels)) {
        return -1;
    }
    if (channels != 1 && PyBUF_HAS_FLAG(flags, PyBUF_F_CONTIGUOUS)) {
        PyErr_SetString(pgExc_BufferError,
                        "polyphonic sound is not Fortran contiguous");
        return -1;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_ND)) {
        ndim = channels > 1 ? 2 : 1;
        samples = chunk->alen / (itemsize * channels);
        shape = PyMem_New(Py_ssize_t, 2 * ndim);
        if (!shape) {
            PyErr_NoMemory();
            return -1;
        }
        shape[ndim - 1] = channels;
        shape[0] = samples;
        if (PyBUF_HAS_FLAG(flags, PyBUF_STRIDES)) {
            strides = shape + ndim;
            strides[0] = itemsize * channels;
            strides[ndim - 1] = itemsize;
        }
    }
    Py_INCREF(obj);
    view->obj = obj;
    view->buf = chunk->abuf;
    view->len = (Py_ssize_t)chunk->alen;
    view->readonly = 0;
    view->itemsize = itemsize;
    view->format = PyBUF_HAS_FLAG(flags, PyBUF_FORMAT) ? format : 0;
    view->ndim = ndim;
    view->shape = shape;
    view->strides = strides;
    view->suboffsets = 0;
    view->internal = shape;
    return 0;
}

static void
snd_releasebuffer(PyObject *obj, Py_buffer *view)
{
    if (view->internal) {
        PyMem_Free(view->internal);
        view->internal = 0;
    }
}

static PyBufferProcs sound_as_buffer[] = {{snd_getbuffer, snd_releasebuffer}};

/*sound object internals*/
static void
sound_dealloc(pgSoundObject *self)
{
    Mix_Chunk *chunk = pgSound_AsChunk((PyObject *)self);
    if (chunk) {
        Py_BEGIN_ALLOW_THREADS;
        Mix_FreeChunk(chunk);
        Py_END_ALLOW_THREADS;
    }
    if (self->mem)
        PyMem_Free(self->mem);
    if (self->weakreflist)
        PyObject_ClearWeakRefs((PyObject *)self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyTypeObject pgSound_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "Sound",
    .tp_basicsize = sizeof(pgSoundObject),
    .tp_dealloc = (destructor)sound_dealloc,
    .tp_as_buffer = sound_as_buffer,
    .tp_flags =
        (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_NEWBUFFER),
    .tp_doc = DOC_PYGAMEMIXERSOUND,
    .tp_weaklistoffset = offsetof(pgSoundObject, weakreflist),
    .tp_methods = sound_methods,
    .tp_getset = sound_getset,
    .tp_init = sound_init,
    .tp_new = PyType_GenericNew,
};

/* channel object methods */
static PyObject *
chan_play(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int channelnum = pgChannel_AsInt(self);
    PyObject *sound;
    Mix_Chunk *chunk;
    int loops = 0, playtime = -1, fade_ms = 0;

    char *kwids[] = {"Sound", "loops", "maxtime", "fade_ms", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|iii", kwids,
                                     &pgSound_Type, &sound, &loops, &playtime,
                                     &fade_ms))
        return NULL;
    chunk = pgSound_AsChunk(sound);
    CHECK_CHUNK_VALID(chunk, NULL);

    Py_BEGIN_ALLOW_THREADS;
    if (fade_ms > 0) {
        channelnum = Mix_FadeInChannelTimed(channelnum, chunk, loops, fade_ms,
                                            playtime);
    }
    else {
        channelnum = Mix_PlayChannelTimed(channelnum, chunk, loops, playtime);
    }
    if (channelnum != -1)
        Mix_GroupChannel(channelnum, (int)(intptr_t)chunk);
    Py_END_ALLOW_THREADS;

    Py_XDECREF(channeldata[channelnum].sound);
    Py_XDECREF(channeldata[channelnum].queue);
    channeldata[channelnum].sound = sound;
    channeldata[channelnum].queue = NULL;
    Py_INCREF(sound);
    Py_RETURN_NONE;
}

static PyObject *
chan_queue(PyObject *self, PyObject *sound)
{
    int channelnum = pgChannel_AsInt(self);
    Mix_Chunk *chunk;

    if (!pgSound_Check(sound)) {
        return RAISE(PyExc_TypeError,
                     "The argument must be an instance of Sound");
    }

    chunk = pgSound_AsChunk(sound);
    CHECK_CHUNK_VALID(chunk, NULL);
    if (!channeldata[channelnum].sound) /*nothing playing*/
    {
        Py_BEGIN_ALLOW_THREADS;
        channelnum = Mix_PlayChannelTimed(channelnum, chunk, 0, -1);
        if (channelnum != -1)
            Mix_GroupChannel(channelnum, (int)(intptr_t)chunk);
        Py_END_ALLOW_THREADS;

        channeldata[channelnum].sound = sound;
        Py_INCREF(sound);
    }
    else {
        Py_XDECREF(channeldata[channelnum].queue);
        channeldata[channelnum].queue = sound;
        Py_INCREF(sound);
    }
    Py_RETURN_NONE;
}

static PyObject *
chan_get_busy(PyObject *self, PyObject *_null)
{
    int channelnum = pgChannel_AsInt(self);
    MIXER_INIT_CHECK();

    return PyBool_FromLong(Mix_Playing(channelnum));
}

static PyObject *
chan_fadeout(PyObject *self, PyObject *args)
{
    int channelnum = pgChannel_AsInt(self);
    int _time;
    if (!PyArg_ParseTuple(args, "i", &_time))
        return NULL;

    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    Mix_FadeOutChannel(channelnum, _time);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject *
chan_stop(PyObject *self, PyObject *_null)
{
    int channelnum = pgChannel_AsInt(self);
    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    Mix_HaltChannel(channelnum);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject *
chan_pause(PyObject *self, PyObject *_null)
{
    int channelnum = pgChannel_AsInt(self);
    MIXER_INIT_CHECK();

    Mix_Pause(channelnum);
    Py_RETURN_NONE;
}

static PyObject *
chan_unpause(PyObject *self, PyObject *_null)
{
    int channelnum = pgChannel_AsInt(self);
    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    Mix_Resume(channelnum);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject *
chan_set_volume(PyObject *self, PyObject *args)
{
    int channelnum = pgChannel_AsInt(self);
    float volume, stereovolume = -1.11f;
#ifdef Py_DEBUG
    int result;
#endif
    Uint8 left, right;
    PyThreadState *_save;

    if (!PyArg_ParseTuple(args, "f|f", &volume, &stereovolume))
        return NULL;

    MIXER_INIT_CHECK();
    if ((stereovolume <= -1.10f) && (stereovolume >= -1.12f)) {
        /* The normal volume will be used. No panning. so panning is
         * set to full. this is in case it was set previously to
         * something else. NOTE: there is no way to GetPanning
         * variables.
         */
        left = 255;
        right = 255;

        _save = PyEval_SaveThread();
        if (!Mix_SetPanning(channelnum, left, right)) {
            PyEval_RestoreThread(_save);
            return RAISE(pgExc_SDLError, Mix_GetError());
        }
        PyEval_RestoreThread(_save);
    }
    else {
        /* NOTE: here the volume will be set to 1.0 and the panning will
         * be used. */
        left = (Uint8)(volume * 255);
        right = (Uint8)(stereovolume * 255);
        /*
        printf("left:%d:  right:%d:\n", left, right);
        */

        _save = PyEval_SaveThread();
        if (!Mix_SetPanning(channelnum, left, right)) {
            PyEval_RestoreThread(_save);
            return RAISE(pgExc_SDLError, Mix_GetError());
        }
        PyEval_RestoreThread(_save);

        volume = 1.0f;
    }

#ifdef Py_DEBUG
    result =
#endif
        Mix_Volume(channelnum, (int)(volume * 128));
    Py_RETURN_NONE;
}

static PyObject *
chan_get_volume(PyObject *self, PyObject *_null)
{
    int channelnum = pgChannel_AsInt(self);
    int volume;

    MIXER_INIT_CHECK();

    volume = Mix_Volume(channelnum, -1);

    return PyFloat_FromDouble(volume / 128.0);
}

static PyObject *
chan_get_sound(PyObject *self, PyObject *_null)
{
    int channelnum = pgChannel_AsInt(self);
    PyObject *sound;

    sound = channeldata[channelnum].sound;
    if (!sound)
        Py_RETURN_NONE;

    Py_INCREF(sound);
    return sound;
}

static PyObject *
chan_get_queue(PyObject *self, PyObject *_null)
{
    int channelnum = pgChannel_AsInt(self);
    PyObject *sound;

    sound = channeldata[channelnum].queue;
    if (!sound)
        Py_RETURN_NONE;

    Py_INCREF(sound);
    return sound;
}

static PyObject *
chan_set_endevent(PyObject *self, PyObject *args)
{
    int channelnum = pgChannel_AsInt(self);
    int event = SDL_NOEVENT;

    if (!PyArg_ParseTuple(args, "|i", &event))
        return NULL;

    channeldata[channelnum].endevent = event;
    Py_RETURN_NONE;
}

static PyObject *
chan_get_endevent(PyObject *self, PyObject *_null)
{
    int channelnum = pgChannel_AsInt(self);

    return PyLong_FromLong(channeldata[channelnum].endevent);
}

static PyMethodDef channel_methods[] = {
    {"play", (PyCFunction)chan_play, METH_VARARGS | METH_KEYWORDS,
     DOC_CHANNELPLAY},
    {"queue", chan_queue, METH_O, DOC_CHANNELQUEUE},
    {"get_busy", (PyCFunction)chan_get_busy, METH_NOARGS, DOC_CHANNELGETBUSY},
    {"fadeout", chan_fadeout, METH_VARARGS, DOC_CHANNELFADEOUT},
    {"stop", (PyCFunction)chan_stop, METH_NOARGS, DOC_CHANNELSTOP},
    {"pause", (PyCFunction)chan_pause, METH_NOARGS, DOC_CHANNELPAUSE},
    {"unpause", (PyCFunction)chan_unpause, METH_NOARGS, DOC_CHANNELUNPAUSE},
    {"set_volume", chan_set_volume, METH_VARARGS, DOC_CHANNELSETVOLUME},
    {"get_volume", (PyCFunction)chan_get_volume, METH_NOARGS,
     DOC_CHANNELGETVOLUME},

    {"get_sound", (PyCFunction)chan_get_sound, METH_NOARGS,
     DOC_CHANNELGETSOUND},
    {"get_queue", (PyCFunction)chan_get_queue, METH_NOARGS,
     DOC_CHANNELGETQUEUE},

    {"set_endevent", chan_set_endevent, METH_VARARGS, DOC_CHANNELSETENDEVENT},
    {"get_endevent", (PyCFunction)chan_get_endevent, METH_NOARGS,
     DOC_CHANNELGETENDEVENT},

    {NULL, NULL, 0, NULL}};

/* channel object internals */

static void
channel_dealloc(PyObject *self)
{
    PyObject_Free(self);
}

static int
_channel_init(pgChannelObject *self, int channelnum)
{
    if (!SDL_WasInit(SDL_INIT_AUDIO)) {
        PyErr_SetString(pgExc_SDLError, "mixer not initialized");
        return -1;
    }
    if (channelnum < 0 || channelnum >= Mix_GroupCount(-1)) {
        PyErr_SetString(PyExc_IndexError, "invalid channel index");
        return -1;
    }
    self->chan = channelnum;
    return 0;
}

static int
channel_init(pgChannelObject *self, PyObject *args, PyObject *kwargs)
{
    int channelnum;
    if (!PyArg_ParseTuple(args, "i", &channelnum)) {
        return -1;
    }

    return _channel_init(self, channelnum);
}

static PyTypeObject pgChannel_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "Channel",
    .tp_basicsize = sizeof(pgChannelObject),
    .tp_dealloc = channel_dealloc,
    .tp_doc = DOC_PYGAMEMIXERCHANNEL,
    .tp_methods = channel_methods,
    .tp_init = (initproc)channel_init,
    .tp_new = PyType_GenericNew,
};

/*mixer module methods*/

static PyObject *
get_num_channels(PyObject *self, PyObject *_null)
{
    MIXER_INIT_CHECK();
    return PyLong_FromLong(Mix_GroupCount(-1));
}

static PyObject *
set_num_channels(PyObject *self, PyObject *args)
{
    int numchans, i;
    if (!PyArg_ParseTuple(args, "i", &numchans))
        return NULL;

    MIXER_INIT_CHECK();
    if (numchans > numchanneldata) {
        struct ChannelData *cd_org = channeldata;
        channeldata = (struct ChannelData *)realloc(
            channeldata, sizeof(struct ChannelData) * numchans);
        if (!channeldata) {
            /* Restore the original to avoid leaking it */
            channeldata = cd_org;
            return PyErr_NoMemory();
        }
        for (i = numchanneldata; i < numchans; ++i) {
            channeldata[i].sound = NULL;
            channeldata[i].queue = NULL;
            channeldata[i].endevent = 0;
        }
        numchanneldata = numchans;
    }

    Py_BEGIN_ALLOW_THREADS;
    Mix_AllocateChannels(numchans);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject *
set_reserved(PyObject *self, PyObject *args)
{
    int numchans_requested;
    int numchans_reserved;
    if (!PyArg_ParseTuple(args, "i", &numchans_requested))
        return NULL;

    MIXER_INIT_CHECK();

    numchans_reserved = Mix_ReserveChannels(numchans_requested);
    return PyLong_FromLong(numchans_reserved);
}

static PyObject *
get_busy(PyObject *self, PyObject *_null)
{
    if (!SDL_WasInit(SDL_INIT_AUDIO))
        return PyBool_FromLong(0);

    return PyBool_FromLong(Mix_Playing(-1));
}

static PyObject *
mixer_find_channel(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int chan, force = 0;
    static char *keywords[] = {"force", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", keywords, &force)) {
        return NULL;
    }

    MIXER_INIT_CHECK();

    chan = Mix_GroupAvailable(-1);
    if (chan == -1) {
        if (!force)
            Py_RETURN_NONE;
        chan = Mix_GroupOldest(-1);
    }
    return pgChannel_New(chan);
}

static PyObject *
mixer_fadeout(PyObject *self, PyObject *args)
{
    int _time;
    if (!PyArg_ParseTuple(args, "i", &_time))
        return NULL;

    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    Mix_FadeOutChannel(-1, _time);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject *
mixer_stop(PyObject *self, PyObject *_null)
{
    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    Mix_HaltChannel(-1);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject *
mixer_pause(PyObject *self, PyObject *_null)
{
    MIXER_INIT_CHECK();

    Mix_Pause(-1);
    Py_RETURN_NONE;
}

static PyObject *
mixer_unpause(PyObject *self, PyObject *_null)
{
    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    Mix_Resume(-1);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

/* Function to get the SDL mixer version data (linked or compiled).
 *
 * Ref:
 * https://www.libsdl.org/projects/SDL_mixer/docs/SDL_mixer_8.html#SEC8
 */
static PyObject *
mixer_get_sdl_mixer_version(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int linked = 1; /* Default is linked version. */

    static char *keywords[] = {"linked", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|p", keywords, &linked)) {
        return NULL; /* Exception already set. */
    }

    /* MIXER_INIT_CHECK() is not required for these methods. */

    if (linked) {
        /* linked version */
        const SDL_version *v = Mix_Linked_Version();
        return Py_BuildValue("iii", v->major, v->minor, v->patch);
    }
    else {
        /* compiled version */
        SDL_version v;
        SDL_MIXER_VERSION(&v);
        return Py_BuildValue("iii", v.major, v.minor, v.patch);
    }
}

static int
_chunk_from_buf(const void *buf, Py_ssize_t len, Mix_Chunk **chunk,
                Uint8 **mem)
{
    Uint8 *m = (Uint8 *)PyMem_Malloc((size_t)len);

    if (!m) {
        PyErr_NoMemory();
        return -1;
    }
    *chunk = Mix_QuickLoad_RAW(m, (Uint32)len);
    if (!*chunk) {
        PyMem_Free(m);
        PyErr_NoMemory();
        return -1;
    }
    memcpy(m, (Uint8 *)buf, (size_t)len);
    *mem = m;
    return 0;
}

static int
_chunk_from_array(void *buf, PG_sample_format_t view_format, int ndim,
                  Py_ssize_t *shape, Py_ssize_t *strides, Mix_Chunk **chunk,
                  Uint8 **mem)
{
    /* TODO: This is taken from _numericsndarray without additions.
     * So this should be extended to properly handle integer sign
     * and byte order. These changes will not be backward compatible.
     */
    int freq;
    Uint16 format;
    int channels;
    int itemsize;
    int view_itemsize = PG_SAMPLE_SIZE(view_format);
    Uint8 *src, *dst;
    Py_ssize_t memsize;
    Py_ssize_t loop1, loop2, step1, step2, length, length2 = 0;

    if (!Mix_QuerySpec(&freq, &format, &channels)) {
        PyErr_SetString(pgExc_SDLError, "mixer not initialized");
        return -1;
    }

    /* Check for compatible values.
     */
    if (channels == 1) {
        if (ndim != 1) {
            PyErr_SetString(PyExc_ValueError,
                            "Array must be 1-dimensional for mono mixer");
            return -1;
        }
    }
    else {
        if (ndim != 2) {
            PyErr_SetString(PyExc_ValueError,
                            "Array must be 2-dimensional for stereo mixer");
            return -1;
        }
        if (shape[1] != channels) {
            PyErr_SetString(PyExc_ValueError,
                            "Array depth must match number of mixer channels");
            return -1;
        }
    }
    itemsize = _format_itemsize(format);
    /*
    printf("!! itemsize: %d\n", itemsize);
    */
    if (itemsize < 0) {
        return -1;
    }
    if (view_itemsize != 1 && view_itemsize != 2 && view_itemsize != 4) {
        PyErr_Format(PyExc_ValueError, "Unsupported integer size %d",
                     view_itemsize);
        return -1;
    }
    length = shape[0];
    step1 = strides ? strides[0] : (Py_ssize_t)view_itemsize * channels;
    length2 = ndim;
    if (ndim == 2) {
        step2 = strides ? strides[1] : view_itemsize;
    }
    else {
        step2 = step1;
    }
    memsize = length * channels * itemsize;
    /*
    printf("memsize: %d\n", (int)memsize);
    */

    /* Create chunk.
     */
    dst = (Uint8 *)PyMem_Malloc((size_t)memsize);
    if (!dst) {
        PyErr_NoMemory();
        return -1;
    }
    *chunk = Mix_QuickLoad_RAW(dst, (Uint32)memsize);
    if (!*chunk) {
        PyMem_Free(dst);
        PyErr_NoMemory();
        return -1;
    }
    *mem = dst;

    /*
    printf("!! step1: %d, step2: %d, view_itemsize: %d, length: %d\n",
           step1, step2, view_itemsize, length);
    */
    /* Copy samples.
     */
    if (step1 == (Py_ssize_t)itemsize * channels && step2 == itemsize) {
        /*OPTIMIZATION: in these cases, we don't need to loop through
         *the samples individually, because the bytes are already laid
         *out correctly*/
        memcpy(dst, buf, memsize);
    }
    else if (itemsize == 1) {
        for (loop1 = 0; loop1 < length; loop1++) {
            src = (Uint8 *)buf + (loop1 * step1);
            switch (view_itemsize) {
                case 1:
                    for (loop2 = 0; loop2 < length2;
                         loop2++, dst += 1, src += step2) {
                        *(Uint8 *)dst = (Uint8) * ((Uint8 *)src);
                    }
                    break;
                case 2:
                    for (loop2 = 0; loop2 < length2;
                         loop2++, dst += 1, src += step2) {
                        *(Uint8 *)dst = (Uint8) * ((Uint16 *)src);
                    }
                    break;
                case 4:
                    for (loop2 = 0; loop2 < length2;
                         loop2++, dst += 1, src += step2) {
                        *(Uint8 *)dst = (Uint8) * ((Uint32 *)src);
                    }
                    break;
            }
        }
    }
    else {
        for (loop1 = 0; loop1 < length; loop1++) {
            src = (Uint8 *)buf + (loop1 * step1);
            switch (view_itemsize) {
                case 1:
                    for (loop2 = 0; loop2 < length2;
                         loop2++, dst += 2, src += step2) {
                        *(Uint16 *)dst = (Uint16)(*((Uint8 *)src) << 8);
                    }
                    break;
                case 2:
                    for (loop2 = 0; loop2 < length2;
                         loop2++, dst += 2, src += step2) {
                        *(Uint16 *)dst = (Uint16) * ((Uint16 *)src);
                    }
                    break;
                case 4:
                    for (loop2 = 0; loop2 < length2;
                         loop2++, dst += 2, src += step2) {
                        *(Uint16 *)dst = (Uint16) * ((Uint32 *)src);
                    }
                    break;
            }
        }
    }

    return 0;
}

static int
sound_init(PyObject *self, PyObject *arg, PyObject *kwarg)
{
    static const char arg_cnt_err_msg[] =
        "Sound takes either 1 positional or 1 keyword argument";
    PyObject *obj = NULL;
    PyObject *file = NULL;
    PyObject *buffer = NULL;
    PyObject *array = NULL;
    PyObject *keys;
    PyObject *kencoded;
    SDL_RWops *rw;
    Mix_Chunk *chunk = NULL;
    Uint8 *mem = NULL;

    ((pgSoundObject *)self)->chunk = NULL;
    ((pgSoundObject *)self)->mem = NULL;

    /* Similar to MIXER_INIT_CHECK(), but different return value. */
    if (!SDL_WasInit(SDL_INIT_AUDIO)) {
        PyErr_SetString(pgExc_SDLError, "mixer not initialized");
        return -1;
    }

    /* Process arguments, returning cleaner error messages than
       PyArg_ParseTupleAndKeywords would.
    */
    if (arg != NULL && PyTuple_GET_SIZE(arg)) {
        if ((kwarg != NULL && PyDict_Size(kwarg)) || /* conditional and */
            PyTuple_GET_SIZE(arg) != 1) {
            PyErr_SetString(PyExc_TypeError, arg_cnt_err_msg);
            return -1;
        }
        obj = PyTuple_GET_ITEM(arg, 0);

        if (PyUnicode_Check(obj)) {
            file = obj;
            obj = NULL;
        }
        else {
            file = obj;
            buffer = obj;
        }
    }
    else if (kwarg != NULL) {
        if (PyDict_Size(kwarg) != 1) {
            PyErr_SetString(PyExc_TypeError, arg_cnt_err_msg);
            return -1;
        }
        if ((file = PyDict_GetItemString(kwarg, "file")) == NULL &&
            (buffer = PyDict_GetItemString(kwarg, "buffer")) == NULL &&
            (array = PyDict_GetItemString(kwarg, "array")) == NULL) {
            keys = PyDict_Keys(kwarg);
            if (keys == NULL) {
                return -1;
            }
            kencoded =
                pg_EncodeString(PyList_GET_ITEM(keys, 0), NULL, NULL, NULL);
            Py_DECREF(keys);
            if (kencoded == NULL) {
                return -1;
            }
            PyErr_Format(PyExc_TypeError,
                         "Unrecognized keyword argument '%.1024s'",
                         PyBytes_AS_STRING(kencoded));
            Py_DECREF(kencoded);
            return -1;
        }
        if (buffer != NULL && PyUnicode_Check(buffer)) { /* conditional and */
            PyErr_SetString(PyExc_TypeError,
                            "Unicode object not allowed as buffer object");
            return -1;
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, arg_cnt_err_msg);
        return -1;
    }

    if (file != NULL) {
        rw = pgRWops_FromObject(file);

        if (rw == NULL) {
            if (obj) {
                /* use 'buffer' as fallback for single arg */
                PyErr_Clear();
                goto LOAD_BUFFER;
            }
            return -1;
        }
        Py_BEGIN_ALLOW_THREADS;
        chunk = Mix_LoadWAV_RW(rw, 1);
        Py_END_ALLOW_THREADS;
        if (chunk == NULL) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            return -1;
        }
    }

LOAD_BUFFER:

    if (!chunk && buffer) {
        Py_buffer view;
        int rcode;

        view.obj = 0;
        if (PyObject_GetBuffer(buffer, &view, PyBUF_SIMPLE)) {
            if (obj != NULL) {
                PyErr_Clear();
            }
            else {
                PyErr_Format(PyExc_TypeError,
                             "Expected object with buffer interface: got a %s",
                             Py_TYPE(buffer)->tp_name);
                return -1;
            }
        }
        else {
            rcode = _chunk_from_buf(view.buf, view.len, &chunk, &mem);
            PyBuffer_Release(&view);
            if (rcode) {
                return -1;
            }
            ((pgSoundObject *)self)->mem = mem;
        }
    }

    if (array != NULL) {
        pg_buffer pg_view;
        PG_sample_format_t view_format;
        int rcode;

        pg_view.view.itemsize = 0;
        pg_view.view.obj = 0;
        if (pgObject_GetBuffer(array, &pg_view, PyBUF_FORMAT | PyBUF_ND)) {
            return -1;
        }
        view_format = _format_view_to_audio((Py_buffer *)&pg_view);
        if (!view_format) {
            pgBuffer_Release(&pg_view);
            return -1;
        }
        rcode = _chunk_from_array(pg_view.view.buf, view_format,
                                  pg_view.view.ndim, pg_view.view.shape,
                                  pg_view.view.strides, &chunk, &mem);
        pgBuffer_Release(&pg_view);
        if (rcode) {
            return -1;
        }
        ((pgSoundObject *)self)->mem = mem;
    }

    if (chunk == NULL) {
        if (obj == NULL) {
            PyErr_SetString(PyExc_TypeError, "Unrecognized argument");
        }
        else {
            PyErr_Format(PyExc_TypeError, "Unrecognized argument (type %s)",
                         Py_TYPE(obj)->tp_name);
        }
        return -1;
    }

    ((pgSoundObject *)self)->chunk = chunk;
    return 0;
}

static PyMethodDef _mixer_methods[] = {
    {"_internal_mod_init", (PyCFunction)pgMixer_AutoInit, METH_NOARGS,
     "auto initialize for mixer"},
    {"init", (PyCFunction)pg_mixer_init, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEMIXERINIT},
    {"quit", (PyCFunction)mixer_quit, METH_NOARGS, DOC_PYGAMEMIXERQUIT},
    {"get_init", (PyCFunction)pg_mixer_get_init, METH_NOARGS,
     DOC_PYGAMEMIXERGETINIT},
    {"pre_init", (PyCFunction)pre_init, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEMIXERPREINIT},
    {"get_num_channels", (PyCFunction)get_num_channels, METH_NOARGS,
     DOC_PYGAMEMIXERGETNUMCHANNELS},
    {"set_num_channels", set_num_channels, METH_VARARGS,
     DOC_PYGAMEMIXERSETNUMCHANNELS},
    {"set_reserved", set_reserved, METH_VARARGS, DOC_PYGAMEMIXERSETRESERVED},

    {"get_busy", (PyCFunction)get_busy, METH_NOARGS, DOC_PYGAMEMIXERGETBUSY},
    {"find_channel", (PyCFunction)mixer_find_channel,
     METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEMIXERFINDCHANNEL},
    {"fadeout", mixer_fadeout, METH_VARARGS, DOC_PYGAMEMIXERFADEOUT},
    {"stop", (PyCFunction)mixer_stop, METH_NOARGS, DOC_PYGAMEMIXERSTOP},
    {"pause", (PyCFunction)mixer_pause, METH_NOARGS, DOC_PYGAMEMIXERPAUSE},
    {"unpause", (PyCFunction)mixer_unpause, METH_NOARGS,
     DOC_PYGAMEMIXERUNPAUSE},
    {"get_sdl_mixer_version", (PyCFunction)mixer_get_sdl_mixer_version,
     METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEMIXERGETSDLMIXERVERSION},
    /*  { "lookup_frequency", lookup_frequency, 1, doc_lookup_frequency
       },*/

    {NULL, NULL, 0, NULL}};

static PyObject *
pgSound_New(Mix_Chunk *chunk)
{
    pgSoundObject *soundobj;

    if (!chunk)
        return RAISE(PyExc_RuntimeError, "unable to create sound.");

    soundobj = (pgSoundObject *)pgSound_Type.tp_new(&pgSound_Type, NULL, NULL);
    if (soundobj) {
        soundobj->mem = NULL;
        soundobj->chunk = chunk;
    }

    return (PyObject *)soundobj;
}

static PyObject *
pgChannel_New(int channelnum)
{
    pgChannelObject *chanobj = PyObject_New(pgChannelObject, &pgChannel_Type);
    if (!chanobj) {
        return NULL;
    }
    if (_channel_init(chanobj, channelnum)) {
        Py_DECREF(chanobj);
        return NULL;
    }
    return (PyObject *)chanobj;
}

#if BUILD_STATIC
// avoid conflict with PyInit_mixer in _sdl2/mixer.c
MODINIT_DEFINE(pg_mixer)
#else
MODINIT_DEFINE(mixer)
#endif
{
    PyObject *module, *apiobj, *music = NULL;
    static void *c_api[PYGAMEAPI_MIXER_NUMSLOTS];

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "mixer",
                                         DOC_PYGAMEMIXER,
                                         -1,
                                         _mixer_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */

    /*imported needed apis*/
    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_pygame_rwobject();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_pygame_event();
    if (PyErr_Occurred()) {
        return NULL;
    }

    /* type preparation */
    if (PyType_Ready(&pgSound_Type) < 0) {
        return NULL;
    }
    if (PyType_Ready(&pgChannel_Type) < 0) {
        return NULL;
    }

    /* create the module */
    module = PyModule_Create(&_module);
    if (module == NULL) {
        return NULL;
    }

    Py_INCREF(&pgSound_Type);
    if (PyModule_AddObject(module, "Sound", (PyObject *)&pgSound_Type)) {
        Py_DECREF(&pgSound_Type);
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(&pgSound_Type);
    if (PyModule_AddObject(module, "SoundType", (PyObject *)&pgSound_Type)) {
        Py_DECREF(&pgSound_Type);
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(&pgChannel_Type);
    if (PyModule_AddObject(module, "ChannelType",
                           (PyObject *)&pgChannel_Type)) {
        Py_DECREF(&pgChannel_Type);
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(&pgChannel_Type);
    if (PyModule_AddObject(module, "Channel", (PyObject *)&pgChannel_Type)) {
        Py_DECREF(&pgChannel_Type);
        Py_DECREF(module);
        return NULL;
    }
    /* export the c api */
    c_api[0] = &pgSound_Type;
    c_api[1] = pgSound_New;
    c_api[2] = pgSound_Play;
    c_api[3] = &pgChannel_Type;
    c_api[4] = pgChannel_New;
    apiobj = encapsulate_api(c_api, "mixer");
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_XDECREF(apiobj);
        Py_DECREF(module);
        return NULL;
    }

    music = import_music();
    if (music) {
        if (PyModule_AddObject(module, "music", music)) {
            Py_DECREF(music);
            Py_DECREF(module);
            return NULL;
        }
    }
    else {
        PyErr_Clear();
    }
    return module;
}
