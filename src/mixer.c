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
#include "pgarrinter.h"

#if !defined(EXPORT_BUFFER)
#define EXPORT_BUFFER HAVE_NEW_BUFPROTO
#endif

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
#define PG_SAMPLE_SIZE(sf) ((sf) & 0x0ffffu)
#define PG_IS_SAMPLE_SIGNED(sf) ((sf) & PG_SAMPLE_SIGNED != 0)
#define PG_IS_SAMPLE_NATIVE_ENDIAN(sf) ((sf) & PG_SAMPLE_NATIVE_ENDIAN != 0)
#define PG_IS_SAMPLE_LITTLE_ENDIAN(sf) \
    ((sf) & PG_SAMPLE_LITTLE_ENDIAN == PG_SAMPLE_LITTLE_ENDIAN)
#define PG_IS_SAMPLE_BIG_ENDIAN(sf) \
    ((sf) & PG_SAMPLE_BIG_ENDIAN == PG_SAMPLE_BIG_ENDIAN)

/* Since they are documented, the default init values are defined here
   rather than taken from SDL_mixer. It also means that the default
   size is defined in Pygame, rather than SDL AUDIO_xxx, terms.
 */
#define PYGAME_MIXER_DEFAULT_FREQUENCY 22050
#define PYGAME_MIXER_DEFAULT_SIZE -16
#define PYGAME_MIXER_DEFAULT_CHANNELS 2
#define PYGAME_MIXER_DEFAULT_CHUNKSIZE 4096

static PyTypeObject PySound_Type;
static PyTypeObject PyChannel_Type;
static PyObject* PySound_New (Mix_Chunk*);
static PyObject* PyChannel_New (int);
#define PySound_Check(x) (Py_TYPE(x) == &PySound_Type)
#define PyChannel_Check(x) (Py_TYPE(x) == &PyChannel_Type)

static int request_frequency = PYGAME_MIXER_DEFAULT_FREQUENCY;
static int request_size = PYGAME_MIXER_DEFAULT_SIZE;
static int request_stereo = PYGAME_MIXER_DEFAULT_CHANNELS;
static int request_chunksize = PYGAME_MIXER_DEFAULT_CHUNKSIZE;

static int sound_init (PyObject* self, PyObject* arg, PyObject* kwarg);

struct ChannelData
{
    PyObject* sound;
    PyObject* queue;
    int endevent;
};
static struct ChannelData *channeldata = NULL;
static int numchanneldata = 0;

Mix_Music** current_music;
Mix_Music** queue_music;

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

    default:
        PyErr_Format(PyExc_SystemError,
                     "Pygame bug (mixer.Sound): unknown mixer format %d",
                     (int)format);
    }
    return size;
}

static int
_get_array_interface(PyObject *obj,
                     PyObject **cobj_p,
                     PyArrayInterface **inter_p)
{
    PyObject *cobj = PyObject_GetAttrString(obj, "__array_struct__");
    PyArrayInterface *inter = NULL;

    if (cobj == NULL) {
        if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
                PyErr_Clear();
                PyErr_SetString(PyExc_ValueError,
                                "no C-struct array interface");
        }
        return -1;
    }

#if PG_HAVE_COBJECT
    if (PyCObject_Check(cobj)) {
        inter = (PyArrayInterface *)PyCObject_AsVoidPtr(cobj);
    }
#endif
#if PG_HAVE_CAPSULE
    if (PyCapsule_IsValid(cobj, NULL)) {
        inter = (PyArrayInterface *)PyCapsule_GetPointer(cobj, NULL);
    }
#endif
    if (inter == NULL ||   /* conditional or */
        inter->two != 2  ) {
        Py_DECREF(cobj);
        PyErr_SetString(PyExc_ValueError, "invalid array interface");
        return -1;
    }

    *cobj_p = cobj;
    *inter_p = inter;
    return 0;
}

static PG_sample_format_t
_format_inter_to_audio(PyArrayInterface *inter)
{
    PG_sample_format_t format = 0;
    int itemsize = inter->itemsize;

    switch (inter->typekind) {
        case 'u':
        break;

        case 'i':
        format |= PG_SAMPLE_SIGNED;
        break;

        default:
        PyErr_Format(PyExc_ValueError,
                     "Array has unsupported item format '%c'",
                     (int)inter->typekind);
        return 0;
    }
    if (itemsize <= 0 || itemsize > 0xFFFFl) {
        PyErr_Format(PyExc_ValueError,
                     "Array has unsupported integer size %d", itemsize);
        return 0;
    }
    format += itemsize;
    format |= inter->flags & PAI_NOTSWAPPED ? PG_SAMPLE_NATIVE_ENDIAN : 0;
    return format;
}

#if HAVE_NEW_BUFPROTO
static PG_sample_format_t
_format_view_to_audio(Py_buffer *view)
{
    int fstr_len;
    int native_size = 0;
    int index = 0;
    PG_sample_format_t format = 0;

    if (!view->format) {
        /* Assume unsigned byte */
        return (PG_sample_format_t)sizeof(unsigned char);
    }
    fstr_len = strlen(view->format);
    if (fstr_len < 1 || fstr_len > 2) {
        RAISE(PyExc_ValueError, "Array has unsupported item format");
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
            RAISE(PyExc_ValueError, "Array has unsupported item format");
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

    case 'l':
        format |= PG_SAMPLE_SIGNED;
        format += native_size ? sizeof(long int) : 4;
        break;

    case 'L':
        format += native_size ? sizeof(unsigned long int) : 4;
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
#endif

static void
endsound_callback (int channel)
{
    if (channeldata)
    {
    if (channeldata[channel].endevent && SDL_WasInit (SDL_INIT_VIDEO))
    {
        SDL_Event e;
        memset (&e, 0, sizeof(e));
        e.type = channeldata[channel].endevent;
            if (e.type >= SDL_USEREVENT && e.type < SDL_NUMEVENTS)
                e.user.code = channel;
        SDL_PushEvent (&e);
    }
    if (channeldata[channel].queue)
    {
        int channelnum;
        Mix_Chunk* sound = PySound_AsChunk (channeldata[channel].queue);
        Py_XDECREF (channeldata[channel].sound);
        channeldata[channel].sound = channeldata[channel].queue;
        channeldata[channel].queue = NULL;
        channelnum = Mix_PlayChannelTimed (channel, sound, 0, -1);
        if (channelnum != -1)
            Mix_GroupChannel (channelnum, (intptr_t)sound);
    }
    else
    {
        Py_XDECREF (channeldata[channel].sound);
        channeldata[channel].sound = NULL;
    }
    }
}

static void
autoquit(void)
{
    int i;
    if (SDL_WasInit (SDL_INIT_AUDIO))
    {
        Mix_HaltMusic ();

        if (channeldata)
        {
            for (i = 0; i < numchanneldata; ++i)
            {
                Py_XDECREF (channeldata[i].sound);
                Py_XDECREF (channeldata[i].queue);
            }
            free (channeldata);
            channeldata = NULL;
            numchanneldata = 0;
        }

        if (current_music)
        {
            if (*current_music)
            {
                Mix_FreeMusic (*current_music);
                *current_music = NULL;
            }
            current_music = NULL;
        }
        if (queue_music)
        {
            if (*queue_music)
            {
                Mix_FreeMusic (*queue_music);
                *queue_music = NULL;
            }
            queue_music = NULL;
        }

        Mix_CloseAudio ();
        SDL_QuitSubSystem (SDL_INIT_AUDIO);
    }
}

static PyObject*
_init (int freq, int size, int stereo, int chunk)
{
    Uint16 fmt = 0;
    int i;

    if (!freq) {
    freq = request_frequency;
    }
    if (!size) {
    size = request_size;
    }
    if (!stereo) {
    stereo = request_stereo;
    }
    if (!chunk) {
    chunk = request_chunksize;
    }
    if (stereo >= 2)
        stereo = 2;
    else
        stereo = 1;

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
    default:
    PyErr_Format(PyExc_ValueError, "unsupported size %i", size);
    return NULL;
    }

    /* printf("size:%d:\n", size); */



    /*make chunk a power of 2*/
    for (i = 0; 1 << i < chunk; ++i); /*yes, semicolon on for loop*/
    chunk = MAX (1 << i, 256);        /*do this after foo loop exits*/

    if (!SDL_WasInit (SDL_INIT_AUDIO))
    {
        PyGame_RegisterQuit (autoquit);

        if (!channeldata) /*should always be null*/
        {
            numchanneldata = MIX_CHANNELS;
            channeldata = (struct ChannelData*)
                malloc (sizeof (struct ChannelData) *numchanneldata);
            for (i = 0; i < numchanneldata; ++i)
            {
                channeldata[i].sound = NULL;
                channeldata[i].queue = NULL;
                channeldata[i].endevent = 0;
            }
        }

        if (SDL_InitSubSystem (SDL_INIT_AUDIO) == -1)
            return PyInt_FromLong (0);

        if (Mix_OpenAudio (freq, fmt, stereo, chunk) == -1)
        {
            SDL_QuitSubSystem (SDL_INIT_AUDIO);
            return PyInt_FromLong (0);
        }
#if MIX_MAJOR_VERSION>=1 && MIX_MINOR_VERSION>=2 && MIX_PATCHLEVEL>=3
        Mix_ChannelFinished (endsound_callback);
#endif

        /* A bug in sdl_mixer where the stereo is reversed for 8 bit.
           So we use this CPU hogging effect to reverse it for us.
           Hopefully this bug is fixed in SDL_mixer 1.2.9
        printf("MIX_MAJOR_VERSION :%d: MIX_MINOR_VERSION :%d: MIX_PATCHLEVEL :%d: \n",
               MIX_MAJOR_VERSION, MIX_MINOR_VERSION, MIX_PATCHLEVEL);
        */

#if MIX_MAJOR_VERSION>=1 && MIX_MINOR_VERSION>=2 && MIX_PATCHLEVEL<=8
        if(fmt == AUDIO_U8) {
            if(!Mix_SetReverseStereo(MIX_CHANNEL_POST, 1)) {
                /* We do nothing... because might as well just let it go ahead. */
                /* return RAISE (PyExc_SDLError, Mix_GetError());
                */
            }
        }
#endif


        Mix_VolumeMusic (127);
    }
    return PyInt_FromLong (1);
}

static PyObject*
autoinit (PyObject* self, PyObject* arg)
{
    int freq = 0, size = 0, stereo = 0, chunk = 0;

    if (!PyArg_ParseTuple (arg, "|iiii", &freq, &size, &stereo, &chunk))
        return NULL;

    return _init (freq, size, stereo, chunk);
}

static PyObject*
quit (PyObject* self)
{
    autoquit ();
    Py_RETURN_NONE;
}

static PyObject*
init (PyObject* self, PyObject* args, PyObject* keywds)
{
    int freq = 0, size = 0, stereo = 0, chunk = 0;
    PyObject* result;
    int value;

    static char *kwids[] = {"frequency", "size", "channels", "buffer", NULL};
    if (!PyArg_ParseTupleAndKeywords (args, keywds, "|iiii", kwids,
                      &freq, &size, &stereo, &chunk)) {
    return NULL;
    }
    result = _init (freq, size, stereo, chunk);
    if (!result)
        return NULL;
    value = PyObject_IsTrue (result);
    Py_DECREF (result);
    if (!value)
        return RAISE (PyExc_SDLError, SDL_GetError());

    Py_RETURN_NONE;
}

static PyObject*
get_init (PyObject* self)
{
    int freq, channels, realform;
    Uint16 format;

    if (!SDL_WasInit (SDL_INIT_AUDIO))
        Py_RETURN_NONE;

    if (!Mix_QuerySpec (&freq, &format, &channels))
        Py_RETURN_NONE;

    //create a signed or unsigned number of bits per sample
    // XXX: When mixer is init'd with a format of -8, this returns +8
    realform = format&~0xff ? - (format&0xff) : format&0xff;
    return Py_BuildValue ("(iii)", freq, realform, channels);
}

static PyObject*
pre_init (PyObject* self, PyObject* args, PyObject* keywds)
{
    static char *kwids[] = {"frequency", "size", "channels", "buffer", NULL};

    request_frequency = 0;
    request_size = 0;
    request_stereo = 0;
    request_chunksize = 0;
    if (!PyArg_ParseTupleAndKeywords (args, keywds, "|iiii", kwids,
                      &request_frequency, &request_size,
                      &request_stereo, &request_chunksize))
        return NULL;
    if (!request_frequency) {
    request_frequency = PYGAME_MIXER_DEFAULT_FREQUENCY;
    }
    if (!request_size) {
    request_size = PYGAME_MIXER_DEFAULT_SIZE;
    }
    if (!request_stereo) {
    request_stereo = PYGAME_MIXER_DEFAULT_CHANNELS;
    }
    if (!request_chunksize) {
    request_chunksize = PYGAME_MIXER_DEFAULT_CHUNKSIZE;
    }
    Py_RETURN_NONE;
}

/* sound object methods */

static PyObject*
snd_play (PyObject* self, PyObject* args, PyObject* kwargs)
{
    Mix_Chunk* chunk = PySound_AsChunk (self);
    int channelnum = -1;
    int loops = 0, playtime = -1, fade_ms = 0;

    char *kwids[] = { "loops", "maxtime", "fade_ms", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iii", kwids, &loops, &playtime, &fade_ms))
       return NULL;

    if (fade_ms > 0)
    {
        channelnum = Mix_FadeInChannelTimed (-1, chunk, loops, fade_ms, playtime);
    }
    else
    {
        channelnum = Mix_PlayChannelTimed (-1, chunk, loops, playtime);
    }
    if (channelnum == -1)
        Py_RETURN_NONE;

    Py_XDECREF (channeldata[channelnum].sound);
    Py_XDECREF (channeldata[channelnum].queue);
    channeldata[channelnum].queue = NULL;
    channeldata[channelnum].sound = self;
    Py_INCREF (self);

    //make sure volume on this arbitrary channel is set to full
    Mix_Volume (channelnum, 128);

    Mix_GroupChannel (channelnum, (intptr_t)chunk);
    return PyChannel_New (channelnum);
}

static PyObject*
snd_get_num_channels (PyObject* self)
{
    Mix_Chunk* chunk = PySound_AsChunk (self);
    MIXER_INIT_CHECK ();
    return PyInt_FromLong (Mix_GroupCount ((intptr_t)chunk));
}

static PyObject*
snd_fadeout (PyObject* self, PyObject* args)
{
    Mix_Chunk* chunk = PySound_AsChunk (self);
    int _time;
    if (!PyArg_ParseTuple (args, "i", &_time))
        return NULL;

    MIXER_INIT_CHECK ();

    Mix_FadeOutGroup ((intptr_t)chunk, _time);
    Py_RETURN_NONE;
}

static PyObject*
snd_stop (PyObject* self)
{
    Mix_Chunk* chunk = PySound_AsChunk (self);
    MIXER_INIT_CHECK ();
    Mix_HaltGroup ((intptr_t)chunk);
    Py_RETURN_NONE;
}

static PyObject*
snd_set_volume (PyObject* self, PyObject* args)
{
    Mix_Chunk* chunk = PySound_AsChunk (self);
    float volume;

    if (!PyArg_ParseTuple (args, "f", &volume))
        return NULL;

    MIXER_INIT_CHECK ();

    Mix_VolumeChunk (chunk, (int)(volume*128));
    Py_RETURN_NONE;
}

static PyObject*
snd_get_volume (PyObject* self)
{
    Mix_Chunk* chunk = PySound_AsChunk (self);
    int volume;
    MIXER_INIT_CHECK ();

    volume = Mix_VolumeChunk (chunk, -1);
    return PyFloat_FromDouble (volume / 128.0);
}

static PyObject*
snd_get_length (PyObject* self)
{
    Mix_Chunk* chunk = PySound_AsChunk (self);
    int freq, channels, mixerbytes, numsamples;
    Uint16 format;
    MIXER_INIT_CHECK ();

    Mix_QuerySpec (&freq, &format, &channels);
    if (format==AUDIO_S8 || format==AUDIO_U8)
        mixerbytes = 1;
    else
        mixerbytes = 2;
    numsamples = chunk->alen / mixerbytes / channels;

    return PyFloat_FromDouble ((float)numsamples / (float)freq);
}

static PyObject*
snd_get_buffer (PyObject* self)
{
    PyObject *buffer;
    Mix_Chunk* chunk;
    MIXER_INIT_CHECK ();

    chunk = PySound_AsChunk (self);
    buffer = PyBufferProxy_New (self, chunk->abuf, (Py_ssize_t) chunk->alen,
                                NULL);
    if (!buffer)
        return RAISE (PyExc_SDLError, "could acquire a buffer for the sound");
    return buffer;
}

#if PY3
static void
snd_arraystruct_capsule_destr(PyObject *capsule)
{
    PyArrayInterface *inter =
        (PyArrayInterface *)PyCapsule_GetPointer(capsule, 0);

    PyMem_Free(inter->shape);
    PyMem_Free(inter);

    /* The context may be NULL in the unlike case PyCapsule_SetContext failed.
     */
    Py_XDECREF((PyObject *)PyCapsule_GetContext(capsule));
}
#else
static void
snd_arraystruct_cobject_destr(void *inter_vp, void *desc_vp)
{
    PyArrayInterface *inter = (PyArrayInterface *)inter_vp;

    PyMem_Free(inter->shape);
    PyMem_Free(inter);
    Py_DECREF((PyObject *)desc_vp);
}
#endif

static PyObject *
snd_get_arraystruct(PyObject *self, void *closure)
{
    Mix_Chunk *chunk = PySound_AsChunk(self);
    PyArrayInterface *inter;
    int nd;
    Py_intptr_t *shape;
    Py_intptr_t *strides;
    int freq = 0;
    int channels;
    Py_ssize_t len;
    Uint16 format;
    PyObject *desc;
    PyObject *cobj;

    MIXER_INIT_CHECK();
    desc = Py_BuildValue("sO", "PyArrayInterface Version 3", self);
    if (!desc) {
        return NULL;
    }
    inter = PyMem_New(PyArrayInterface, 1);
    if (!inter) {
        Py_DECREF(desc);
        return PyErr_NoMemory();
    }
    Mix_QuerySpec(&freq, &format, &channels);
    nd = channels > 1 ? 2 : 1;
    len = chunk->alen;
    inter->two = 2;
    inter->nd = nd;
    switch (format) {
        case AUDIO_U8:
        inter->typekind = 'u';
        inter->itemsize = 1;
        break;

        case AUDIO_S8:
        inter->typekind = 'i';
        inter->itemsize = 1;
        break;

        case AUDIO_U16SYS:
        inter->typekind = 'u';
        inter->itemsize = 2;
        break;

        default:
        inter->typekind = 'i';
        inter->itemsize = 2;
        break;
    }
    inter->flags = PAI_CONTIGUOUS | PAI_ALIGNED |
                   PAI_NOTSWAPPED | PAI_WRITEABLE;
    shape = PyMem_New(Py_intptr_t, 2 * nd);
    if (!shape) {
        Py_DECREF(desc);
        PyMem_Free(inter);
        return PyErr_NoMemory();
    }
    strides = shape + nd;
    strides[0] = inter->itemsize * channels;
    shape[0] = len / strides[0];
    if (nd == 2) {
        shape[1] = channels;
        strides[1] = inter->itemsize;
    }
    inter->shape = shape;
    inter->strides = strides;
    inter->data = chunk->abuf;
    inter->descr = 0;
#if PY3
    cobj = PyCapsule_New(inter, 0, snd_arraystruct_capsule_destr);
#else
    cobj = PyCObject_FromVoidPtrAndDesc(inter,
                                        desc,
                                        snd_arraystruct_cobject_destr);
#endif
    if (!cobj) {
        Py_DECREF(desc);
        PyMem_Free(inter->shape);
        PyMem_Free(inter);
        return NULL;
    }
#if PY3
    else if (PyCapsule_SetContext(cobj, desc)) {
        Py_DECREF(cobj);
        return NULL;
    }
#endif
    return cobj;
}

static PyMethodDef sound_methods[] =
{
    { "play", (PyCFunction) snd_play, METH_VARARGS | METH_KEYWORDS,
      DOC_SOUNDPLAY },
    { "get_num_channels", (PyCFunction) snd_get_num_channels, METH_NOARGS,
      DOC_SOUNDGETNUMCHANNELS },
    { "fadeout", snd_fadeout, METH_VARARGS, DOC_SOUNDFADEOUT },
    { "stop", (PyCFunction) snd_stop, METH_NOARGS, DOC_SOUNDSTOP },
    { "set_volume", snd_set_volume, METH_VARARGS, DOC_SOUNDSETVOLUME },
    { "get_volume", (PyCFunction) snd_get_volume, METH_NOARGS,
      DOC_SOUNDGETVOLUME },
    { "get_length", (PyCFunction) snd_get_length, METH_NOARGS,
      DOC_SOUNDGETLENGTH },
    { "get_buffer", (PyCFunction) snd_get_buffer, METH_NOARGS,
      DOC_SOUNDGETBUFFER },
    { NULL, NULL, 0, NULL }
};

static PyGetSetDef sound_getset[] =
{
    {"__array_struct__", snd_get_arraystruct, NULL, "Version 3", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};


/*buffer protocol*/

#if EXPORT_BUFFER
static int
snd_buffer_iteminfo(char **format, Py_ssize_t *itemsize, int *channels)
{
    static char fmt_AUDIO_U8[] = "B";
    static char fmt_AUDIO_S8[] = "b";
    static char fmt_AUDIO_U16SYS[] = "=H";
    static char fmt_AUDIO_S16SYS[] = "=h";
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
    }

    PyErr_Format(PyExc_SystemError,
                 "Pygame bug (mixer.Sound): unknown mixer format %d",
                 (int)mixer_format);
    return -1;
}

static int
snd_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    Mix_Chunk *chunk = PySound_AsChunk(obj);
    int channels;
    char *format = '\0';
    int ndim = 1;
    Py_ssize_t *shape = 0;
    Py_ssize_t *strides = 0;
    Py_ssize_t itemsize = 0;
    Py_ssize_t samples;
    int fortran_order = (flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS;

    if (flags != PyBUF_SIMPLE) {
        if (snd_buffer_iteminfo(&format, &itemsize, &channels)) {
            return -1;
        }
        if ((flags & PyBUF_FORMAT) != PyBUF_FORMAT) {
            format = 0;
        }
        if ((flags & PyBUF_ND) == PyBUF_ND) {
            ndim = channels > 1 ? 2 : 1;
            samples = chunk->alen / (itemsize * channels);
            shape = PyMem_New(Py_ssize_t, 2 * ndim);
            if (!shape) {
                PyErr_NoMemory();
                return -1;
            }
	    view->internal = shape;
            if (fortran_order) {
                shape[0] = channels;
                shape[ndim - 1] = samples;
            }
            else {
                shape[ndim - 1] = channels;
                shape[0] = samples;
            }
        }
        if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
            strides = shape + ndim;
            if (fortran_order) {
                strides[ndim - 1] = itemsize * channels;
                strides[0] = itemsize;
            }
            else {
                strides[0] = itemsize * channels;
                strides[ndim - 1] = itemsize;
            }
        }
    }

    if (PyBuffer_FillInfo(view, obj, chunk->abuf, chunk->alen, 0, flags)) {
        return -1;
    }
    if (flags != PyBUF_SIMPLE) {
        view->format = format;
        view->ndim = ndim;
        view->shape = shape;
        view->strides = strides;
        view->itemsize = itemsize;
    }
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

#if HAVE_OLD_BUFPROTO
#define snd_getreadbuffer 0
#define snd_getwritebuffer 0
#define snd_getsegcount 0
#define snd_getcharbuffer 0
#endif

static PyBufferProcs sound_as_buffer[] =
{
    {
#if HAVE_OLD_BUFPROTO
        snd_getreadbuffer,
        snd_getwritebuffer,
        snd_getsegcount,
        snd_getcharbuffer,
#endif
        snd_getbuffer,
        snd_releasebuffer
    }
};
#else
#define sound_as_buffer 0
#endif /* #if EXPORT_BUFFER */


/*sound object internals*/
static void
sound_dealloc (PySoundObject* self)
{
    Mix_Chunk* chunk = PySound_AsChunk ((PyObject*)self);
    if (chunk)
        Mix_FreeChunk (chunk);
    if (self->mem)
        PyMem_Free (self->mem);
    if (self->weakreflist)
        PyObject_ClearWeakRefs ((PyObject*)self);
    Py_TYPE(self)->tp_free ((PyObject*)self);
}

static PyTypeObject PySound_Type =
{
    TYPE_HEAD (NULL, 0)
    "Sound",
    sizeof(PySoundObject),
    0,
    (destructor)sound_dealloc,
    0,
    0,
    0,                          /* setattr */
    0,                          /* compare */
    0,                          /* repr */
    0,                          /* as_number */
    0,                          /* as_sequence */
    0,                          /* as_mapping */
    (hashfunc)NULL,             /* hash */
    (ternaryfunc)NULL,          /* call */
    (reprfunc)NULL,             /* str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    sound_as_buffer,            /* tp_as_buffer */
    (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
    Py_TPFLAGS_HAVE_NEWBUFFER), /* tp_flags */
    DOC_PYGAMEMIXERSOUND,       /* Documentation string */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    offsetof(PySoundObject, weakreflist),
                                /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    sound_methods,              /* tp_methods */
    0,                          /* tp_members */
    sound_getset,               /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    sound_init,                 /* tp_init */
    0,                          /* tp_alloc */
    0,                          /* tp_new */
};

//PyType_GenericNew,                    /* tp_new */

/* channel object methods */
static PyObject*
chan_play (PyObject* self, PyObject* args, PyObject* kwargs)
{
    int channelnum = PyChannel_AsInt (self);
    PyObject* sound;
    Mix_Chunk* chunk;
    int loops = 0, playtime = -1, fade_ms = 0;

    char *kwids[] = { "Sound", "loops", "maxtime", "fade_ms", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|iii", kwids, &PySound_Type, &sound,
                                     &loops, &playtime, &fade_ms))
       return NULL;
    chunk = PySound_AsChunk (sound);

    if (fade_ms > 0)
    {
        channelnum = Mix_FadeInChannelTimed (channelnum, chunk, loops, fade_ms, playtime);
    }
    else
    {
        channelnum = Mix_PlayChannelTimed (channelnum, chunk, loops, playtime);
    }
    if (channelnum != -1)
        Mix_GroupChannel (channelnum, (intptr_t)chunk);

    Py_XDECREF (channeldata[channelnum].sound);
    Py_XDECREF (channeldata[channelnum].queue);
    channeldata[channelnum].sound = sound;
    channeldata[channelnum].queue = NULL;
    Py_INCREF (sound);
    Py_RETURN_NONE;
}

static PyObject*
chan_queue (PyObject* self, PyObject* args)
{
    int channelnum = PyChannel_AsInt (self);
    PyObject* sound;
    Mix_Chunk* chunk;

    if (!PyArg_ParseTuple (args, "O!", &PySound_Type, &sound))
        return NULL;
    chunk = PySound_AsChunk (sound);

    if (!channeldata[channelnum].sound) /*nothing playing*/
    {
        channelnum = Mix_PlayChannelTimed (channelnum, chunk, 0, -1);
        if (channelnum != -1)
            Mix_GroupChannel (channelnum, (intptr_t)chunk);

        channeldata[channelnum].sound = sound;
        Py_INCREF (sound);
    }
    else
    {
        Py_XDECREF (channeldata[channelnum].queue);
        channeldata[channelnum].queue = sound;
        Py_INCREF (sound);
    }
    Py_RETURN_NONE;
}

static PyObject*
chan_get_busy (PyObject* self)
{
    int channelnum = PyChannel_AsInt (self);
    MIXER_INIT_CHECK ();

    return PyInt_FromLong (Mix_Playing (channelnum));
}

static PyObject*
chan_fadeout (PyObject* self, PyObject* args)
{
    int channelnum = PyChannel_AsInt (self);
    int _time;
    if (!PyArg_ParseTuple (args, "i", &_time))
        return NULL;

    MIXER_INIT_CHECK ();

    Mix_FadeOutChannel (channelnum, _time);
    Py_RETURN_NONE;
}

static PyObject*
chan_stop (PyObject* self)
{
    int channelnum = PyChannel_AsInt (self);
    MIXER_INIT_CHECK ();

    Mix_HaltChannel (channelnum);
    Py_RETURN_NONE;
}

static PyObject*
chan_pause (PyObject* self)
{
    int channelnum = PyChannel_AsInt (self);
    MIXER_INIT_CHECK ();

    Mix_Pause (channelnum);
    Py_RETURN_NONE;
}

static PyObject*
chan_unpause (PyObject* self)
{
    int channelnum = PyChannel_AsInt (self);
    MIXER_INIT_CHECK ();

    Mix_Resume (channelnum);
    Py_RETURN_NONE;
}

static PyObject*
chan_set_volume (PyObject* self, PyObject* args)
{
    int channelnum = PyChannel_AsInt (self);
    float volume, stereovolume=-1.11f;
    int result;
    Uint8 left, right;

    if (!PyArg_ParseTuple (args, "f|f", &volume, &stereovolume))
        return NULL;

    MIXER_INIT_CHECK ();
#if MIX_MAJOR_VERSION>=1 && MIX_MINOR_VERSION>=2 && MIX_PATCHLEVEL>=1
    if ((stereovolume <= -1.10f) && (stereovolume >= -1.12f))
    {
        /* The normal volume will be used.  No panning.  so panning is
         * set to full.  this is incase it was set previously to
         * something else.  NOTE: there is no way to GetPanning
         * variables.
         */
        left = 255;
        right = 255;

        if(!Mix_SetPanning(channelnum, left, right)) {
            return RAISE (PyExc_SDLError, Mix_GetError());
        }
    }
    else
    {
        /* NOTE: here the volume will be set to 1.0 and the panning will
         * be used. */
        left = (Uint8)(volume * 255);
        right = (Uint8)(stereovolume * 255);
        /*
        printf("left:%d:  right:%d:\n", left, right);
        */

        if(!Mix_SetPanning(channelnum, left, right)) {
            return RAISE (PyExc_SDLError, Mix_GetError());
        }

        volume = 1.0f;
    }
#else
    if (! ((stereovolume <= -1.10f) && (stereovolume >= -1.12f)))
        volume = (volume + stereovolume) * 0.5f;
#endif

    result = Mix_Volume (channelnum, (int)(volume*128));
    Py_RETURN_NONE;
}

static PyObject*
chan_get_volume (PyObject* self)
{
    int channelnum = PyChannel_AsInt (self);
    int volume;

    MIXER_INIT_CHECK ();

    volume = Mix_Volume (channelnum, -1);

    return PyFloat_FromDouble (volume / 128.0);
}

static PyObject*
chan_get_sound (PyObject* self)
{
    int channelnum = PyChannel_AsInt (self);
    PyObject* sound;

    sound = channeldata[channelnum].sound;
    if (!sound)
        Py_RETURN_NONE;

    Py_INCREF (sound);
    return sound;
}

static PyObject*
chan_get_queue(PyObject* self)
{
    int channelnum = PyChannel_AsInt (self);
    PyObject* sound;

    sound = channeldata[channelnum].queue;
    if (!sound)
        Py_RETURN_NONE;

    Py_INCREF (sound);
    return sound;
}

static PyObject*
chan_set_endevent (PyObject* self, PyObject* args)
{
    int channelnum = PyChannel_AsInt (self);
    int event = SDL_NOEVENT;

    if (!PyArg_ParseTuple (args, "|i", &event))
        return NULL;

    channeldata[channelnum].endevent = event;
    Py_RETURN_NONE;
}

static PyObject*
chan_get_endevent (PyObject* self)
{
    int channelnum = PyChannel_AsInt (self);

    return PyInt_FromLong (channeldata[channelnum].endevent);
}

static PyMethodDef channel_methods[] =
{
    { "play", (PyCFunction) chan_play, METH_VARARGS | METH_KEYWORDS,
      DOC_CHANNELPLAY },
    { "queue", chan_queue, METH_VARARGS, DOC_CHANNELQUEUE },
    { "get_busy", (PyCFunction) chan_get_busy, METH_NOARGS,
      DOC_CHANNELGETBUSY },
    { "fadeout", chan_fadeout, METH_VARARGS, DOC_CHANNELFADEOUT },
    { "stop", (PyCFunction) chan_stop, METH_NOARGS, DOC_CHANNELSTOP },
    { "pause", (PyCFunction) chan_pause, METH_NOARGS, DOC_CHANNELPAUSE },
    { "unpause", (PyCFunction) chan_unpause, METH_NOARGS, DOC_CHANNELUNPAUSE },
    { "set_volume", chan_set_volume, METH_VARARGS, DOC_CHANNELSETVOLUME },
    { "get_volume", (PyCFunction) chan_get_volume, METH_NOARGS,
      DOC_CHANNELGETVOLUME },

    { "get_sound", (PyCFunction) chan_get_sound, METH_NOARGS,
      DOC_CHANNELGETSOUND },
    { "get_queue", (PyCFunction) chan_get_queue, METH_NOARGS,
      DOC_CHANNELGETQUEUE },

    { "set_endevent", chan_set_endevent, METH_VARARGS, DOC_CHANNELSETENDEVENT },
    { "get_endevent", (PyCFunction) chan_get_endevent, METH_NOARGS,
      DOC_CHANNELGETENDEVENT },

    { NULL, NULL, 0, NULL }
};


/* channel object internals */

static void channel_dealloc (PyObject* self)
{
    PyObject_DEL (self);
}


static PyTypeObject PyChannel_Type =
{
    TYPE_HEAD (NULL, 0)
    "Channel",                  /* name */
    sizeof(PyChannelObject),    /* basic size */
    0,                          /* itemsize */
    channel_dealloc,            /* dealloc */
    0,                          /* print */
    0,                          /* getattr */
    0,                          /* setattr */
    0,                          /* compare */
    0,                          /* repr */
    0,                          /* as_number */
    0,                          /* as_sequence */
    0,                          /* as_mapping */
    (hashfunc)0,                /* hash */
    (ternaryfunc)0,             /* call */
    0,                          /* str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    0,                          /* flags */
    DOC_PYGAMEMIXERCHANNEL,     /* Documentation string */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    channel_methods,            /* tp_methods */
    0,                          /* tp_members */
    0,                          /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    0,                          /* tp_init */
    0,                          /* tp_alloc */
    0,                          /* tp_new */
};

/*mixer module methods*/

static PyObject*
get_num_channels (PyObject* self)
{
    MIXER_INIT_CHECK ();
    return PyInt_FromLong (Mix_GroupCount (-1));
}

static PyObject*
set_num_channels (PyObject* self, PyObject* args)
{
    int numchans, i;
    if (!PyArg_ParseTuple (args, "i", &numchans))
        return NULL;

    MIXER_INIT_CHECK ();
    if (numchans > numchanneldata)
    {
        channeldata = (struct ChannelData*)
            realloc (channeldata, sizeof (struct ChannelData) * numchans);
        for (i = numchanneldata; i < numchans; ++i)
        {
            channeldata[i].sound = NULL;
            channeldata[i].queue = NULL;
            channeldata[i].endevent = 0;
        }
        numchanneldata = numchans;
    }

    Mix_AllocateChannels (numchans);
    Py_RETURN_NONE;
}

static PyObject*
set_reserved (PyObject* self, PyObject* args)
{
    int numchans;
    if (!PyArg_ParseTuple (args, "i", &numchans))
        return NULL;

    MIXER_INIT_CHECK ();

    Mix_ReserveChannels (numchans);
    Py_RETURN_NONE;
}

static PyObject*
get_busy (PyObject* self)
{
    if (!SDL_WasInit (SDL_INIT_AUDIO))
        return PyInt_FromLong (0);

    return PyInt_FromLong (Mix_Playing (-1));
}

static PyObject*
Channel (PyObject* self, PyObject* args)
{
    int chan;
    if (!PyArg_ParseTuple (args, "i", &chan))
        return NULL;

    MIXER_INIT_CHECK ();
    return PyChannel_New (chan);
}


static PyObject*
mixer_find_channel (PyObject* self, PyObject* args)
{
    int chan, force = 0;
    if (!PyArg_ParseTuple (args, "|i", &force))
        return NULL;

    MIXER_INIT_CHECK ();

    chan = Mix_GroupAvailable (-1);
    if (chan == -1)
    {
        if (!force)
            Py_RETURN_NONE;
        chan = Mix_GroupOldest (-1);
    }
    return PyChannel_New (chan);
}

static PyObject*
mixer_fadeout (PyObject* self, PyObject* args)
{
    int _time;
    if (!PyArg_ParseTuple (args, "i", &_time))
        return NULL;

    MIXER_INIT_CHECK ();

    Mix_FadeOutChannel (-1, _time);
    Py_RETURN_NONE;
}

static PyObject*
mixer_stop (PyObject* self)
{
    MIXER_INIT_CHECK ();

    Mix_HaltChannel (-1);
    Py_RETURN_NONE;
}

static PyObject*
mixer_pause (PyObject* self)
{
    MIXER_INIT_CHECK ();

    Mix_Pause (-1);
    Py_RETURN_NONE;
}

static PyObject*
mixer_unpause (PyObject* self)
{
    MIXER_INIT_CHECK ();

    Mix_Resume (-1);
    Py_RETURN_NONE;
}

static int
_chunk_from_buf(const void *buf, Py_ssize_t len, Mix_Chunk **chunk, Uint8 **mem)
{
    Uint8 *m = (Uint8 *)PyMem_Malloc((size_t)len);

    if (!m)
    {
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
                  Py_ssize_t *shape, Py_ssize_t *strides,
                  Mix_Chunk **chunk, Uint8 **mem)
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
    Py_ssize_t loop1, loop2, step1, step2, length, length2=0;

    if (!Mix_QuerySpec(&freq, &format, &channels)) {
        RAISE(PyExc_SDLError, "Mixer not initialized");
        return -1;
    }

    /* Check for compatible values.
     */
    if (channels == 1) {
        if (ndim != 1) {
            RAISE(PyExc_ValueError,
                 "Array must be 1-dimensional for mono mixer");
            return -1;
        }
    }
    else {
        if (ndim != 2) {
            RAISE(PyExc_ValueError,
                  "Array must be 2-dimensional for stereo mixer");
            return -1;
        }
        if (shape[1] != channels) {
            RAISE(PyExc_ValueError,
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
        PyErr_Format(PyExc_ValueError,
                     "Unsupported integer size %d", view_itemsize);
        return -1;
    }
    length = shape[0];
    step1 = strides ? strides[0] : view_itemsize * channels;
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
    if (step1 == itemsize * channels && step2 == itemsize) {
        /*OPTIMIZATION: in these cases, we don't need to loop through
         *the samples individually, because the bytes are already layed
         *out correctly*/
        memcpy(dst, buf, memsize);
    }
    else if (itemsize == 1) {
        for (loop1 = 0; loop1 < length; loop1++) {
            src = (Uint8*)buf + (loop1 * step1);
            switch (view_itemsize) {

            case 1:
                for (loop2=0; loop2<length2; loop2++, dst+=1, src+=step2) {
                    *(Uint8*)dst = (Uint8)*((Uint8*)src);
                }
                break;
            case 2:
                for (loop2=0; loop2<length2; loop2++, dst+=1, src+=step2) {
                    *(Uint8*)dst = (Uint8)*((Uint16*)src);
                }
                break;
            case 4:
                for (loop2=0; loop2<length2; loop2++, dst+=1, src+=step2) {
                    *(Uint8*)dst = (Uint8)*((Uint32*)src);
                }
                break;
            }
        }
    }
    else {
        for (loop1 = 0; loop1 < length; loop1++)
        {
            src = (Uint8*)buf + (loop1 *step1);
            switch (view_itemsize) {

            case 1:
                for (loop2=0; loop2<length2; loop2++, dst+=2, src+=step2) {
                    *(Uint16*)dst = (Uint16)(*((Uint8*)src)<<8);
                }
                break;
            case 2:
                for (loop2=0; loop2<length2; loop2++, dst+=2, src+=step2) {
                    *(Uint16*)dst = (Uint16)*((Uint16*)src);
                }
                break;
            case 4:
                for (loop2=0; loop2<length2; loop2++, dst+=2, src+=step2) {
                    *(Uint16*)dst = (Uint16)*((Uint32*)src);
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

    ((PySoundObject *)self)->chunk = NULL;
    ((PySoundObject *)self)->mem = NULL;

    /* Process arguments, returning cleaner error messages than
       PyArg_ParseTupleAndKeywords would.
    */
    if (arg != NULL && PyTuple_GET_SIZE(arg)) {
        if ((kwarg != NULL && PyDict_Size(kwarg)) || /* conditional and */
            PyTuple_GET_SIZE(arg) != 1)              {
            RAISE(PyExc_TypeError, arg_cnt_err_msg);
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
            RAISE(PyExc_TypeError, arg_cnt_err_msg);
            return -1;
        }
        if ((file = PyDict_GetItemString(kwarg, "file")) == NULL &&
            (buffer = PyDict_GetItemString(kwarg, "buffer")) == NULL &&
            (array = PyDict_GetItemString(kwarg, "array")) == NULL     ) {
            keys = PyDict_Keys(kwarg);
            if (keys == NULL) {
                return -1;
            }
            kencoded = RWopsEncodeString(PyList_GET_ITEM(keys, 0),
                                         NULL, NULL, NULL);
            Py_DECREF(keys);
            if (kencoded == NULL) {
                return -1;
            }
            PyErr_Format(PyExc_TypeError,
                         "Unrecognized keyword argument '%.1024s'",
                         Bytes_AS_STRING(kencoded));
            Py_DECREF(kencoded);
            return -1;
        }
        if (buffer != NULL && PyUnicode_Check(buffer)) { /* conditional and */
            RAISE(PyExc_TypeError,
                  "Unicode object not allowed as buffer object");
            return -1;
        }
    }
    else {
        RAISE(PyExc_TypeError, arg_cnt_err_msg);
        return -1;
    }

    if (file != NULL) {
        rw = RWopsFromObject(file);
        if (rw == NULL) {
            /* RWopsFromObject only raises critical Python exceptions,
               so automatically pass them on.
            */
            return -1;
        }
        if (RWopsCheckObject(rw)) {
            chunk = Mix_LoadWAV_RW(rw, 1);
        }
        else {
            Py_BEGIN_ALLOW_THREADS;
            chunk = Mix_LoadWAV_RW(rw, 1);
            Py_END_ALLOW_THREADS;
        }
        if (chunk == NULL && obj == NULL) {
            obj = RWopsEncodeString(file, NULL, NULL, NULL);
            if (obj != NULL) {
                if (obj == Py_None) {
                    RAISE(PyExc_SDLError, SDL_GetError());
                }
                else {
                    PyErr_Format(PyExc_SDLError,
                                 "Unable to open file '%s'",
                                 Bytes_AS_STRING(obj));
                }
                Py_XDECREF(obj);
            }
            return -1;
        }
    }

#if HAVE_NEW_BUFPROTO
#if PY2
    if (!chunk && buffer && /* conditional and */
        PyObject_CheckBuffer(buffer)) {
#else
    if (!chunk && buffer) {
#endif
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
            ((PySoundObject *)self)->mem = mem;
        }
    }
#endif

#if PY2
    if (chunk == NULL && buffer != NULL) {
        const void *buf = NULL;
        Py_ssize_t buflen = 0;

        if (PyObject_AsReadBuffer(buffer, &buf, &buflen)) {
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
            if (_chunk_from_buf(buf, buflen, &chunk, &mem)) {
                return -1;
            }
            ((PySoundObject *)self)->mem = mem;
        }
    }
#endif

#if HAVE_NEW_BUFPROTO
    if (array != NULL && /* conditional and */
        PyObject_CheckBuffer(array)) {
        Py_buffer view;
        PG_sample_format_t view_format;
        int rcode;

        view.itemsize = 0;
        view.obj = 0;
        if (PyObject_GetBuffer(array, &view, PyBUF_FORMAT | PyBUF_ND)) {
            return -1;
        }
        view_format = _format_view_to_audio(&view);
        if (!view_format) {
            PyBuffer_Release(&view);
            return -1;
        }
        rcode = _chunk_from_array(view.buf, view_format, view.ndim,
                                  view.shape, view.strides,
                                  &chunk, &mem);
        PyBuffer_Release(&view);
        if (rcode) {
            return -1;
        }
        ((PySoundObject *)self)->mem = mem;
    }
#endif

    if (chunk == NULL && array != NULL) {
        PyArrayInterface *inter = 0;
        PyObject *cobj = 0;
        PG_sample_format_t array_format;
        int rcode;

        if (_get_array_interface(array, &cobj, &inter)) {
            return -1;
        }
        if (!(inter->flags & PAI_CONTIGUOUS)) {
            RAISE(PyExc_ValueError,
                  "Array is discontiguous");
            Py_DECREF(cobj);
            return -1;
        }
        if (inter->nd > 1 && inter->flags & PAI_FORTRAN) {
            RAISE(PyExc_ValueError, "Array is channel first");
            Py_DECREF(cobj);
            return -1;
        }
        array_format = _format_inter_to_audio(inter);
        if (!array_format) {
            Py_DECREF(cobj);
            return -1;
        }
        rcode = _chunk_from_array(inter->data, array_format, inter->nd,
                                  inter->shape, inter->strides,
                                  &chunk, &mem);
        Py_DECREF(cobj);
        if (rcode) {
            return -1;
        }
        ((PySoundObject *)self)->mem = mem;
    }

    if (chunk == NULL) {
        PyErr_Format(PyExc_TypeError,
                     "Unrecognized argument (type %s)",
                     Py_TYPE(obj)->tp_name);
        return -1;
    }

    ((PySoundObject *)self)->chunk = chunk;
    return 0;
}

static PyMethodDef _mixer_methods[] =
{
    { "__PYGAMEinit__", autoinit, METH_VARARGS, "auto initialize for mixer" },
    { "init", (PyCFunction) init, METH_VARARGS | METH_KEYWORDS,
      DOC_PYGAMEMIXERINIT },
    { "quit", (PyCFunction) quit, METH_NOARGS, DOC_PYGAMEMIXERQUIT },
    { "get_init", (PyCFunction) get_init, METH_NOARGS, DOC_PYGAMEMIXERGETINIT },
    { "pre_init", (PyCFunction) pre_init, METH_VARARGS | METH_KEYWORDS,
      DOC_PYGAMEMIXERPREINIT },
    { "get_num_channels", (PyCFunction) get_num_channels, METH_NOARGS,
      DOC_PYGAMEMIXERGETNUMCHANNELS },
    { "set_num_channels", set_num_channels, METH_VARARGS,
      DOC_PYGAMEMIXERSETNUMCHANNELS },
    { "set_reserved", set_reserved, METH_VARARGS, DOC_PYGAMEMIXERSETRESERVED },

    { "get_busy", (PyCFunction) get_busy, METH_NOARGS, DOC_PYGAMEMIXERGETBUSY },
    { "Channel", Channel, METH_VARARGS, DOC_PYGAMEMIXERCHANNEL },
    { "find_channel", mixer_find_channel, METH_VARARGS,
      DOC_PYGAMEMIXERFINDCHANNEL },
    { "fadeout", mixer_fadeout, METH_VARARGS, DOC_PYGAMEMIXERFADEOUT },
    { "stop", (PyCFunction) mixer_stop, METH_NOARGS, DOC_PYGAMEMIXERSTOP },
    { "pause", (PyCFunction) mixer_pause, METH_NOARGS, DOC_PYGAMEMIXERPAUSE },
    { "unpause", (PyCFunction) mixer_unpause, METH_NOARGS,
      DOC_PYGAMEMIXERUNPAUSE },
/*  { "lookup_frequency", lookup_frequency, 1, doc_lookup_frequency },*/

    { NULL, NULL, 0, NULL }
};

static PyObject*
PySound_New (Mix_Chunk* chunk)
{
    PySoundObject* soundobj;

    if (!chunk)
        return RAISE (PyExc_RuntimeError, "unable to create sound.");

    soundobj = (PySoundObject *)PySound_Type.tp_new (&PySound_Type, NULL, NULL);
    if (soundobj)
    {
        soundobj->mem = NULL;
        soundobj->chunk = chunk;
    }

    return (PyObject*)soundobj;
}

static PyObject*
PyChannel_New (int channelnum)
{
    PyChannelObject* chanobj;

    if (channelnum < 0 || channelnum >= Mix_GroupCount (-1))
        return RAISE (PyExc_IndexError, "invalid channel index");

    chanobj = PyObject_NEW (PyChannelObject, &PyChannel_Type);
    if (!chanobj)
        return NULL;

    chanobj->chan = channelnum;
    return (PyObject*)chanobj;
}

MODINIT_DEFINE (mixer)
{
    PyObject *module, *dict, *apiobj, *music=NULL;
    int ecode;
    static void* c_api[PYGAMEAPI_MIXER_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "mixer",
        DOC_PYGAMEMIXER,
        -1,
        _mixer_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */

    /*imported needed apis*/
    import_pygame_base ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }
    import_pygame_rwobject ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }
    import_pygame_bufferproxy ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready (&PySound_Type) < 0) {
        MODINIT_ERROR;
    }
    if (PyType_Ready (&PyChannel_Type) < 0) {
        MODINIT_ERROR;
    }

    /* create the module */
    PySound_Type.tp_new = &PyType_GenericNew;
#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 (MODPREFIX "mixer", _mixer_methods, DOC_PYGAMEMIXER);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict (module);

    if (PyDict_SetItemString (dict,
                              "Sound",
                              (PyObject *)&PySound_Type) < 0) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    if (PyDict_SetItemString (dict,
                              "SoundType",
                              (PyObject *)&PySound_Type) < 0) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    if (PyDict_SetItemString (dict,
                              "ChannelType",
                              (PyObject *)&PyChannel_Type) < 0) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    /* export the c api */
    c_api[0] = &PySound_Type;
    c_api[1] = PySound_New;
    c_api[2] = snd_play;
    c_api[3] = &PyChannel_Type;
    c_api[4] = PyChannel_New;
    c_api[5] = autoinit;
    c_api[6] = autoquit;
    apiobj = encapsulate_api (c_api, "mixer");
    if (apiobj == NULL) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString (dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF (apiobj);
    if (ecode < 0) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    music = PyImport_ImportModule(IMPPREFIX "mixer_music");
    if (music == NULL) {
        PyErr_Clear ();
        /* try loading it under this name...
        */
        music = PyImport_ImportModule(RELATIVE_MODULE("mixer_music"));
        /*printf("NOTE3: here in mixer.c...\n");
         */
    }

    if (music != NULL) {
        PyObject* ptr, *_dict;
        /* printf("NOTE: failed loading pygame.mixer_music in src/mixer.c\n");
         */
        if (PyModule_AddObject (module, "music", music) < 0) {
            DECREF_MOD (module);
            Py_DECREF (music);
            MODINIT_ERROR;
        }
        _dict = PyModule_GetDict (music);
        ptr = PyDict_GetItemString (_dict, "_MUSIC_POINTER");
        current_music =
            (Mix_Music**)PyCapsule_GetPointer (ptr, "pygame.music_mixer."
                                                    "_MUSIC_POINTER");
        ptr = PyDict_GetItemString (_dict, "_QUEUE_POINTER");
        queue_music =
            (Mix_Music**)PyCapsule_GetPointer (ptr, "pygame.music_mixer."
                                                    "_QUEUE_POINTER");
    }
    else /*music module not compiled? cleanly ignore*/
    {
        current_music = NULL;
        PyErr_Clear ();
    }
    MODINIT_RETURN (module);
}
