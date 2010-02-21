/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2008 Marcus von Appen

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

*/
#define PYGAME_SDLMIXERCHANNEL_INTERNAL

#include "mixermod.h"
#include "pgsdl.h"
#include "pgmixer.h"
#include "sdlmixerbase_doc.h"

static PyObject* _channel_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _channel_init (PyObject *chunk, PyObject *args, PyObject *kwds);
static void _channel_dealloc (PyChannel *self);

static PyObject* _channel_play (PyObject *self, PyObject *args, PyObject *kwds);
static PyObject* _channel_pause (PyObject *self);
static PyObject* _channel_halt (PyObject *self);
static PyObject* _channel_resume (PyObject *self);
static PyObject* _channel_fadein (PyObject *self, PyObject *args,
    PyObject *kwds);
static PyObject* _channel_fadeout (PyObject *self, PyObject *args);
static PyObject* _channel_expire (PyObject *self, PyObject *args);

static PyObject* _channel_getchunk (PyObject *self, void *closure);
static PyObject* _channel_getvolume (PyObject *self, void *closure);
static int _channel_setvolume (PyObject *self, PyObject *value, void *closure);
static PyObject* _channel_getplaying (PyObject *self, void *closure);
static PyObject* _channel_getpaused (PyObject *self, void *closure);
static PyObject* _channel_getfading (PyObject *self, void *closure);

/**
 */
static PyMethodDef _channel_methods[] = {
    { "play", (PyCFunction)_channel_play, METH_VARARGS | METH_KEYWORDS,
      DOC_BASE_CHANNEL_PLAY },
    { "pause", (PyCFunction) _channel_pause, METH_NOARGS,
      DOC_BASE_CHANNEL_PAUSE },
    { "halt", (PyCFunction) _channel_halt, METH_NOARGS, DOC_BASE_CHANNEL_HALT },
    { "resume", (PyCFunction) _channel_resume, METH_NOARGS,
      DOC_BASE_CHANNEL_RESUME },
    { "fade_in", (PyCFunction)_channel_fadein, METH_VARARGS | METH_KEYWORDS,
      DOC_BASE_CHANNEL_FADE_IN },
    { "fade_out", _channel_fadeout, METH_O, DOC_BASE_CHANNEL_FADE_OUT },
    { "expire", _channel_expire, METH_O, DOC_BASE_CHANNEL_EXPIRE },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _channel_getsets[] = {
    { "chunk", _channel_getchunk, NULL, DOC_BASE_CHANNEL_CHUNK, NULL },
    { "volume", _channel_getvolume, _channel_setvolume,
      DOC_BASE_CHANNEL_VOLUME, NULL },
    { "playing", _channel_getplaying, NULL, DOC_BASE_CHANNEL_PLAYING, NULL },
    { "paused", _channel_getpaused, NULL, DOC_BASE_CHANNEL_PAUSED, NULL },
    { "fading", _channel_getfading, NULL, DOC_BASE_CHANNEL_FADING, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PyChannel_Type =
{
    TYPE_HEAD(NULL,0)
    "sdlmixer.Channel",         /* tp_name */
    sizeof (PyChannel),         /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _channel_dealloc,   /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    0,                          /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    DOC_BASE_CHANNEL,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _channel_methods,           /* tp_methods */
    0,                          /* tp_members */
    _channel_getsets,           /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _channel_init,   /* tp_init */
    0,                          /* tp_alloc */
    _channel_new,               /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0,                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0                           /* tp_version_tag */
#endif
};

static void
_channel_dealloc (PyChannel *self)
{
    Mix_HaltChannel (self->channel);

    if (self->playchunk)
    {
        ((PyChunk*)self->playchunk)->playchannel = -1;
        Py_DECREF (self->playchunk);
    }
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static PyObject*
_channel_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyChannel *channel = (PyChannel *) type->tp_alloc (type, 0);
    if (!channel)
        return NULL;

    channel->channel = -1;
    channel->playchunk = NULL;
    return (PyObject*) channel;
}

static int
_channel_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    int channel = 0, check, amount;
    
    ASSERT_MIXER_OPEN(-1);

    if (!PyArg_ParseTuple (args, "|i", &channel))
        return -1;

    if (channel <= 0) /* The user wants a new channel */
        channel = -1;

    /* Query the channels */
    amount = Mix_AllocateChannels (-1);
        
    if (channel == -1)
    {
        /* Keep all channels and allocate a new one, if the user want it
         * that way, otherwise return the matching channel. */
        check = Mix_AllocateChannels (amount + 1);
        if (check != amount + 1)
        {
            PyErr_SetString (PyExc_PyGameError, "channel allocation mismatch");
            return -1;
        }
    }
    return 0;
}

/* Getters/Setters */
static PyObject*
_channel_getchunk (PyObject *self, void *closure)
{
    ASSERT_MIXER_OPEN(NULL);

    /* Alternative to Mix_GetChunk */
    if (((PyChannel*)self)->playchunk)
    {
        Py_INCREF (((PyChannel*)self)->playchunk);
        return ((PyChannel*)self)->playchunk;
    }
    Py_RETURN_NONE;
}

static PyObject*
_channel_getvolume (PyObject *self, void *closure)
{
    ASSERT_MIXER_OPEN(NULL);
    return PyInt_FromLong (Mix_Volume (((PyChannel*)self)->channel, -1));
}

static int
_channel_setvolume (PyObject *self, PyObject *value, void *closure)
{
    int volume;

    ASSERT_MIXER_OPEN(-1);

    if (!IntFromObj (value, &volume))
        return -1;

    if (volume < 0 || volume > MIX_MAX_VOLUME)
    {
        PyErr_SetString (PyExc_ValueError, "volume must be in the range 0-128");
        return -1;
    }
    Mix_Volume (((PyChannel*)self)->channel, volume);
    return 0;
}

static PyObject*
_channel_getplaying (PyObject *self, void *closure)
{
    ASSERT_MIXER_OPEN(NULL);
    return PyBool_FromLong (Mix_Playing (((PyChannel*)self)->channel));
}

static PyObject*
_channel_getpaused (PyObject *self, void *closure)
{
    ASSERT_MIXER_OPEN(NULL);
    return PyBool_FromLong (Mix_Paused (((PyChannel*)self)->channel));
}

static PyObject*
_channel_getfading (PyObject *self, void *closure)
{
    ASSERT_MIXER_OPEN(NULL);
    return PyLong_FromUnsignedLong (Mix_FadingChannel
        (((PyChannel*)self)->channel));
}

/* Methods */
static PyObject*
_channel_play (PyObject *self, PyObject *args, PyObject *kwds)
{
    int loops = -1, ticks = -1, retval;
    PyObject *obj;
    PyChannel *chan = (PyChannel*) self;
    PyChunk *chunk;

    static char *kwlist[] = { "sound", "loops", "ticks", NULL };
    
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "O|ii:play", kwlist, &obj,
        &loops, &ticks))
        return NULL;

    if (!PyChunk_Check (obj))
    {
        PyErr_SetString (PyExc_TypeError, "chunk must be a Chunk");
        return NULL;
    }

    chunk = (PyChunk*) obj;
    loops = MAX (loops, -1);
    ticks = MAX (ticks, -1);

    if (ticks == -1)
        retval = Mix_PlayChannel (chan->channel, chunk->chunk, loops);
    else
        retval = Mix_PlayChannelTimed (chan->channel, chunk->chunk, loops,
            ticks);

    if (retval == -1)
    {
        PyErr_SetString (PyExc_PyGameError, Mix_GetError ());
        return NULL;
    }

    /* Set the chunk's playing channel. */
    if (chan->playchunk)
    {
        ((PyChunk*)chan->playchunk)->playchannel = -1;
        Py_DECREF (chan->playchunk);
    }
    chunk->playchannel = chan->channel;
    chan->playchunk = obj;
    Py_INCREF (chan->playchunk);

    Py_RETURN_NONE;
}

static PyObject*
_channel_pause (PyObject *self)
{
    Mix_Pause (((PyChannel*)self)->channel);
    Py_RETURN_NONE;
}

static PyObject*
_channel_halt (PyObject *self)
{
    PyChannel *chan = (PyChannel*)self;

    Mix_HaltChannel (chan->channel);
    if (chan->playchunk)
    {
        ((PyChunk*)chan->playchunk)->playchannel = -1;
        Py_DECREF (chan);
    }
    
    Py_RETURN_NONE;
}

static PyObject*
_channel_resume (PyObject *self)
{
    Mix_Resume (((PyChannel*)self)->channel);
    Py_RETURN_NONE;
}

static PyObject*
_channel_fadein (PyObject *self, PyObject *args, PyObject *kwds)
{
    int loops = -1, ticks = -1, ms, retval;
    PyObject *obj;
    PyChannel *chan = (PyChannel*) self;
    PyChunk *chunk;

    static char *kwlist[] = { "sound", "ms", "loops", "ticks", NULL };
    
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "Oi|ii:fade_in", kwlist,
        &obj, &ms, &loops, &ticks))
        return NULL;

    if (ms < 0)
    {
        PyErr_SetString (PyExc_ValueError, "ms must not be negative");
        return NULL;
    }

    if (!PyChunk_Check (obj))
    {
        PyErr_SetString (PyExc_TypeError, "chunk must be a Chunk");
        return NULL;
    }

    chunk = (PyChunk*) obj;
    loops = MAX (loops, -1);
    ticks = MAX (ticks, -1);

    if (ticks == -1)
        retval = Mix_FadeInChannel (chan->channel, chunk->chunk, loops, ms);
    else
        retval = Mix_FadeInChannelTimed (chan->channel, chunk->chunk, loops, ms,
            ticks);

    if (retval == -1)
    {
        PyErr_SetString (PyExc_PyGameError, Mix_GetError ());
        return NULL;
    }

    /* Set the chunk's playing channel. */
    if (chan->playchunk)
    {
        ((PyChunk*)chan->playchunk)->playchannel = -1;
        Py_DECREF (chan->playchunk);
    }
    chunk->playchannel = chan->channel;
    chan->playchunk = obj;
    Py_INCREF (chan->playchunk);

    Py_RETURN_NONE;
}

static PyObject*
_channel_fadeout (PyObject *self, PyObject *args)
{
    int ms;
    
    ASSERT_MIXER_OPEN (NULL);

    if (!IntFromObj (args, &ms))
        return NULL;
    if (ms < 0)
    {
        PyErr_SetString (PyExc_ValueError, "ms must not be negative");
        return NULL;
    }
    Mix_FadeOutChannel (((PyChannel*)self)->channel, ms);
    Py_RETURN_NONE;
}

static PyObject*
_channel_expire (PyObject *self, PyObject *args)
{
    int ms;
    
    ASSERT_MIXER_OPEN (NULL);
    
    if (!IntFromObj (args, &ms))
        return NULL;
    if (ms < 0)
    {
        PyErr_SetString (PyExc_ValueError, "ms must not be negative");
        return NULL;
    }
    Mix_ExpireChannel (((PyChannel*)self)->channel, ms);
    Py_RETURN_NONE;
}


/* C API */
PyObject*
PyChannel_New (void)
{
    PyChannel *channel;
    int id;
    
    ASSERT_MIXER_OPEN(NULL);
    
    channel = (PyChannel*) PyChannel_Type.tp_new (&PyChannel_Type, NULL, NULL);
    if (!channel)
        return NULL;

    id = Mix_AllocateChannels (-1);
    Mix_AllocateChannels (id + 1);
    
    channel->channel = id;
    return (PyObject*) channel;
}

PyObject*
PyChannel_NewFromIndex (int _index)
{
    int last;
    PyChannel *channel;
    
    ASSERT_MIXER_OPEN(NULL);

    last = Mix_AllocateChannels (-1);
    if (last < _index)
        Mix_AllocateChannels (_index + 1);
    
    channel = (PyChannel*) PyChannel_Type.tp_new (&PyChannel_Type, NULL, NULL);
    if (!channel)
        return NULL;
    channel->channel = _index;
    return (PyObject*) channel;

}

void
channel_export_capi (void **capi)
{
    capi[PYGAME_SDLMIXERCHANNEL_FIRSTSLOT] = &PyChannel_Type;
    capi[PYGAME_SDLMIXERCHANNEL_FIRSTSLOT+1] = (void *)PyChannel_New;
    capi[PYGAME_SDLMIXERCHANNEL_FIRSTSLOT+2] = (void *)PyChannel_NewFromIndex;
}
