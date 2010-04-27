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
#define PYGAME_SDLMIXER_INTERNAL

#include "pgmixer.h"
#include "pgsdl.h"
#include "sdlmixerchannel_doc.h"

static PyObject* _channel_allocate (PyObject *self, PyObject *args);
static PyObject* _channel_opened (PyObject *self);
static PyObject* _channel_setvolume (PyObject *self, PyObject *args);
static PyObject* _channel_getvolume (PyObject *self);
static PyObject* _channel_pause (PyObject *self);
static PyObject* _channel_resume (PyObject *self);
static PyObject* _channel_halt (PyObject *self);
static PyObject* _channel_playing (PyObject *self);
static PyObject* _channel_paused (PyObject *self);
static PyObject* _channel_expire (PyObject *self, PyObject *args);
static PyObject* _channel_fadeout (PyObject *self, PyObject *args);

static PyMethodDef _channel_methods[] = {
    { "allocate", _channel_allocate, METH_O, DOC_CHANNEL_ALLOCATE },
    { "opened", (PyCFunction) _channel_opened, METH_NOARGS,
      DOC_CHANNEL_OPENED },
    { "set_volume", _channel_setvolume, METH_O, DOC_CHANNEL_SET_VOLUME },
    { "get_volume", (PyCFunction) _channel_getvolume, METH_NOARGS,
      DOC_CHANNEL_GET_VOLUME },
    { "pause", (PyCFunction) _channel_pause, METH_NOARGS, DOC_CHANNEL_PAUSE },
    { "resume", (PyCFunction) _channel_resume, METH_NOARGS,
      DOC_CHANNEL_RESUME },
    { "halt", (PyCFunction) _channel_halt, METH_NOARGS, DOC_CHANNEL_HALT },
    { "expire", _channel_expire, METH_O, DOC_CHANNEL_EXPIRE },
    { "fade_out", _channel_fadeout, METH_O, DOC_CHANNEL_FADE_OUT },
    { "playing", (PyCFunction) _channel_playing, METH_NOARGS,
      DOC_CHANNEL_PLAYING },
    { "paused", (PyCFunction) _channel_paused, METH_NOARGS,
      DOC_CHANNEL_PAUSED },
    { NULL, NULL, 0, NULL },
};

static PyObject*
_channel_allocate (PyObject *self, PyObject *args)
{
    PyObject *tuple, *chan;
    int chans, i;
    
    ASSERT_MIXER_OPEN (NULL);
    
    if (!IntFromObj (args, &chans))
        return NULL;
    if (chans < 0)
    {
        PyErr_SetString (PyExc_TypeError, "channels must not be negative");
        return NULL;
    }
    chans = Mix_AllocateChannels (chans);
    tuple = PyTuple_New (chans);
    if (!tuple)
        return NULL;
    
    for (i = 0; i < chans; i++)
    {
        chan = PyChannel_NewFromIndex (i);
        if (!chan)
        {
            Py_DECREF (tuple);
            return NULL;
        }
        PyTuple_SET_ITEM (tuple, (Py_ssize_t) i, chan);
    }
    return tuple;
}

static PyObject*
_channel_opened (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);
    return PyInt_FromLong (Mix_AllocateChannels (-1));
}

static PyObject*
_channel_setvolume (PyObject *self, PyObject *args)
{
    int volume;
    
    ASSERT_MIXER_OPEN (NULL);
    
    if (!IntFromObj (args, &volume))
        return NULL;
    if (volume < 0 || volume > MIX_MAX_VOLUME)
    {
        PyErr_SetString (PyExc_ValueError, "volume must be in the range 0-128");
        return NULL;
    }
    return PyInt_FromLong (Mix_Volume (-1, volume));
}

static PyObject*
_channel_getvolume (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);
    return PyInt_FromLong (Mix_Volume (-1, -1));
}

static PyObject*
_channel_pause (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);
    Mix_Pause (-1);
    Py_RETURN_NONE;
}

static PyObject*
_channel_resume (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);
    Mix_Resume (-1);
    Py_RETURN_NONE;
}

static PyObject*
_channel_halt (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);
    Mix_HaltChannel (-1);
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
    Mix_ExpireChannel (-1, ms);
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
    return PyInt_FromLong (Mix_FadeOutChannel (-1, ms));
}

static PyObject*
_channel_playing (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);
    return PyInt_FromLong (Mix_Playing (-1));
}

static PyObject*
_channel_paused (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);
    return PyInt_FromLong (Mix_Paused (-1));
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_channel (void)
#else
PyMODINIT_FUNC initchannel (void)
#endif
{
    PyObject *mod;

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "channel",
        DOC_CHANNEL,
        -1,
        _channel_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("channel", _channel_methods, DOC_CHANNEL);
#endif
    if (!mod)
        goto fail;

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    if (import_pygame2_sdlmixer_base ()<0)
        goto fail;
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
