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
#define PYGAME_SDLMIXERMUSIC_INTERNAL

#include "mixermod.h"
#include "pgsdl.h"
#include "pgmixer.h"
#include "sdlmixerbase_doc.h"

static PyObject* _music_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _music_init (PyObject *music, PyObject *args, PyObject *kwds);
static void _music_dealloc (PyMusic *self);

static PyObject* _music_play (PyObject *self, PyObject *args);
static PyObject* _music_fadein (PyObject *self, PyObject *args, PyObject *kwds);

static PyObject* _music_gettype (PyObject *self, void *closure);

/**
 */
static PyMethodDef _music_methods[] = {
    { "play", _music_play, METH_VARARGS, DOC_BASE_MUSIC_PLAY },
    { "fade_in", (PyCFunction) _music_fadein, METH_VARARGS | METH_KEYWORDS,
      DOC_BASE_MUSIC_FADE_IN },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _music_getsets[] = {
    { "type", _music_gettype, NULL, DOC_BASE_MUSIC_TYPE, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PyMusic_Type =
{
    TYPE_HEAD(NULL,0)
    "sdlmixer.Music",           /* tp_name */
    sizeof (PyMusic),           /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _music_dealloc,   /* tp_dealloc */
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
    DOC_BASE_MUSIC,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _music_methods,             /* tp_methods */
    0,                          /* tp_members */
    _music_getsets,             /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _music_init,     /* tp_init */
    0,                          /* tp_alloc */
    _music_new,                 /* tp_new */
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
_music_dealloc (PyMusic *self)
{
    if (self->music)
        Mix_FreeMusic (self->music);
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static PyObject*
_music_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyMusic *music = (PyMusic*) type->tp_alloc (type, 0);
    if (!music)
        return NULL;
    music->music = NULL;
    return (PyObject*) music;
}

static int
_music_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *file;
    Mix_Music *music;
    SDL_RWops *rw;
    int autoclose;
    
    ASSERT_MIXER_OPEN (-1);
    
    if (!PyArg_ParseTuple (args, "O", &file))
        return -1;

    rw = PyRWops_NewRO_Threaded (file, &autoclose);
    if (!rw)
        return -1;

    Py_BEGIN_ALLOW_THREADS;
    music = Mix_LoadMUS_RW (rw);
    Py_END_ALLOW_THREADS;

    if (!autoclose)
        PyRWops_Close (rw, autoclose);
    
    if (!music)
    {
        PyErr_SetString (PyExc_PyGameError, Mix_GetError ());
        return -1;
    }

    ((PyMusic*)self)->music = music;
    return 0;
}

/* Getters/Setters */
static PyObject*
_music_gettype (PyObject *self, void *closure)
{
    ASSERT_MIXER_OPEN(NULL);
    return PyLong_FromUnsignedLong (Mix_GetMusicType (((PyMusic*)self)->music));
}

/* Methods */
static PyObject*
_music_play (PyObject *self, PyObject *args)
{
    int loops = -1, ret;

    ASSERT_MIXER_OPEN(NULL);

    if (!PyArg_ParseTuple (args, "|i:play", &loops))
        return NULL;

    Py_BEGIN_ALLOW_THREADS;
    ret = Mix_PlayMusic (((PyMusic*)self)->music, loops);
    Py_END_ALLOW_THREADS;

    if (ret == -1)
    {
        PyErr_SetString (PyExc_PyGameError, Mix_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_music_fadein (PyObject *self, PyObject *args, PyObject *kwds)
{
    int loops = -1, ms, retval;
    double pos = 0;

    static char *kwlist[] = { "ms", "loops", "pos", NULL };
    
    ASSERT_MIXER_OPEN(NULL);

    if (!PyArg_ParseTupleAndKeywords (args, kwds, "i|id:fade_in", kwlist, &ms,
        &loops, &pos))
        return NULL;

    if (ms < 0)
    {
        PyErr_SetString (PyExc_ValueError, "ms must not be negative");
        return NULL;
    }

    pos = MAX (pos, 0);
    loops = MAX (loops, -1);

    Py_BEGIN_ALLOW_THREADS;
    if (pos == 0)
        retval = Mix_FadeInMusic (((PyMusic*)self)->music, loops, ms);
    else
        retval = Mix_FadeInMusicPos (((PyMusic*)self)->music, loops, ms, pos);
    Py_END_ALLOW_THREADS;

    if (retval == -1)
    {
        PyErr_SetString (PyExc_PyGameError, Mix_GetError ());
        return NULL;
    }
    
    Py_RETURN_NONE;
}

/* C API */
PyObject*
PyMusic_New (char *filename)
{
    PyMusic *music;
    Mix_Music *sample;

    ASSERT_MIXER_OPEN(NULL);

    if (!filename)
        return NULL;
    
    music = (PyMusic*) PyMusic_Type.tp_new (&PyMusic_Type, NULL, NULL);
    if (!music)
        return NULL;
    Py_BEGIN_ALLOW_THREADS;
    sample = Mix_LoadMUS (filename);
    Py_END_ALLOW_THREADS;
    if (!sample)
    {
        PyErr_SetString (PyExc_PyGameError, Mix_GetError ());
        return NULL;
    }
    music->music = sample;
    return (PyObject*) music;
}

void
music_export_capi (void **capi)
{
    capi[PYGAME_SDLMIXERMUSIC_FIRSTSLOT] = &PyMusic_Type;
    capi[PYGAME_SDLMIXERMUSIC_FIRSTSLOT+1] = (void *)PyMusic_New;
}
