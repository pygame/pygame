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
#define PYGAME_SDLMIXERCHUNK_INTERNAL

#include "mixermod.h"
#include "pgsdl.h"
#include "pgmixer.h"
#include "sdlmixerbase_doc.h"

static PyObject* _chunk_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _chunk_init (PyObject *chunk, PyObject *args, PyObject *kwds);
static void _chunk_dealloc (PyChunk *self);

static PyObject* _chunk_getbuf (PyObject *self, void *closure);
static PyObject* _chunk_getlen (PyObject *self, void *closure);
static PyObject* _chunk_getvolume (PyObject *self, void *closure);
static int _chunk_setvolume (PyObject *self, PyObject *value, void *closure);

/**
 */
static PyGetSetDef _chunk_getsets[] = {
    { "buf", _chunk_getbuf, NULL, DOC_BASE_CHUNK_BUF, NULL },
    { "len", _chunk_getlen, NULL, DOC_BASE_CHUNK_LEN, NULL },
    { "volume", _chunk_getvolume, _chunk_setvolume, DOC_BASE_CHUNK_VOLUME,
      NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PyChunk_Type =
{
    TYPE_HEAD(NULL,0)
    "sdlmixer.Chunk",           /* tp_name */
    sizeof (PyChunk),           /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _chunk_dealloc,   /* tp_dealloc */
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
    DOC_BASE_CHUNK,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    0,                          /* tp_methods */
    0,                          /* tp_members */
    _chunk_getsets,             /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _chunk_init,     /* tp_init */
    0,                          /* tp_alloc */
    _chunk_new,                 /* tp_new */
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
_chunk_dealloc (PyChunk *self)
{
    if (self->chunk)
    {
        if (self->playchannel != -1)
            Mix_HaltChannel (self->playchannel);
        Mix_FreeChunk (self->chunk);
    }
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static PyObject*
_chunk_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyChunk *chunk = (PyChunk*) type->tp_alloc (type, 0);
    if (!chunk)
        return NULL;
    chunk->playchannel = -1;
    chunk->chunk = NULL;
    return (PyObject*) chunk;
}

static int
_chunk_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *file;
    Mix_Chunk *chunk;
    SDL_RWops *rw;
    int autoclose;

    ASSERT_MIXER_OPEN (-1);
    
    if (!PyArg_ParseTuple (args, "O", &file))
        return -1;

    rw = PyRWops_NewRO_Threaded (file, &autoclose);
    if (!rw)
        return -1;

    Py_BEGIN_ALLOW_THREADS;
    chunk = Mix_LoadWAV_RW (rw, autoclose);
    Py_END_ALLOW_THREADS;
    
    if (!autoclose)
        PyRWops_Close (rw, autoclose);
    
    ((PyChunk*)self)->chunk = chunk;
    ((PyChunk*)self)->playchannel = -1;
    return 0;
}

/* Getters/Setters */
static PyObject*
_chunk_getbuf (PyObject *self, void *closure)
{
    PyChunk *chunk = (PyChunk *) self;

    ASSERT_MIXER_OPEN(NULL);

    return PyBufferProxy_New (self, (void*)chunk->chunk->abuf,
        (Py_ssize_t) chunk->chunk->alen, NULL);
}

static PyObject*
_chunk_getlen (PyObject *self, void *closure)
{
    ASSERT_MIXER_OPEN(NULL);
    return PyLong_FromUnsignedLong (((PyChunk*)self)->chunk->alen);
}

static PyObject*
_chunk_getvolume (PyObject *self, void *closure)
{
    ASSERT_MIXER_OPEN(NULL);
    return PyInt_FromLong (Mix_VolumeChunk (((PyChunk*)self)->chunk, -1));
}

static int
_chunk_setvolume (PyObject *self, PyObject *value, void *closure)
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
    Mix_VolumeChunk (((PyChunk*)self)->chunk, volume);
    return 0;
}

/* C API */
PyObject*
PyChunk_New (char *filename)
{
    PyChunk *chunk;
    Mix_Chunk *sample;

    ASSERT_MIXER_OPEN(NULL);
    
    if (!filename)
        return NULL;

    chunk = (PyChunk*) PyChunk_Type.tp_new (&PyChunk_Type, NULL, NULL);
    if (!chunk)
        return NULL;
    Py_BEGIN_ALLOW_THREADS;
    sample = Mix_LoadWAV (filename);
    Py_END_ALLOW_THREADS;
    if (!sample)
    {
        PyErr_SetString (PyExc_PyGameError, Mix_GetError ());
        return NULL;
    }
    chunk->chunk = sample;
    return (PyObject*) chunk;
}

PyObject*
PyChunk_NewFromMixerChunk (Mix_Chunk *sample)
{
    PyChunk *chunk;

    ASSERT_MIXER_OPEN(NULL);

    if (!sample)
        return NULL;

    chunk = (PyChunk*) PyChunk_Type.tp_new (&PyChunk_Type, NULL, NULL);
    if (!chunk)
        return NULL;
    chunk->chunk = sample;
    return (PyObject*) chunk;
}

void
chunk_export_capi (void **capi)
{
    capi[PYGAME_SDLMIXERCHUNK_FIRSTSLOT] = &PyChunk_Type;
    capi[PYGAME_SDLMIXERCHUNK_FIRSTSLOT+1] = (void *)PyChunk_New;
    capi[PYGAME_SDLMIXERCHUNK_FIRSTSLOT+2] = (void *)PyChunk_NewFromMixerChunk;
}
