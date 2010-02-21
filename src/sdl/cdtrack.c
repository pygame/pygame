/*
  pygame - Python Game Library
  Copyright (C) 2008 Marcus von Appen

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
#define PYGAME_SDLTRACK_INTERNAL

#include "cdrommod.h"
#include "pgsdl.h"
#include "sdlcdrom_doc.h"

static PyObject* _cdtrack_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _cdtrack_init (PyObject *track, PyObject *args, PyObject *kwds);
static void _cdtrack_dealloc (PyCDTrack *self);

static PyObject* _cdtrack_getid (PyObject *self, void *closure);
static PyObject* _cdtrack_gettype (PyObject *self, void *closure);
static PyObject* _cdtrack_getlength (PyObject *self, void *closure);
static PyObject* _cdtrack_getoffset (PyObject *self, void *closure);
static PyObject* _cdtrack_gettime (PyObject *self, void *closure);
static PyObject* _cdtrack_getminutes (PyObject *self, void *closure);
static PyObject* _cdtrack_getseconds (PyObject *self, void *closure);

/**
 */
static PyGetSetDef _cdtrack_getsets[] = {
    { "id", _cdtrack_getid, NULL, DOC_CDROM_CDTRACK_ID, NULL },
    { "type", _cdtrack_gettype, NULL, DOC_CDROM_CDTRACK_TYPE, NULL },
    { "length", _cdtrack_getlength, NULL, DOC_CDROM_CDTRACK_LENGTH, NULL },
    { "offset", _cdtrack_getoffset, NULL, DOC_CDROM_CDTRACK_OFFSET, NULL },
    { "time", _cdtrack_gettime, NULL, DOC_CDROM_CDTRACK_TIME, NULL },
    { "minutes", _cdtrack_getminutes, NULL, DOC_CDROM_CDTRACK_MINUTES, NULL },
    { "seconds", _cdtrack_getseconds, NULL, DOC_CDROM_CDTRACK_SECONDS, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PyCDTrack_Type =
{
    TYPE_HEAD(NULL,0)
    "cdrom.CDTrack",              /* tp_name */
    sizeof (PyCDTrack),         /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _cdtrack_dealloc,   /* tp_dealloc */
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
    DOC_CDROM_CDTRACK,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    0,                          /* tp_methods */
    0,                          /* tp_members */
    _cdtrack_getsets,           /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _cdtrack_init,   /* tp_init */
    0,                          /* tp_alloc */
    _cdtrack_new,               /* tp_new */
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
_cdtrack_dealloc (PyCDTrack *self)
{
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static PyObject*
_cdtrack_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyCDTrack *track = (PyCDTrack*) type->tp_alloc (type, 0);
    track->track.id = 0;
    track->track.type = 0;
    track->track.length = 0;
    track->track.offset = 0;
    return (PyObject*) track;
}

static int
_cdtrack_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    return 0;
}

/* Getters/Setters */
static PyObject*
_cdtrack_getid (PyObject *self, void *closure)
{
    return PyInt_FromLong (((PyCDTrack*)self)->track.id);
}

static PyObject*
_cdtrack_gettype (PyObject *self, void *closure)
{
    return PyInt_FromLong (((PyCDTrack*)self)->track.type);
}

static PyObject*
_cdtrack_getlength (PyObject *self, void *closure)
{
    return PyLong_FromUnsignedLong (((PyCDTrack*)self)->track.length);
}

static PyObject*
_cdtrack_getoffset (PyObject *self, void *closure)
{
    return PyLong_FromUnsignedLong (((PyCDTrack*)self)->track.offset);
}

static PyObject*
_cdtrack_gettime (PyObject *self, void *closure)
{
    int mins, secs;
    SDL_CDtrack track = ((PyCDTrack*)self)->track;
    
    mins = (track.length / CD_FPS) / 60;
    secs = (track.length / CD_FPS) % 60;
    
    return Py_BuildValue ("(ii)", mins, secs);
}

static PyObject*
_cdtrack_getminutes (PyObject *self, void *closure)
{
    double mins;
    SDL_CDtrack track = ((PyCDTrack*)self)->track;
    
    mins = ((track.length * 1.0f) / CD_FPS) / 60.0f;
    return PyFloat_FromDouble (mins);
}

static PyObject*
_cdtrack_getseconds (PyObject *self, void *closure)
{
    int secs;
    SDL_CDtrack track = ((PyCDTrack*)self)->track;
    
    secs = track.length / CD_FPS;
    return PyInt_FromLong (secs);
}

/* C API */
PyObject*
PyCDTrack_New (SDL_CDtrack track)
{
    PyCDTrack *t = (PyCDTrack*) PyCDTrack_Type.tp_new (&PyCDTrack_Type,
        NULL, NULL);
    if (!t)
        return NULL;
    
    t->track = track;
    return (PyObject*) t;
}

void
cdtrack_export_capi (void **capi)
{
    capi[PYGAME_SDLCDTRACK_FIRSTSLOT] = &PyCDTrack_Type;
    capi[PYGAME_SDLCDTRACK_FIRSTSLOT+1] = (void *)PyCDTrack_New;
}

