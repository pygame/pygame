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
#define PYGAME_SDLCDROM_INTERNAL

#include "cdrommod.h"
#include "pgsdl.h"
#include "sdlcdrom_doc.h"

static PyObject* _cd_new (PyTypeObject *type, PyObject *args, PyObject *kwds);
static int _cd_init (PyObject *cd, PyObject *args, PyObject *kwds);
static void _cd_dealloc (PyCD *self);

static PyObject* _cd_getname (PyObject *self, void *closure);
static PyObject* _cd_getindex (PyObject *self, void *closure);
static PyObject* _cd_getstatus (PyObject *self, void *closure);
static PyObject* _cd_getnumtracks (PyObject *self, void *closure);
static PyObject* _cd_getcurtrack (PyObject *self, void *closure);
static PyObject* _cd_getcurframe (PyObject *self, void *closure);
static PyObject* _cd_gettracks (PyObject *self, void *closure);

static PyObject* _cd_open (PyObject *self);
static PyObject* _cd_close (PyObject *self);
static PyObject* _cd_play (PyObject *self, PyObject *args);
static PyObject* _cd_playtracks (PyObject *self, PyObject *args,
    PyObject *kwds);
static PyObject* _cd_pause (PyObject *self);
static PyObject* _cd_resume (PyObject *self);
static PyObject* _cd_stop (PyObject *self);
static PyObject* _cd_eject (PyObject *self);

/**
 */
static PyMethodDef _cd_methods[] = {
    { "open", (PyCFunction) _cd_open, METH_NOARGS, DOC_CDROM_CD_OPEN },
    { "close", (PyCFunction) _cd_close, METH_NOARGS, DOC_CDROM_CD_CLOSE },
    { "play", _cd_play, METH_VARARGS, DOC_CDROM_CD_PLAY },
    { "play_tracks", (PyCFunction) _cd_playtracks, METH_VARARGS | METH_KEYWORDS,
      DOC_CDROM_CD_PLAY_TRACKS },
    { "pause", (PyCFunction) _cd_pause, METH_NOARGS, DOC_CDROM_CD_PAUSE },
    { "resume", (PyCFunction) _cd_resume, METH_NOARGS, DOC_CDROM_CD_RESUME },
    { "stop", (PyCFunction) _cd_stop, METH_NOARGS, DOC_CDROM_CD_STOP },
    { "eject", (PyCFunction) _cd_eject, METH_NOARGS, DOC_CDROM_CD_EJECT },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _cd_getsets[] = {
    { "name", _cd_getname, NULL, DOC_CDROM_CD_NAME, NULL },
    { "index", _cd_getindex, NULL, DOC_CDROM_CD_INDEX, NULL },
    { "status", _cd_getstatus, NULL, DOC_CDROM_CD_STATUS, NULL },
    { "num_tracks", _cd_getnumtracks, NULL, DOC_CDROM_CD_NUM_TRACKS, NULL },
    { "cur_track", _cd_getcurtrack, NULL, DOC_CDROM_CD_CUR_TRACK, NULL },
    { "cur_frame", _cd_getcurframe, NULL, DOC_CDROM_CD_CUR_FRAME, NULL },
    { "tracks", _cd_gettracks, NULL, DOC_CDROM_CD_TRACKS, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PyCD_Type =
{
    TYPE_HEAD(NULL,0)
    "cdrom.CD",                 /* tp_name */
    sizeof (PyCD),              /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _cd_dealloc,   /* tp_dealloc */
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
    DOC_CDROM_CD,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _cd_methods,                /* tp_methods */
    0,                          /* tp_members */
    _cd_getsets,                /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _cd_init,        /* tp_init */
    0,                          /* tp_alloc */
    _cd_new,                    /* tp_new */
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
_cd_dealloc (PyCD *self)
{
    if (self->cd)
    {
        SDL_CDClose (self->cd);
        cdrommod_remove_drive (self->index);
    }
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static PyObject*
_cd_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyCD* cd = (PyCD*) type->tp_alloc (type, 0);
    if (!cd)
        return NULL;
    cd->cd = NULL;
    cd->index = -1;
    return (PyObject*) cd;
}

static int
_cd_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    int _index;
    SDL_CD *cd;
    
    ASSERT_CDROM_INIT(-1);
    
    if (!PyArg_ParseTuple (args, "i", &_index))
        return -1;
    if (_index < 0 || _index > SDL_CDNumDrives ())
    {
        PyErr_SetString (PyExc_ValueError, "invalid cdrom drive index");
        return -1;
    }
    cd = SDL_CDOpen (_index);
    if (!cd)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return -1;
    }

    ((PyCD*)self)->cd = cd;
    ((PyCD*)self)->index = _index;
    cdrommod_add_drive (_index, cd);

    return 0;
}

/* Getters/Setters */
static PyObject*
_cd_getname (PyObject *self, void *closure)
{
    ASSERT_CDROM_INIT(NULL);
    return Text_FromUTF8 (SDL_CDName (((PyCD*)self)->index));
}

static PyObject*
_cd_getindex (PyObject *self, void *closure)
{
    ASSERT_CDROM_INIT(NULL);
    return PyInt_FromLong (((PyCD*)self)->index);
}

static PyObject*
_cd_getstatus (PyObject *self, void *closure)
{
    ASSERT_CDROM_OPEN(self, NULL);
    return PyInt_FromLong (SDL_CDStatus (((PyCD*)self)->cd));
}

static PyObject*
_cd_getnumtracks (PyObject *self, void *closure)
{
    ASSERT_CDROM_OPEN(self, NULL);
    return PyInt_FromLong (((PyCD*)self)->cd->numtracks);
}

static PyObject*
_cd_getcurtrack (PyObject *self, void *closure)
{
    ASSERT_CDROM_OPEN(self, NULL);
    return PyInt_FromLong (((PyCD*)self)->cd->cur_track);
}

static PyObject*
_cd_getcurframe (PyObject *self, void *closure)
{
    ASSERT_CDROM_OPEN(self, NULL);
    return PyInt_FromLong (((PyCD*)self)->cd->cur_frame);
}

static PyObject*
_cd_gettracks (PyObject *self, void *closure)
{
    PyObject *list, *track;
    int i;
    SDL_CD *cd = ((PyCD*)self)->cd;
    
    ASSERT_CDROM_OPEN(self, NULL);
    
    list = PyList_New (0);
    if (!list)
        return NULL;
    
    for (i = 0; i < cd->numtracks; i++)
    {
        track = PyCDTrack_New (cd->track[i]);
        if (!track)
        {
            Py_DECREF (list);
            return NULL;
        }
        
        if (PyList_Append (list, track) == -1)
        {
            Py_DECREF (list);
            Py_DECREF (track);
            return NULL;
        }
        Py_DECREF (track);
    }
    
    return list;
}

/* Methods */
static PyObject*
_cd_open (PyObject *self)
{
    SDL_CD *cd;
    PyCD *cdrom = (PyCD*)self;
    
    ASSERT_CDROM_INIT(NULL);
    
    cd = cdrommod_get_drive (cdrom->index);
    if (cd)
    {
        cdrom->cd = cd;
        Py_RETURN_NONE; /* Already open */
    }

    cd = SDL_CDOpen (cdrom->index);
    if (!cd)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    cdrom->cd = cd;
    cdrommod_add_drive (cdrom->index, cd);

    Py_RETURN_NONE;
}

static PyObject*
_cd_close (PyObject *self)
{
    PyCD *cdrom = (PyCD*)self;
    SDL_CD* cd;

    ASSERT_CDROM_INIT(NULL);
    
    cd = cdrommod_get_drive (cdrom->index);
    if (!cd)
        Py_RETURN_NONE; /* Already closed */
    
    SDL_CDClose (cdrom->cd);
    cdrom->cd = NULL;
    cdrommod_remove_drive (cdrom->index);

    Py_RETURN_NONE;
}

static PyObject*
_cd_play (PyObject *self, PyObject *args)
{
    int start, length;
    PyObject *asfps = Py_False;
    int istrue;
    
    SDL_CD *cd = ((PyCD*)self)->cd;
    ASSERT_CDROM_OPEN(self, NULL);

    if (!PyArg_ParseTuple (args, "ii|O:play", &start, &length, &asfps))
        return NULL;

    if (start < 0 || length < 0)
    {
        PyErr_SetString (PyExc_ValueError,
            "start and length must not be negative");
        return NULL;
    }

    istrue = PyObject_IsTrue (asfps);
    if (istrue == -1)
        return NULL;

    if (istrue == 0)
    {
        /* Start and length are in seconds */
        start *= CD_FPS;
        length *= CD_FPS;
    }

    SDL_CDStatus (cd);
    
    if (SDL_CDPlay (cd, start, length) == -1)
    {
        PyErr_SetString (PyExc_ValueError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_cd_playtracks (PyObject *self, PyObject *args, PyObject *kwds)
{
    int start_track = 0, start = 0, ntracks = 0, length = 0;
    PyObject *asfps = Py_False;
    int istrue;
    SDL_CD *cd = ((PyCD*)self)->cd;
    
    static char *kwlist[] = { "starttrack", "ntracks", "start", "length",
                              "asfps", NULL };
    
    ASSERT_CDROM_OPEN(self, NULL);

    if (!PyArg_ParseTupleAndKeywords (args, kwds, "|iiiiO:play_tracks", kwlist,
            &start_track, &ntracks, &start, &length, asfps))
        return NULL;

    if (start_track < 0 || start_track > cd->numtracks)
    {
        PyErr_SetString (PyExc_ValueError, "invalid start track");
        return NULL;
    }
    
    if (ntracks < 0 || ntracks > cd->numtracks - start_track)
    {
        PyErr_SetString (PyExc_ValueError, "invalid track amount");
        return NULL;
    }
    
    if (start < 0 || length < 0)
    {
        PyErr_SetString (PyExc_ValueError,
            "start and length must not be negative");
        return NULL;
    }

    istrue = PyObject_IsTrue (asfps);
    if (istrue == -1)
        return NULL;

    if (istrue == 0)
    {
        /* Start and length are in seconds */
        start *= CD_FPS;
        length *= CD_FPS;
    }
    
    SDL_CDStatus (cd);
    
    if (SDL_CDPlayTracks (cd, start_track, start, ntracks, length) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_cd_pause (PyObject *self)
{
    SDL_CD *cd = ((PyCD*)self)->cd;
    ASSERT_CDROM_OPEN(self, NULL);

    if (SDL_CDPause (cd) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_cd_resume (PyObject *self)
{
    SDL_CD *cd = ((PyCD*)self)->cd;
    ASSERT_CDROM_OPEN(self, NULL);

    if (SDL_CDResume (cd) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_cd_stop (PyObject *self)
{
    SDL_CD *cd = ((PyCD*)self)->cd;
    ASSERT_CDROM_OPEN(self, NULL);

    if (SDL_CDStop (cd) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_cd_eject (PyObject *self)
{
    SDL_CD *cd = ((PyCD*)self)->cd;
    ASSERT_CDROM_OPEN(self, NULL);

    if (SDL_CDEject (cd) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

/* C API */
PyObject*
PyCD_New (int _index)
{
    PyCD *cd;
    SDL_CD *cdrom;
    
    ASSERT_CDROM_INIT(NULL);
    
    if (_index < 0 || _index > SDL_CDNumDrives ())
    {
        PyErr_SetString (PyExc_ValueError, "invalid cdrom drive index");
        return NULL;
    }

    cd = (PyCD*) PyCD_Type.tp_new (&PyCD_Type, NULL, NULL);
    if (!cd)
        return NULL;

    cdrom = SDL_CDOpen (_index);
    if (!cdrom)
    {
        Py_DECREF (cd);
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    
    cd->cd = cdrom;
    cd->index = _index;
    cdrommod_add_drive (_index, cdrom);
    return (PyObject*) cd;
}

void
cdrom_export_capi (void **capi)
{
    capi[PYGAME_SDLCDROM_FIRSTSLOT] = &PyCD_Type;
    capi[PYGAME_SDLCDROM_FIRSTSLOT+1] = (void *)PyCD_New;
}
