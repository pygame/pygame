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

#define PYGAMEAPI_CDROM_INTERNAL
#include "pygame.h"
#include "pgcompat.h"
#include "doc/cdrom_doc.h"

#define CDROM_MAXDRIVES 32
static SDL_CD* cdrom_drivedata[CDROM_MAXDRIVES] = {NULL};
static PyTypeObject PyCD_Type;
static PyObject* PyCD_New(int id);
#define PyCD_Check(x) ((x)->ob_type == &PyCD_Type)

static void
cdrom_autoquit (void)
{
    int loop;
    for (loop = 0; loop < CDROM_MAXDRIVES; ++loop) {
        if (cdrom_drivedata[loop]) {
            SDL_CDClose (cdrom_drivedata[loop]);
            cdrom_drivedata[loop] = NULL;
        }
    }

    if (SDL_WasInit (SDL_INIT_CDROM)) {
        SDL_QuitSubSystem (SDL_INIT_CDROM);
    }
}

static PyObject*
cdrom_autoinit (PyObject* self)
{
    if (!SDL_WasInit (SDL_INIT_CDROM)) {
        if (SDL_InitSubSystem (SDL_INIT_CDROM)) {
            return PyInt_FromLong (0);
        }
        PyGame_RegisterQuit (cdrom_autoquit);
    }
    return PyInt_FromLong (1);
}

static PyObject*
cdrom_quit (PyObject* self)
{
    cdrom_autoquit ();
    Py_RETURN_NONE;
}

static PyObject*
cdrom_init (PyObject* self)
{
    PyObject* result;
    int istrue;

    result = cdrom_autoinit (self);
    istrue = PyObject_IsTrue (result);
    Py_DECREF (result);
    if (!istrue) {
        return RAISE (PyExc_SDLError, SDL_GetError ());
    }
    Py_RETURN_NONE;
}

static PyObject*
get_init (PyObject* self)
{
    return PyInt_FromLong (SDL_WasInit (SDL_INIT_CDROM) != 0);
}

static void
cd_dealloc (PyObject* self)
{
    PyObject_DEL (self);
}

static PyObject*
CD (PyObject* self, PyObject* args)
{
    int id;
    if (!PyArg_ParseTuple (args, "i", &id)) {
        return NULL;
    }

    CDROM_INIT_CHECK ();
    return PyCD_New (id);
}

static PyObject*
get_count (PyObject* self)
{
    CDROM_INIT_CHECK ();
    return PyInt_FromLong (SDL_CDNumDrives ());
}

static PyObject*
cd_init (PyObject* self)
{
    int cd_id = PyCD_AsID (self);

    CDROM_INIT_CHECK ();
    if (!cdrom_drivedata[cd_id]) {
        cdrom_drivedata[cd_id] = SDL_CDOpen (cd_id);
        if (!cdrom_drivedata[cd_id]) {
            return RAISE (PyExc_SDLError, "Cannot initialize device");
        }
    }
    Py_RETURN_NONE;
}

static PyObject*
cd_quit (PyObject* self)
{
    int cd_id = PyCD_AsID (self);

    CDROM_INIT_CHECK ();

    if (cdrom_drivedata[cd_id]) {
        SDL_CDClose (cdrom_drivedata[cd_id]);
        cdrom_drivedata[cd_id] = NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
cd_get_init (PyObject* self)
{
    int cd_id = PyCD_AsID (self);
    return PyInt_FromLong (cdrom_drivedata[cd_id] != NULL);
}

static PyObject*
cd_play (PyObject* self, PyObject* args)
{
    int cd_id = PyCD_AsID (self);
    SDL_CD* cdrom = cdrom_drivedata[cd_id];
    int result, track, startframe, numframes, playforever=0;
    float start=0.0f, end=0.0f;
    PyObject *endobject=NULL;

    if (!PyArg_ParseTuple (args, "i|fO", &track, &start, &endobject)) {
        return NULL;
    }
    if (endobject == Py_None) {
        playforever = 1;
    }
    else if (!PyArg_ParseTuple (args, "i|ff", &track, &start, &end)) {
        return NULL;
    }

    CDROM_INIT_CHECK ();
    if (!cdrom) {
        return RAISE (PyExc_SDLError, "CD drive not initialized");
    }
    SDL_CDStatus (cdrom);
    if (track < 0 || track >= cdrom->numtracks) {
        return RAISE (PyExc_IndexError, "Invalid track number");
    }
    if (cdrom->track[track].type != SDL_AUDIO_TRACK) {
        return RAISE (PyExc_SDLError, "CD track type is not audio");
    }
	
    /*validate times*/
    if (playforever) {
        end = start;
    }
    else if (start == end && start != 0.0f) {
        Py_RETURN_NONE;
    }
	
    startframe = (int)(start * CD_FPS);
    numframes = 0;
    if (startframe < 0) {
        startframe = 0;
    }
    if (end) {
        numframes = (int) ((end-start) * CD_FPS);
    }
    else {
        numframes = cdrom->track[track].length - startframe;
    }
    if (numframes < 0 ||
        startframe > (int) (cdrom->track[track].length * CD_FPS)) {
        Py_RETURN_NONE;
    }

    result = SDL_CDPlayTracks (cdrom, track, startframe, 0, numframes);
    if (result == -1) {
        return RAISE (PyExc_SDLError, SDL_GetError ());
    }

    Py_RETURN_NONE;
}

static PyObject*
cd_pause (PyObject* self)
{
    int cd_id = PyCD_AsID (self);
    SDL_CD* cdrom = cdrom_drivedata[cd_id];
    int result;

    CDROM_INIT_CHECK ();
    if (!cdrom) {
        return RAISE (PyExc_SDLError, "CD drive not initialized");
    }

    result = SDL_CDPause (cdrom);
    if (result == -1) {
        return RAISE (PyExc_SDLError, SDL_GetError ());
    }
    Py_RETURN_NONE;
}

static PyObject*
cd_resume (PyObject* self)
{
    int cd_id = PyCD_AsID (self);
    SDL_CD* cdrom = cdrom_drivedata[cd_id];
    int result;

    CDROM_INIT_CHECK ();
    if (!cdrom) {
        return RAISE (PyExc_SDLError, "CD drive not initialized");
    }

    result = SDL_CDResume (cdrom);
    if (result == -1) {
        return RAISE (PyExc_SDLError, SDL_GetError ());
    }
    Py_RETURN_NONE;
}

static PyObject*
cd_stop (PyObject* self)
{
    int cd_id = PyCD_AsID (self);
    SDL_CD* cdrom = cdrom_drivedata[cd_id];
    int result;

    CDROM_INIT_CHECK ();
    if (!cdrom) {
        return RAISE (PyExc_SDLError, "CD drive not initialized");
    }

    result = SDL_CDStop (cdrom);
    if (result == -1) {
        return RAISE (PyExc_SDLError, SDL_GetError ());
    }
    Py_RETURN_NONE;
}

static PyObject*
cd_eject (PyObject* self)
{
    int cd_id = PyCD_AsID (self);
    SDL_CD* cdrom = cdrom_drivedata[cd_id];
    int result;

    CDROM_INIT_CHECK ();
    if (!cdrom) {
        return RAISE (PyExc_SDLError, "CD drive not initialized");
    }

    result = SDL_CDEject (cdrom);
    if (result == -1) {
        return RAISE (PyExc_SDLError, SDL_GetError ());
    }

    Py_RETURN_NONE;
}

static PyObject*
cd_get_empty (PyObject* self)
{
    int cd_id = PyCD_AsID (self);
    SDL_CD* cdrom = cdrom_drivedata[cd_id];
    int status;

    CDROM_INIT_CHECK ();
    if (!cdrom) {
        return RAISE (PyExc_SDLError, "CD drive not initialized");
    }

    status = SDL_CDStatus (cdrom);
    return PyInt_FromLong (status == CD_TRAYEMPTY);
}

static PyObject*
cd_get_busy (PyObject* self)
{
    int cd_id = PyCD_AsID (self);
    SDL_CD* cdrom = cdrom_drivedata[cd_id];
    int status;

    CDROM_INIT_CHECK ();
    if (!cdrom) {
        return RAISE (PyExc_SDLError, "CD drive not initialized");
    }

    status = SDL_CDStatus (cdrom);
    return PyInt_FromLong (status == CD_PLAYING);
}

static PyObject*
cd_get_paused (PyObject* self)
{
    int cd_id = PyCD_AsID (self);
    SDL_CD* cdrom = cdrom_drivedata[cd_id];
    int status;

    CDROM_INIT_CHECK ();
    if (!cdrom) {
        return RAISE (PyExc_SDLError, "CD drive not initialized");
    }

    status = SDL_CDStatus (cdrom);
    return PyInt_FromLong (status == CD_PAUSED);
}


static PyObject*
cd_get_current (PyObject* self)
{
    int cd_id = PyCD_AsID (self);
    SDL_CD* cdrom = cdrom_drivedata[cd_id];
    int track;
    float seconds;

    CDROM_INIT_CHECK ();
    if (!cdrom) {
        return RAISE (PyExc_SDLError, "CD drive not initialized");
    }

    SDL_CDStatus (cdrom);
    track = cdrom->cur_track;
    seconds = cdrom->cur_frame / (float) CD_FPS;

    return Py_BuildValue ("(if)", track, seconds);
}

static PyObject*
cd_get_numtracks (PyObject* self)
{
    int cd_id = PyCD_AsID (self);
    SDL_CD* cdrom = cdrom_drivedata[cd_id];

    CDROM_INIT_CHECK ();
    if (!cdrom) {
        return RAISE (PyExc_SDLError, "CD drive not initialized");
    }

    SDL_CDStatus (cdrom);
    return PyInt_FromLong (cdrom->numtracks);
}

static PyObject*
cd_get_id (PyObject* self)
{
    int cd_id = PyCD_AsID (self);
    return PyInt_FromLong (cd_id);
}

static PyObject*
cd_get_name (PyObject* self)
{
    int cd_id = PyCD_AsID (self);
    CDROM_INIT_CHECK ();
    return Text_FromUTF8 (SDL_CDName (cd_id));
}

static PyObject*
cd_get_track_audio (PyObject* self, PyObject* args)
{
    int cd_id = PyCD_AsID (self);
    SDL_CD* cdrom = cdrom_drivedata[cd_id];
    int track;

    if (!PyArg_ParseTuple (args, "i", &track)) {
        return NULL;
    }

    CDROM_INIT_CHECK ();
    if (!cdrom) {
        return RAISE (PyExc_SDLError, "CD drive not initialized");
    }
    SDL_CDStatus (cdrom);
    if (track < 0 || track >= cdrom->numtracks) {
        return RAISE (PyExc_IndexError, "Invalid track number");
    }

    return PyInt_FromLong (cdrom->track[track].type == SDL_AUDIO_TRACK);
}

static PyObject*
cd_get_track_length (PyObject* self, PyObject* args)
{
    int cd_id = PyCD_AsID (self);
    SDL_CD* cdrom = cdrom_drivedata[cd_id];
    int track;

    if (!PyArg_ParseTuple (args, "i", &track)) {
        return NULL;
    }

    CDROM_INIT_CHECK ();
    if (!cdrom) {
        return RAISE (PyExc_SDLError, "CD drive not initialized");
    }
    SDL_CDStatus (cdrom);
    if (track < 0 || track >= cdrom->numtracks) {
        return RAISE (PyExc_IndexError, "Invalid track number");
    }
    if (cdrom->track[track].type != SDL_AUDIO_TRACK) {
        return PyFloat_FromDouble (0.0);
    }

    return PyFloat_FromDouble (cdrom->track[track].length / (double) CD_FPS);
}

static PyObject*
cd_get_track_start (PyObject* self, PyObject* args)
{
    int cd_id = PyCD_AsID (self);
    SDL_CD* cdrom = cdrom_drivedata[cd_id];
    int track;

    if (!PyArg_ParseTuple (args, "i", &track)) {
        return NULL;
    }

    CDROM_INIT_CHECK ();
    if (!cdrom) {
        return RAISE (PyExc_SDLError, "CD drive not initialized");
    }
    SDL_CDStatus (cdrom);
    if (track < 0 || track >= cdrom->numtracks) {
        return RAISE (PyExc_IndexError, "Invalid track number");
    }

    return PyFloat_FromDouble (cdrom->track[track].offset / (double) CD_FPS);
}

static PyObject*
cd_get_all (PyObject* self, PyObject* args)
{
    int cd_id = PyCD_AsID (self);
    SDL_CD* cdrom = cdrom_drivedata[cd_id];
    int track;
    PyObject *tuple, *item;

    CDROM_INIT_CHECK ();
    if (!cdrom) {
        return RAISE (PyExc_SDLError, "CD drive not initialized");
    }

    SDL_CDStatus (cdrom);
    tuple = PyTuple_New (cdrom->numtracks);
    if (!tuple) {
        return NULL;
    }
    for (track=0; track < cdrom->numtracks; track++) {
        int audio = cdrom->track[track].type == SDL_AUDIO_TRACK;
        double start = cdrom->track[track].offset / (double) CD_FPS;
        double length = cdrom->track[track].length / (double) CD_FPS;
        double end = start + length;
        item = PyTuple_New (4);
        if (!item) {
            Py_DECREF (tuple);
            return NULL;
        }
        PyTuple_SET_ITEM (item, 0, PyInt_FromLong (audio));
        PyTuple_SET_ITEM (item, 1, PyFloat_FromDouble (start));
        PyTuple_SET_ITEM (item, 2, PyFloat_FromDouble (end));
        PyTuple_SET_ITEM (item, 3, PyFloat_FromDouble (length));
        PyTuple_SET_ITEM (tuple, track, item);
    }
    return tuple;
}

static PyMethodDef cd_methods[] =
{
    { "init", (PyCFunction) cd_init, METH_NOARGS, DOC_CDINIT },
    { "quit", (PyCFunction) cd_quit, METH_NOARGS, DOC_CDQUIT },
    { "get_init", (PyCFunction) cd_get_init, METH_NOARGS, DOC_CDGETINIT },

    { "play", cd_play, METH_VARARGS, DOC_CDINIT },
    { "pause", (PyCFunction) cd_pause, METH_NOARGS, DOC_CDPAUSE },
    { "resume", (PyCFunction) cd_resume, METH_NOARGS, DOC_CDRESUME },
    { "stop", (PyCFunction) cd_stop, METH_NOARGS, DOC_CDSTOP },
    { "eject", (PyCFunction) cd_eject, METH_NOARGS, DOC_CDEJECT },

    { "get_empty", (PyCFunction) cd_get_empty, METH_NOARGS, DOC_CDGETEMPTY },
    { "get_busy", (PyCFunction) cd_get_busy, METH_NOARGS, DOC_CDGETBUSY },
    { "get_paused", (PyCFunction) cd_get_paused, METH_NOARGS, DOC_CDGETPAUSED },
    { "get_current", (PyCFunction) cd_get_current, METH_NOARGS,
      DOC_CDGETCURRENT },
    { "get_numtracks", (PyCFunction) cd_get_numtracks, METH_NOARGS,
      DOC_CDGETNUMTRACKS },
    { "get_id", (PyCFunction) cd_get_id, METH_NOARGS, DOC_CDGETINIT },
    { "get_name", (PyCFunction) cd_get_name, METH_NOARGS, DOC_CDGETNAME },
    { "get_all", (PyCFunction) cd_get_all, METH_NOARGS, DOC_CDGETALL },

    { "get_track_audio", cd_get_track_audio, METH_VARARGS,
      DOC_CDGETTRACKAUDIO },
    { "get_track_length", cd_get_track_length, METH_VARARGS,
      DOC_CDGETTRACKLENGTH },
    { "get_track_start", cd_get_track_start, METH_VARARGS,
      DOC_CDGETTRACKSTART },

    { NULL, NULL, 0, NULL }
};

static PyTypeObject PyCD_Type =
{
    TYPE_HEAD (NULL, 0)
    "CD",                       /* name */
    sizeof(PyCDObject),         /* basic size */
    0,                          /* itemsize */
    cd_dealloc,                 /* dealloc */
    0,                          /* print */
    0,                          /* getattr */
    0,                          /* setattr */
    0,                          /* compare */
    0,                          /* repr */
    0,                          /* as_number */
    0,                          /* as_sequence */
    0,                          /* as_mapping */
    0,                          /* hash */
    0,                          /* call */
    0,                          /* str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    0,                          /* flags */
    DOC_PYGAMECDROMCD,          /* Documentation string */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,	                        /* tp_iter */
    0,                          /* tp_iternext */
    cd_methods,                 /* tp_methods */
    0,                          /* tp_members */
    0,                          /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    0,                          /* tp_init */
    0,				/* tp_alloc */
    0,			        /* tp_new */
};

static PyObject*
PyCD_New (int id)
{
    PyCDObject* cd;

    if (id < 0 || id >= CDROM_MAXDRIVES || id >= SDL_CDNumDrives ()) {
        return RAISE (PyExc_SDLError, "Invalid cdrom device number");
    }

    cd = PyObject_NEW (PyCDObject, &PyCD_Type);
    if(!cd) {
        return NULL;
    }

    cd->id = id;

    return (PyObject*)cd;
}

static PyMethodDef _cdrom_methods[] =
{
    { "__PYGAMEinit__", (PyCFunction) cdrom_autoinit, METH_NOARGS,
      "auto initialize function" },
    { "init", (PyCFunction) cdrom_init, METH_NOARGS, DOC_PYGAMECDROMINIT },
    { "quit", (PyCFunction) cdrom_quit, METH_NOARGS, DOC_PYGAMECDROMQUIT },
    { "get_init", (PyCFunction) get_init, METH_NOARGS, DOC_PYGAMECDROMGETINIT },
    { "get_count", (PyCFunction) get_count, METH_NOARGS,
      DOC_PYGAMECDROMGETCOUNT },
    { "CD", CD, METH_VARARGS, DOC_PYGAMECDROMCD },
    { NULL, NULL, 0, NULL }
};

MODINIT_DEFINE (cdrom)
{
    PyObject *module, *dict, *apiobj;
    int ecode;
    static void* c_api[PYGAMEAPI_CDROM_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "cdrom",
        DOC_PYGAMECDROM,
        -1,
        _cdrom_methods,
        NULL, NULL, NULL, NULL
    };
#endif


    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base ();
    if (PyErr_Occurred ()) {
	MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready (&PyCD_Type) == -1) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 (MODPREFIX "cdrom", 
                             _cdrom_methods, 
                             DOC_PYGAMECDROM);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict (module);

    if (PyDict_SetItemString (dict, "CDType", (PyObject *)&PyCD_Type) == -1) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    /* export the c api */
    c_api[0] = &PyCD_Type;
    c_api[1] = PyCD_New;
    apiobj = encapsulate_api (c_api, "cdrom");
    if (apiobj == NULL) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString (dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF (apiobj);


    if (ecode == -1) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN (module);
}
