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
#define PYGAME_SDLGFXFPS_INTERNAL

#include "gfxmod.h"
#include "pggfx.h"
#include "sdlgfxbase_doc.h"

static PyObject* _fps_new (PyTypeObject *type, PyObject *args, PyObject *kwds);
static void _fps_dealloc (PyFPSmanager *self);
static int _fps_init (PyObject *self, PyObject *args, PyObject *kwds);

static int _fps_setframerate (PyObject *self, PyObject *value, void *closure);
static PyObject* _fps_getframerate (PyObject *self, void *closure);

static PyObject* _fps_delay (PyObject *self);

static PyMethodDef _fps_methods[] = {
    { "delay", (PyCFunction) _fps_delay, METH_NOARGS,
      DOC_BASE_FPSMANAGER_DELAY },
    { NULL, NULL, 0, NULL },
};

static PyGetSetDef _fps_getsets[] = {
    { "framerate", _fps_getframerate, _fps_setframerate,
      DOC_BASE_FPSMANAGER_FRAMERATE, NULL },
    { NULL, NULL, NULL, NULL, NULL },
};

/**
 */
PyTypeObject PyFPSmanager_Type =
{
    TYPE_HEAD(NULL,0)
    "sdlgfx.FPSmanager",        /* tp_name */
    sizeof (PyFPSmanager),      /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _fps_dealloc,  /* tp_dealloc */
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
    DOC_BASE_FPSMANAGER,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _fps_methods,               /* tp_methods */
    0,                          /* tp_members */
    _fps_getsets,               /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _fps_init,       /* tp_init */
    0,                          /* tp_alloc */
    _fps_new,                   /* tp_new */
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
_fps_dealloc (PyFPSmanager *self)
{
    if (self->fps)
        PyMem_Free (self->fps);
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static PyObject*
_fps_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyFPSmanager *fps = (PyFPSmanager*) type->tp_alloc (type, 0);
    if (!fps)
        return NULL;
    fps->fps = NULL;
    return (PyObject *) fps;
}

static int
_fps_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    int rate = FPS_DEFAULT;
    FPSmanager *manager;

    if (!PyArg_ParseTuple (args, "|i", &rate))
        return -1;

    if (rate > FPS_UPPER_LIMIT || rate < FPS_LOWER_LIMIT)
    {
        PyErr_SetString (PyExc_ValueError,
            "framerate must be within the limits");
        return -1;
    }

    manager = PyMem_New (FPSmanager, 1);
    if (!manager)
        return -1;
    SDL_initFramerate (manager);
    if (SDL_setFramerate (manager, rate) == -1)
    {
        PyMem_Free (manager);
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return -1;
    }

    ((PyFPSmanager*)self)->fps = manager;
    return 0;
}

/* Getters/Setters */
static int
_fps_setframerate (PyObject *self, PyObject *value, void *closure)
{
    int rate;
    
    if (!IntFromObj (value, &rate))
        return -1;

    if (rate > FPS_UPPER_LIMIT || rate < FPS_LOWER_LIMIT)
    {
        PyErr_SetString (PyExc_ValueError,
            "framerate must be within the limits");
        return -1;
    }
    if (SDL_setFramerate (((PyFPSmanager*)self)->fps, rate) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return -1;
    }
    return 0;
}

static PyObject*
_fps_getframerate (PyObject *self, void *closure)
{
    return PyInt_FromLong (SDL_getFramerate(((PyFPSmanager*)self)->fps));
}

/* Methods */
static PyObject*
_fps_delay (PyObject *self)
{
    Py_BEGIN_ALLOW_THREADS;
    SDL_framerateDelay (((PyFPSmanager*)self)->fps);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

/* C API */
PyObject*
PyFPSmanager_New (void)
{
    PyFPSmanager *manager;
    FPSmanager *fps;

    manager = (PyFPSmanager*) PyFPSmanager_Type.tp_new (&PyFPSmanager_Type,
        NULL, NULL);
    if (!manager)
        return NULL;

    fps = PyMem_New (FPSmanager, 1);
    if (!fps)
    {
        Py_DECREF (manager);
        return NULL;
    }
    SDL_initFramerate (fps);
    manager->fps = fps;
    return (PyObject*) manager;
}

void
fps_export_capi (void **capi)
{
    capi[PYGAME_SDLGFXFPS_FIRSTSLOT] = &PyFPSmanager_Type;
    capi[PYGAME_SDLGFXFPS_FIRSTSLOT+1] = &PyFPSmanager_New;
}
