/*
  pygame - Python Game Library
  Copyright (C) 2019 David Lönnhager

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

#include "pygame.h"
#include "pgcompat.h"


static PyObject *
pg_touch_num_devices(PyObject *self, PyObject *args)
{
    return PyLong_FromLong(SDL_GetNumTouchDevices());
}

static PyObject *
pg_touch_get_device(PyObject *self, PyObject *index)
{
    SDL_TouchID touchid;
    if (!PyLong_Check(index)) {
        return RAISE(PyExc_TypeError,
                     "index must be an integer "
                     "specifying a device to get the ID for");
    }

    touchid = SDL_GetTouchDevice(PyLong_AsLong(index));
    if (touchid == 0) {
        /* invalid index */
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    return PyLong_FromLongLong(touchid);
}

static PyObject *
pg_touch_num_fingers(PyObject *self, PyObject *device_id)
{
    int fingercount;
    if (!PyLong_Check(device_id)) {
        return RAISE(PyExc_TypeError,
                     "device_id must be an integer "
                     "specifying a touch device");
    }

    VIDEO_INIT_CHECK();

    fingercount =
        SDL_GetNumTouchFingers(PyLong_AsLongLong(device_id));
    if (fingercount == 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    return PyLong_FromLong(fingercount);
}

/* Helper for adding objects to dictionaries. Check for errors with
   PyErr_Occurred() */
static void
_pg_insobj(PyObject *dict, char *name, PyObject *v)
{
    if (v) {
        PyDict_SetItemString(dict, name, v);
        Py_DECREF(v);
    }
}

static PyObject *
pg_touch_get_finger(PyObject *self, PyObject *args, PyObject *kwargs)
{
    char* keywords[] = {"touchid", "index", NULL};
    SDL_TouchID touchid;
    int index;
    SDL_Finger *finger;
    PyObject *fingerobj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "Li", keywords,
                                     &touchid, &index))
    {
        return NULL;
    }

    VIDEO_INIT_CHECK();

    if (!(finger = SDL_GetTouchFinger(touchid, index))) {
        Py_RETURN_NONE;
    }

    fingerobj = PyDict_New();
    if (!fingerobj)
        return NULL;

    _pg_insobj(fingerobj, "id", PyLong_FromLongLong(finger->id));
    _pg_insobj(fingerobj, "x", PyFloat_FromDouble(finger->x));
    _pg_insobj(fingerobj, "y", PyFloat_FromDouble(finger->y));
    _pg_insobj(fingerobj, "pressure", PyFloat_FromDouble(finger->pressure));

    if (PyErr_Occurred()) {
        Py_DECREF(fingerobj);
        return NULL;
    }

    return fingerobj;
}

static PyMethodDef _touch_methods[] = {
    {"get_num_devices", pg_touch_num_devices, METH_NOARGS, NULL},
    {"get_device", pg_touch_get_device, METH_O, NULL},

    {"get_num_fingers", pg_touch_num_fingers, METH_O, NULL},
    {"get_finger", (PyCFunction)pg_touch_get_finger, METH_VARARGS | METH_KEYWORDS, NULL},

    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(touch)
{
    PyObject *module;
#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "touch",
                                         NULL, /* DOC_PYGAMETOUCH */
                                         -1,
                                         _touch_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "touch", _touch_methods,
                            NULL /*DOC_PYGAMETOUCH*/);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
