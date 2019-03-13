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

#include "../pygame.h"
#include "../pgcompat.h"

#include "../doc/touch_doc.h"

#include <structmember.h>

#if PY3
#define INT_CHECK(o) PyLong_Check(o)
#else
#define INT_CHECK(o) (PyInt_Check(o) || PyLong_Check(o))
#endif

static PyObject * pgGesture_New(Sint64 touchid, Sint64 gestureid);

static PyObject *
pg_touch_num_devices(PyObject *self, PyObject *args)
{
    return PyLong_FromLong(SDL_GetNumTouchDevices());
}

static PyObject *
pg_touch_get_device(PyObject *self, PyObject *index)
{
    SDL_TouchID touchid;
    if (!INT_CHECK(index)) {
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
    if (!INT_CHECK(device_id)) {
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
    {"get_num_devices", pg_touch_num_devices, METH_NOARGS, DOC_PYGAMESDL2TOUCHGETNUMDEVICES},
    {"get_device", pg_touch_get_device, METH_O, DOC_PYGAMESDL2TOUCHGETDEVICE},

    {"get_num_fingers", pg_touch_num_fingers, METH_O, DOC_PYGAMESDL2TOUCHGETNUMFINGERS},
    {"get_finger", (PyCFunction)pg_touch_get_finger, METH_VARARGS | METH_KEYWORDS, DOC_PYGAMESDL2TOUCHGETFINGER},

    {NULL, NULL, 0, NULL}};

PyObject *_pg_gesture_from_touchid = 0;

static pgEventObject*
pg_gesture_event_filter(pgEventObject *event)
{
    if (event->type == SDL_DOLLARRECORD || event->type == SDL_DOLLARGESTURE) {
        PyObject *gestureobj;
        Sint64 gestureid;
        PyObject *idobj = PyDict_GetItemString(event->dict,
                                               "gesture_id");
        if (!idobj)
            return NULL;
        gestureid = PyLong_AsLongLong(idobj);

        if (event->type == SDL_DOLLARRECORD) {
            /* create a new Gesture instance */
            Sint64 touchid =
                PyLong_AsLongLong(PyDict_GetItemString(event->dict,
                                                       "touch_id"));
            gestureobj = pgGesture_New(touchid, gestureid);
            if (!gestureobj)
                return NULL;
            if (PyDict_SetItemString(event->dict, "gesture", gestureobj)) {
                Py_DECREF(gestureobj);
                return NULL;
            }

            if (!_pg_gesture_from_touchid) {
                _pg_gesture_from_touchid = PyDict_New();
                if (!_pg_gesture_from_touchid) {
                    Py_DECREF(gestureobj);
                    return NULL;
                }
            }

            PyDict_SetItem(_pg_gesture_from_touchid, idobj, gestureobj);
            Py_DECREF(gestureobj);
        }
        else {
            /* return existing Gesture instance */
            if (!_pg_gesture_from_touchid ||
                !(gestureobj = PyDict_GetItem(_pg_gesture_from_touchid, idobj))) {
                PyErr_SetString(pgExc_SDLError,
                                "received gesture event for unrecorded gesture");
                return NULL;
            }
            if (PyDict_SetItemString(event->dict, "gesture", gestureobj)) {
                return NULL;
            }
        }

        if (PyDict_DelItemString(event->dict, "touch_id") ||
            PyDict_DelItemString(event->dict, "gesture_id"))
            return NULL;
    }
    return event;
}

typedef struct {
    PyObject_HEAD
    SDL_GestureID gestureid;
    SDL_TouchID touchid;
} pgGestureObject;

static PyTypeObject pgGesture_Type;

static PyObject *
pgGesture_New(Sint64 touchid, Sint64 gestureid)
{
    pgGestureObject *gesture;
    gesture = PyObject_NEW(pgGestureObject, &pgGesture_Type);
    if (!gesture) {
        return NULL;
    }

    gesture->touchid = touchid;
    gesture->gestureid = gestureid;
    return (PyObject *)gesture;
}

static void
pg_gesture_dealloc(PyObject *self)
{
    PyObject_DEL(self);
}

static PyObject *
pg_gesture_record(PyObject *self, PyObject *idobj)
{
    SDL_TouchID touchid = PyLong_AsLongLong(idobj);
    if (touchid == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return RAISE(PyExc_TypeError, "id must be an integer touch device id"
                                      " or -1 for any touch device");
    }
    if (SDL_RecordGesture(touchid) == 0) {
        return RAISE(PyExc_ValueError, "cannot find the given touch device id");
    }

    Py_RETURN_NONE;
}

static PyObject *
pg_gesture_load(PyObject *self, PyObject *fileobj)
{
    /* TODO */

    Py_RETURN_NONE;
}

static PyObject *
pg_gesture_save(PyObject *self, PyObject *fileobj)
{
    /* TODO */

    Py_RETURN_NONE;
}

static PyObject *
pg_gesture_saveall(PyObject *self, PyObject *fileobj)
{
    /* TODO */

    Py_RETURN_NONE;
}

static PyMemberDef pg_gesture_members[] = {
    { "gesture_id", T_LONGLONG, offsetof(pgGestureObject, gestureid), READONLY, 0 /* TODO DOC */ },
    { "touch_id", T_LONGLONG, offsetof(pgGestureObject, touchid), READONLY, 0 /* TODO DOC */ },
    NULL
};

static PyMethodDef pg_gesture_methods[] = {
    { "record", pg_gesture_record, METH_O | METH_STATIC, 0 /* TODO DOC */ },
    { "load", pg_gesture_load, METH_O | METH_STATIC, 0 /* TODO DOC */ },
    { "save_all", pg_gesture_saveall, METH_O | METH_STATIC, 0 /* TODO DOC */ },
    { "save", pg_gesture_save, METH_O, 0 /* TODO DOC */ },
    NULL
};

static PyTypeObject pgGesture_Type = {
    TYPE_HEAD(NULL, 0) "Gesture",  /* name */
    sizeof(pgGestureObject),       /* basic size */
    0,                             /* itemsize */
    pg_gesture_dealloc,            /* dealloc */
    0,                             /* print */
    0,                             /* getattr */
    0,                             /* setattr */
    0,                             /* compare */
    0,                             /* repr */
    0,                             /* as_number */
    0,                             /* as_sequence */
    0,                             /* as_mapping */
    0,                             /* hash */
    0,                             /* call */
    0,                             /* str */
    0,                             /* tp_getattro */
    0,                             /* tp_setattro */
    0,                             /* tp_as_buffer */
    0,                             /* flags */
    0 /* TODO */,                  /* tp_doc */
    0,                             /* tp_traverse */
    0,                             /* tp_clear */
    0,                             /* tp_richcompare */
    0,                             /* tp_weaklistoffset */
    0,                             /* tp_iter */
    0,                             /* tp_iternext */
    pg_gesture_methods,            /* tp_methods */
    pg_gesture_members,            /* tp_members */
    0,                             /* tp_getset */
    0,                             /* tp_base */
    0,                             /* tp_dict */
    0,                             /* tp_descr_get */
    0,                             /* tp_descr_set */
    0,                             /* tp_dictoffset */
    0,                             /* tp_init */
    0,                             /* tp_alloc */
    0,                             /* tp_new */
};

MODINIT_DEFINE(touch)
{
    PyObject *module;
#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "touch",
                                         DOC_PYGAMESDL2TOUCH,
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

    import_pygame_event();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    if (PyType_Ready(&pgGesture_Type) < 0 ) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "touch", _touch_methods,
                            DOC_PYGAMESDL2TOUCH);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }

    if (PyModule_AddObject(module, pgGesture_Type.tp_name,
                           (PyObject *)&pgGesture_Type) == -1) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    if (pgEvent_RegisterFilter(pg_gesture_event_filter) == -1) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    MODINIT_RETURN(module);
}
