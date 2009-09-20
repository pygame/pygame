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

#define PYGAMEAPI_JOYSTICK_INTERNAL
#include "pygame.h"
#include "pgcompat.h"
#include "doc/joystick_doc.h"

#define JOYSTICK_MAXSTICKS 32
static SDL_Joystick* joystick_stickdata[JOYSTICK_MAXSTICKS] = {NULL};
static PyTypeObject PyJoystick_Type;
static PyObject* PyJoystick_New (int);
#define PyJoystick_Check(x) ((x)->ob_type == &PyJoystick_Type)

static void
joy_autoquit (void)
{
    int loop;
    for (loop = 0; loop < JOYSTICK_MAXSTICKS; ++loop) {
        if (joystick_stickdata[loop]) {
            SDL_JoystickClose (joystick_stickdata[loop]);
            joystick_stickdata[loop] = NULL;
        }
    }

    if (SDL_WasInit (SDL_INIT_JOYSTICK)) {
        SDL_JoystickEventState (SDL_ENABLE);
        SDL_QuitSubSystem (SDL_INIT_JOYSTICK);
    }
}

static PyObject*
joy_autoinit (PyObject* self)
{
    if (!SDL_WasInit (SDL_INIT_JOYSTICK)) {
        if (SDL_InitSubSystem (SDL_INIT_JOYSTICK)) {
            return PyInt_FromLong (0);
        }
        SDL_JoystickEventState (SDL_ENABLE);
        PyGame_RegisterQuit (joy_autoquit);
    }
    return PyInt_FromLong (1);
}

static PyObject*
quit (PyObject* self)
{
    joy_autoquit ();
    Py_RETURN_NONE;
}

static PyObject*
init (PyObject* self)
{
    PyObject* result;
    int istrue;

    result = joy_autoinit (self);
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
    return PyInt_FromLong (SDL_WasInit (SDL_INIT_JOYSTICK) != 0);
}

/*joystick object funcs*/
static void
joy_dealloc (PyObject* self)
{
    PyObject_DEL (self);
}

static PyObject*
Joystick (PyObject* self, PyObject* args)
{
    int id;	
    if (!PyArg_ParseTuple (args, "i", &id)) {
        return NULL;
    }

    JOYSTICK_INIT_CHECK ();

    return PyJoystick_New (id);
}

static PyObject*
get_count (PyObject* self)
{
    JOYSTICK_INIT_CHECK ();
    return PyInt_FromLong (SDL_NumJoysticks ());
}

static PyObject*
joy_init (PyObject* self)
{
    int joy_id = PyJoystick_AsID (self);

    JOYSTICK_INIT_CHECK ();
    if (!joystick_stickdata[joy_id]) {
        joystick_stickdata[joy_id] = SDL_JoystickOpen (joy_id);
        if (!joystick_stickdata[joy_id]) {
            return RAISE (PyExc_SDLError, SDL_GetError ());
        }
    }
    Py_RETURN_NONE;
}

static PyObject*
joy_quit (PyObject* self)
{
    int joy_id = PyJoystick_AsID (self);

    JOYSTICK_INIT_CHECK ();

    if (joystick_stickdata[joy_id]) {
        SDL_JoystickClose (joystick_stickdata[joy_id]);
        joystick_stickdata[joy_id] = NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
joy_get_init (PyObject* self)
{
    int joy_id = PyJoystick_AsID (self);
    return PyInt_FromLong (joystick_stickdata[joy_id] != NULL);
}

static PyObject*
joy_get_id (PyObject* self)
{
    int joy_id = PyJoystick_AsID (self);
    return PyInt_FromLong (joy_id);
}

static PyObject*
joy_get_name (PyObject* self)
{
    int joy_id = PyJoystick_AsID (self);
    JOYSTICK_INIT_CHECK ();
    return Text_FromUTF8 (SDL_JoystickName (joy_id));
}

static PyObject*
joy_get_numaxes (PyObject* self)
{
    int joy_id = PyJoystick_AsID (self);
    SDL_Joystick* joy = joystick_stickdata[joy_id];

    JOYSTICK_INIT_CHECK ();
    if (!joy) {
        return RAISE (PyExc_SDLError, "Joystick not initialized");
    }

    return PyInt_FromLong (SDL_JoystickNumAxes (joy));
}

static PyObject*
joy_get_axis (PyObject* self, PyObject* args)
{
    int joy_id = PyJoystick_AsID (self);
    SDL_Joystick* joy = joystick_stickdata[joy_id];
    int axis, value;
	
    if (!PyArg_ParseTuple (args, "i", &axis)) {
        return NULL;
    }

    JOYSTICK_INIT_CHECK ();
    if (!joy) {
        return RAISE (PyExc_SDLError, "Joystick not initialized");
    }
    if (axis < 0 || axis >= SDL_JoystickNumAxes (joy)) {
        return RAISE (PyExc_SDLError, "Invalid joystick axis");
    }

    value = SDL_JoystickGetAxis (joy, axis);
#ifdef DEBUG
    printf("SDL_JoystickGetAxis value:%d:\n", value);
#endif

    return PyFloat_FromDouble (value / 32768.0);
}

static PyObject*
joy_get_numbuttons (PyObject* self)
{
    int joy_id = PyJoystick_AsID (self);
    SDL_Joystick* joy = joystick_stickdata[joy_id];

    JOYSTICK_INIT_CHECK ();
    if (!joy) {
        return RAISE (PyExc_SDLError, "Joystick not initialized");
    }

    return PyInt_FromLong (SDL_JoystickNumButtons (joy));
}

static PyObject*
joy_get_button (PyObject* self, PyObject* args)
{
    int joy_id = PyJoystick_AsID (self);
    SDL_Joystick* joy = joystick_stickdata[joy_id];
    int _index, value;
	
    if (!PyArg_ParseTuple (args, "i", &_index)) {
        return NULL;
    }

    JOYSTICK_INIT_CHECK ();
    if (!joy) {
        return RAISE (PyExc_SDLError, "Joystick not initialized");
    }
    if (_index < 0 || _index >= SDL_JoystickNumButtons (joy)) {
        return RAISE (PyExc_SDLError, "Invalid joystick button");
    }

    value = SDL_JoystickGetButton (joy, _index);
#ifdef DEBUG
    printf("SDL_JoystickGetButton value:%d:\n", value);
#endif
    return PyInt_FromLong (value);
}

static PyObject*
joy_get_numballs (PyObject* self)
{
    int joy_id = PyJoystick_AsID (self);
    SDL_Joystick* joy = joystick_stickdata[joy_id];

    JOYSTICK_INIT_CHECK ();
    if (!joy) {
        return RAISE (PyExc_SDLError, "Joystick not initialized");
    }

    return PyInt_FromLong (SDL_JoystickNumBalls (joy));
}

static PyObject*
joy_get_ball (PyObject* self, PyObject* args)
{
    int joy_id = PyJoystick_AsID (self);
    SDL_Joystick* joy = joystick_stickdata[joy_id];
    int _index, dx, dy;
    Uint32 value;
	
    if (!PyArg_ParseTuple (args, "i", &_index)) {
        return NULL;
    }

    JOYSTICK_INIT_CHECK ();
    if (!joy) {
        return RAISE (PyExc_SDLError, "Joystick not initialized");
    }
    value = SDL_JoystickNumBalls (joy);
#ifdef DEBUG
    printf("SDL_JoystickNumBalls value:%d:\n", value);
#endif
    if (_index < 0 || _index >= value) {
        return RAISE (PyExc_SDLError, "Invalid joystick trackball");
    }

    SDL_JoystickGetBall (joy, _index, &dx, &dy);
    return Py_BuildValue ("(ii)", dx, dy);
}

static PyObject*
joy_get_numhats (PyObject* self)
{
    int joy_id = PyJoystick_AsID (self);
    Uint32 value;
    SDL_Joystick* joy = joystick_stickdata[joy_id];

    JOYSTICK_INIT_CHECK ();
    if (!joy) {
        return RAISE (PyExc_SDLError, "Joystick not initialized");
    }

    value = SDL_JoystickNumHats (joy);
#ifdef DEBUG
    printf("SDL_JoystickNumHats value:%d:\n", value);
#endif
    return PyInt_FromLong (value);
}

static PyObject*
joy_get_hat (PyObject* self, PyObject* args)
{
    int joy_id = PyJoystick_AsID (self);
    SDL_Joystick* joy = joystick_stickdata[joy_id];
    int _index, px, py;
    Uint32 value;

    if (!PyArg_ParseTuple (args, "i", &_index)) {
        return NULL;
    }

    JOYSTICK_INIT_CHECK ();
    if (!joy) {
        return RAISE (PyExc_SDLError, "Joystick not initialized");
    }
    if (_index < 0 || _index >= SDL_JoystickNumHats (joy)) {
        return RAISE(PyExc_SDLError, "Invalid joystick hat");
    }

    px = py = 0;
    value = SDL_JoystickGetHat (joy, _index);
#ifdef DEBUG
    printf("SDL_JoystickGetHat value:%d:\n", value);
#endif
    if (value & SDL_HAT_UP) {
        py = 1;
    }
    else if (value & SDL_HAT_DOWN) {
        py = -1;
    }
    if (value & SDL_HAT_RIGHT) {
        px = 1;
    }
    else if (value & SDL_HAT_LEFT) {
        px = -1;
    }
	
    return Py_BuildValue ("(ii)", px, py);
}

static PyMethodDef joy_methods[] =
{
    { "init", (PyCFunction) joy_init, METH_NOARGS, DOC_JOYSTICKINIT },
    { "quit", (PyCFunction) joy_quit, METH_NOARGS, DOC_JOYSTICKQUIT },
    { "get_init", (PyCFunction) joy_get_init, METH_NOARGS,
      DOC_JOYSTICKGETINIT },

    { "get_id", (PyCFunction) joy_get_id, METH_NOARGS, DOC_JOYSTICKGETID },
    { "get_name", (PyCFunction) joy_get_name, METH_NOARGS,
      DOC_JOYSTICKGETNAME },

    { "get_numaxes", (PyCFunction) joy_get_numaxes, METH_NOARGS,
      DOC_JOYSTICKGETNUMAXES },
    { "get_axis", joy_get_axis, METH_VARARGS, DOC_JOYSTICKGETAXIS },
    { "get_numbuttons", (PyCFunction) joy_get_numbuttons, METH_NOARGS,
      DOC_JOYSTICKGETNUMBUTTONS },
    { "get_button", joy_get_button, METH_VARARGS, DOC_JOYSTICKGETBUTTON },
    { "get_numballs", (PyCFunction) joy_get_numballs, METH_NOARGS,
      DOC_JOYSTICKGETNUMBALLS },
    { "get_ball", joy_get_ball, METH_VARARGS, DOC_JOYSTICKGETBALL },
    { "get_numhats", (PyCFunction) joy_get_numhats, METH_NOARGS,
      DOC_JOYSTICKGETNUMHATS },
    { "get_hat", joy_get_hat, METH_VARARGS, DOC_JOYSTICKGETHAT },

    { NULL, NULL, 0, NULL }
};

static PyTypeObject PyJoystick_Type =
{
    TYPE_HEAD (NULL, 0)
    "Joystick",                 /* name */
    sizeof(PyJoystickObject),   /* basic size */
    0,                          /* itemsize */
    joy_dealloc,                /* dealloc */
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
    DOC_PYGAMEJOYSTICKJOYSTICK, /* Documentation string */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,	                        /* tp_iter */
    0,                          /* tp_iternext */
    joy_methods,                /* tp_methods */
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
PyJoystick_New (int id)
{
    PyJoystickObject* joy;

    if (id < 0 || id >= JOYSTICK_MAXSTICKS || id >= SDL_NumJoysticks ()) {
        return RAISE (PyExc_SDLError, "Invalid joystick device number");
    }
	
    joy = PyObject_NEW (PyJoystickObject, &PyJoystick_Type);
    if (!joy) {
        return NULL;
    }

    joy->id = id;

    return (PyObject*)joy;
}

static PyMethodDef _joystick_methods[] =
{
    { "__PYGAMEinit__", (PyCFunction) joy_autoinit, METH_NOARGS,
      "auto initialize function for joystick" },
    { "init", (PyCFunction) init, METH_NOARGS, DOC_PYGAMEJOYSTICKINIT },
    { "quit", (PyCFunction) quit, METH_NOARGS, DOC_PYGAMEJOYSTICKQUIT },
    { "get_init", (PyCFunction) get_init, METH_NOARGS,
      DOC_PYGAMEJOYSTICKGETINIT },
    { "get_count", (PyCFunction) get_count, METH_NOARGS,
      DOC_PYGAMEJOYSTICKGETCOUNT },
    { "Joystick", Joystick, METH_VARARGS, DOC_PYGAMEJOYSTICKJOYSTICK },
    { NULL, NULL, 0, NULL }
};

MODINIT_DEFINE (joystick)
{
    PyObject *module, *dict, *apiobj;
    int ecode;
    static void* c_api[PYGAMEAPI_JOYSTICK_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "joystick",
        DOC_PYGAMEJOYSTICK,
        -1,
        _joystick_methods,
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
    if (PyType_Ready (&PyJoystick_Type) == -1) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 ("joystick", _joystick_methods, DOC_PYGAMEJOYSTICK);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict (module);

    if (PyDict_SetItemString (dict, "JoystickType",
                              (PyObject *)&PyJoystick_Type) == -1) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    /* export the c api */
    c_api[0] = &PyJoystick_Type;
    c_api[1] = PyJoystick_New;
    apiobj = PyCObject_FromVoidPtr (c_api, NULL);
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
