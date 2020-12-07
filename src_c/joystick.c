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

static pgJoystickObject *joylist_head = NULL;
static PyObject *joy_instance_map = NULL;
static PyTypeObject pgJoystick_Type;
static PyObject *pgJoystick_New(int);
static int _joy_map_insert(pgJoystickObject *jstick);
#define pgJoystick_Check(x) ((x)->ob_type == &pgJoystick_Type)

static void
joy_autoquit(void)
{
    /* Walk joystick objects to deallocate the stick objects. */
    pgJoystickObject *cur = joylist_head;
    while (cur) {
        if (cur->joy) {
            SDL_JoystickClose(cur->joy);
            cur->joy = NULL;
        }
        cur = cur->next;
    }

    if (SDL_WasInit(SDL_INIT_JOYSTICK)) {
        SDL_JoystickEventState(SDL_ENABLE);
        SDL_QuitSubSystem(SDL_INIT_JOYSTICK);
    }

}

static PyObject *
joy_autoinit(PyObject *self)
{
    if (!SDL_WasInit(SDL_INIT_JOYSTICK)) {
        if (SDL_InitSubSystem(SDL_INIT_JOYSTICK)) {
            return PyInt_FromLong(0);
        }
        SDL_JoystickEventState(SDL_ENABLE);
        pg_RegisterQuit(joy_autoquit);
    }
    return PyInt_FromLong(1);
}

static PyObject *
quit(PyObject *self)
{
    joy_autoquit();
    Py_RETURN_NONE;
}

static PyObject *
init(PyObject *self)
{
    PyObject *result;
    int istrue;

    result = joy_autoinit(self);
    istrue = PyObject_IsTrue(result);
    Py_DECREF(result);
    if (!istrue) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    Py_RETURN_NONE;
}

static PyObject *
get_init(PyObject *self)
{
    return PyBool_FromLong(SDL_WasInit(SDL_INIT_JOYSTICK) != 0);
}

/*joystick object funcs*/
static void
joy_dealloc(PyObject *self)
{
    pgJoystickObject *jstick = (pgJoystickObject *) self;

    if (jstick->joy) {
        SDL_JoystickClose(jstick->joy);
    }

    if (jstick->prev) {
        jstick->prev->next = jstick->next;
    } else {
        joylist_head = jstick->next;
    }
    if (jstick->next) {
        jstick->next->prev = jstick->prev;
    }

    PyObject_DEL(self);
}

static PyObject *
Joystick(PyObject *self, PyObject *args)
{
    int id;
    if (!PyArg_ParseTuple(args, "i", &id)) {
        return NULL;
    }

    JOYSTICK_INIT_CHECK();

    return pgJoystick_New(id);
}

static PyObject *
get_count(PyObject *self, PyObject *args)
{
    JOYSTICK_INIT_CHECK();
    return PyInt_FromLong(SDL_NumJoysticks());
}


static PyObject *
joy_init(PyObject *self, PyObject *args)
{
    pgJoystickObject *jstick = (pgJoystickObject *) self;

    if (!jstick->joy) {
        jstick->joy = SDL_JoystickOpen(jstick->id);
        if (!jstick->joy) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }

    if (-1 == _joy_map_insert(jstick)) {
        return NULL;
    }

    Py_RETURN_NONE;
}

static int
_joy_map_insert(pgJoystickObject *jstick) {
#if IS_SDLv2
    SDL_JoystickID instance_id;
    PyObject *k, *v;

    if (!joy_instance_map) {
        return -1;
    }

    instance_id = SDL_JoystickInstanceID(jstick->joy);
    if (instance_id < 0) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    k = PyInt_FromLong(instance_id);
    v = PyInt_FromLong(jstick->id);
    if (k && v) {
        PyDict_SetItem(joy_instance_map, k, v);
    }
    Py_XDECREF(k);
    Py_XDECREF(v);
#endif

    return 0;
}

static PyObject *
joy_quit(PyObject *self, PyObject *args)
{
    pgJoystickObject *joy = (pgJoystickObject *) self;

    JOYSTICK_INIT_CHECK();
    if (joy->joy) {
        SDL_JoystickClose(joy->joy);
        joy->joy = NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
joy_get_init(PyObject *self, PyObject *args)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);
    return PyBool_FromLong(joy != NULL);
}

static PyObject *
joy_get_id(PyObject *self, PyObject *args)
{
    int joy_id = pgJoystick_AsID(self);
    return PyInt_FromLong(joy_id);
}

#if IS_SDLv2

static PyObject *
joy_get_instance_id(PyObject *self, PyObject *args)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }

    return PyInt_FromLong(SDL_JoystickInstanceID(joy));
}


static PyObject *
joy_get_guid(PyObject *self, PyObject *args)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);
    SDL_JoystickGUID guid;
    char strguid[33];

    JOYSTICK_INIT_CHECK();
    if (joy) {
        guid = SDL_JoystickGetGUID(joy);
    } else {
        guid = SDL_JoystickGetDeviceGUID(pgJoystick_AsID(self));
    }

    SDL_JoystickGetGUIDString(guid, strguid, 33);

    return Text_FromUTF8(strguid);
}


const char *_pg_powerlevel_string(SDL_JoystickPowerLevel level) {
    switch (level) {
        case SDL_JOYSTICK_POWER_EMPTY:
            return "empty";
        case SDL_JOYSTICK_POWER_LOW:
            return "low";
        case SDL_JOYSTICK_POWER_MEDIUM:
            return "medium";
        case SDL_JOYSTICK_POWER_FULL:
            return "full";
        case SDL_JOYSTICK_POWER_WIRED:
            return "wired";
        case SDL_JOYSTICK_POWER_MAX:
            return "max";
        default:
            return "unknown";
    }
}


static PyObject *
joy_get_power_level(PyObject *self, PyObject *args)
{
    SDL_JoystickPowerLevel level;
    const char *leveltext;
    SDL_Joystick *joy = pgJoystick_AsSDL(self);

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }

    level = SDL_JoystickCurrentPowerLevel(joy);
    leveltext = _pg_powerlevel_string(level);

    return Text_FromUTF8(leveltext);
}

#endif


static PyObject *
joy_get_name(PyObject *self, PyObject *args)
{
#if IS_SDLv1
    int joy_id = pgJoystick_AsID(self);
    JOYSTICK_INIT_CHECK();
    return Text_FromLocale(SDL_JoystickName(joy_id));
#else  /* IS_SDLv2 */
    SDL_Joystick *joy = pgJoystick_AsSDL(self);
    return Text_FromUTF8(SDL_JoystickName(joy));
#endif /* IS_SDLv2 */
}

static PyObject *
joy_get_numaxes(PyObject *self, PyObject *args)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);
    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }

    return PyInt_FromLong(SDL_JoystickNumAxes(joy));
}

static PyObject *
joy_get_axis(PyObject *self, PyObject *args)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);
    int axis, value;

    if (!PyArg_ParseTuple(args, "i", &axis)) {
        return NULL;
    }

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }
    if (axis < 0 || axis >= SDL_JoystickNumAxes(joy)) {
        return RAISE(pgExc_SDLError, "Invalid joystick axis");
    }

    value = SDL_JoystickGetAxis(joy, axis);
#ifdef DEBUG
    /*printf("SDL_JoystickGetAxis value:%d:\n", value);*/
#endif

    return PyFloat_FromDouble(value / 32768.0);
}

static PyObject *
joy_get_numbuttons(PyObject *self, PyObject *args)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }

    return PyInt_FromLong(SDL_JoystickNumButtons(joy));
}

static PyObject *
joy_get_button(PyObject *self, PyObject *args)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);
    int _index, value;

    if (!PyArg_ParseTuple(args, "i", &_index)) {
        return NULL;
    }

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }
    if (_index < 0 || _index >= SDL_JoystickNumButtons(joy)) {
        return RAISE(pgExc_SDLError, "Invalid joystick button");
    }

    value = SDL_JoystickGetButton(joy, _index);
#ifdef DEBUG
    /*printf("SDL_JoystickGetButton value:%d:\n", value);*/
#endif
    return PyInt_FromLong(value);
}

static PyObject *
joy_get_numballs(PyObject *self, PyObject *args)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }

    return PyInt_FromLong(SDL_JoystickNumBalls(joy));
}

static PyObject *
joy_get_ball(PyObject *self, PyObject *args)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);
    int _index, dx, dy;
    int value;

    if (!PyArg_ParseTuple(args, "i", &_index)) {
        return NULL;
    }

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }
    value = SDL_JoystickNumBalls(joy);
#ifdef DEBUG
    /*printf("SDL_JoystickNumBalls value:%d:\n", value);*/
#endif
    if (_index < 0 || _index >= value) {
        return RAISE(pgExc_SDLError, "Invalid joystick trackball");
    }

    SDL_JoystickGetBall(joy, _index, &dx, &dy);
    return Py_BuildValue("(ii)", dx, dy);
}

static PyObject *
joy_get_numhats(PyObject *self, PyObject *args)
{
    Uint32 value;
    SDL_Joystick *joy = pgJoystick_AsSDL(self);

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }

    value = SDL_JoystickNumHats(joy);
#ifdef DEBUG
    /*printf("SDL_JoystickNumHats value:%d:\n", value);*/
#endif
    return PyInt_FromLong(value);
}

static PyObject *
joy_get_hat(PyObject *self, PyObject *args)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);
    int _index, px, py;
    Uint32 value;

    if (!PyArg_ParseTuple(args, "i", &_index)) {
        return NULL;
    }

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }
    if (_index < 0 || _index >= SDL_JoystickNumHats(joy)) {
        return RAISE(pgExc_SDLError, "Invalid joystick hat");
    }

    px = py = 0;
    value = SDL_JoystickGetHat(joy, _index);
#ifdef DEBUG
    /*printf("SDL_JoystickGetHat value:%d:\n", value);*/
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

    return Py_BuildValue("(ii)", px, py);
}

static PyMethodDef joy_methods[] = {
    {"init", joy_init, METH_NOARGS, DOC_JOYSTICKINIT},
    {"quit", joy_quit, METH_NOARGS, DOC_JOYSTICKQUIT},
    {"get_init", joy_get_init, METH_NOARGS, DOC_JOYSTICKGETINIT},

    {"get_id", joy_get_id, METH_NOARGS, DOC_JOYSTICKGETID},
#if IS_SDLv2
    {"get_instance_id", joy_get_instance_id, METH_NOARGS, DOC_JOYSTICKGETINSTANCEID},
    {"get_guid", joy_get_guid, METH_NOARGS, DOC_JOYSTICKGETGUID},
    {"get_power_level", joy_get_power_level, METH_NOARGS, DOC_JOYSTICKGETPOWERLEVEL},
#endif
    {"get_name", joy_get_name, METH_NOARGS, DOC_JOYSTICKGETNAME},

    {"get_numaxes", joy_get_numaxes, METH_NOARGS,
     DOC_JOYSTICKGETNUMAXES},
    {"get_axis", joy_get_axis, METH_VARARGS, DOC_JOYSTICKGETAXIS},
    {"get_numbuttons", joy_get_numbuttons, METH_NOARGS,
     DOC_JOYSTICKGETNUMBUTTONS},
    {"get_button", joy_get_button, METH_VARARGS, DOC_JOYSTICKGETBUTTON},
    {"get_numballs", joy_get_numballs, METH_NOARGS,
     DOC_JOYSTICKGETNUMBALLS},
    {"get_ball", joy_get_ball, METH_VARARGS, DOC_JOYSTICKGETBALL},
    {"get_numhats", joy_get_numhats, METH_NOARGS,
     DOC_JOYSTICKGETNUMHATS},
    {"get_hat", joy_get_hat, METH_VARARGS, DOC_JOYSTICKGETHAT},

    {NULL, NULL, 0, NULL}};

static PyTypeObject pgJoystick_Type = {
    PyVarObject_HEAD_INIT(NULL,0)
    "Joystick",                    /* name */
    sizeof(pgJoystickObject),      /* basic size */
    0,                             /* itemsize */
    joy_dealloc,                   /* dealloc */
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
    DOC_PYGAMEJOYSTICKJOYSTICK,    /* Documentation string */
    0,                             /* tp_traverse */
    0,                             /* tp_clear */
    0,                             /* tp_richcompare */
    0,                             /* tp_weaklistoffset */
    0,                             /* tp_iter */
    0,                             /* tp_iternext */
    joy_methods,                   /* tp_methods */
    0,                             /* tp_members */
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

static PyObject *
pgJoystick_New(int id)
{
    pgJoystickObject *jstick, *cur;
    SDL_Joystick *joy;

    JOYSTICK_INIT_CHECK();

    /* Open the SDL device */
    if (id >= SDL_NumJoysticks()) {
        return RAISE(pgExc_SDLError, "Invalid joystick device number");
    }
    joy = SDL_JoystickOpen(id);
    if (!joy) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    /* Search existing joystick objects to see if we already have this stick. */
    cur = joylist_head;
    while (cur) {
        if (cur->joy == joy) {
            Py_INCREF(cur);
            return (PyObject *) cur;
        }
        cur = cur->next;
    }

    /* Construct the Python object */
    jstick = PyObject_NEW(pgJoystickObject, &pgJoystick_Type);
    if (!jstick) {
        return NULL;
    }
    jstick->id = id;
    jstick->joy = joy;
    jstick->prev = NULL;
    jstick->next = joylist_head;
    if (joylist_head) {
        joylist_head->prev = jstick;
    }
    joylist_head = jstick;

    if (-1 == _joy_map_insert(jstick)) {
        Py_DECREF(jstick);
        return NULL;
    }

    return (PyObject *)jstick;
}

static PyMethodDef _joystick_methods[] = {
    {"__PYGAMEinit__", (PyCFunction)joy_autoinit, METH_NOARGS,
     "auto initialize function for joystick"},
    {"init", (PyCFunction)init, METH_NOARGS, DOC_PYGAMEJOYSTICKINIT},
    {"quit", (PyCFunction)quit, METH_NOARGS, DOC_PYGAMEJOYSTICKQUIT},
    {"get_init", (PyCFunction)get_init, METH_NOARGS,
     DOC_PYGAMEJOYSTICKGETINIT},
    {"get_count", (PyCFunction)get_count, METH_NOARGS,
     DOC_PYGAMEJOYSTICKGETCOUNT},
    {"Joystick", Joystick, METH_VARARGS, DOC_PYGAMEJOYSTICKJOYSTICK},
    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(joystick)
{
    PyObject *module, *dict, *apiobj;
    int ecode;
    static void *c_api[PYGAMEAPI_JOYSTICK_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "joystick",
                                         DOC_PYGAMEJOYSTICK,
                                         -1,
                                         _joystick_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready(&pgJoystick_Type) == -1) {
        MODINIT_ERROR;
    }

    /* Grab the instance -> device id mapping */
    module = PyImport_ImportModule("pygame.event");
    if (!module) {
        MODINIT_ERROR;
    }
    joy_instance_map = PyObject_GetAttrString(module, "_joy_instance_map");
    Py_DECREF(module);

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3("joystick", _joystick_methods, DOC_PYGAMEJOYSTICK);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict(module);

    if (PyDict_SetItemString(dict, "JoystickType",
                             (PyObject *)&pgJoystick_Type) == -1) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    /* export the c api */
    c_api[0] = &pgJoystick_Type;
    c_api[1] = pgJoystick_New;
    apiobj = encapsulate_api(c_api, "joystick");
    if (apiobj == NULL) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF(apiobj);
    if (ecode == -1) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
