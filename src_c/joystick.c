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
static PyTypeObject pgJoystick_Type;
static PyObject *
pgJoystick_New(int);
static int
pgJoystick_GetDeviceIndexByInstanceID(int);
#define pgJoystick_Check(x) ((x)->ob_type == &pgJoystick_Type)

static PyObject *
init(PyObject *self, PyObject *_null)
{
    if (!SDL_WasInit(SDL_INIT_JOYSTICK)) {
        if (!PG_InitSubSystem(SDL_INIT_JOYSTICK)) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        PG_SetJoystickEventsEnabled(SDL_TRUE);
    }
    Py_RETURN_NONE;
}

static PyObject *
quit(PyObject *self, PyObject *_null)
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
        PG_SetJoystickEventsEnabled(SDL_TRUE);
        SDL_QuitSubSystem(SDL_INIT_JOYSTICK);
    }
    Py_RETURN_NONE;
}

static PyObject *
get_init(PyObject *self, PyObject *_null)
{
    return PyBool_FromLong(SDL_WasInit(SDL_INIT_JOYSTICK) != 0);
}

/*joystick object funcs*/
static void
joy_dealloc(PyObject *self)
{
    pgJoystickObject *jstick = (pgJoystickObject *)self;

    if (jstick->joy) {
        SDL_JoystickClose(jstick->joy);
    }

    if (jstick->prev) {
        jstick->prev->next = jstick->next;
    }
    else {
        joylist_head = jstick->next;
    }
    if (jstick->next) {
        jstick->next->prev = jstick->prev;
    }

    Py_TYPE(self)->tp_free(self);
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
get_count(PyObject *self, PyObject *_null)
{
    JOYSTICK_INIT_CHECK();
#if SDL_VERSION_ATLEAST(3, 0, 0)
    int ret;
    SDL_JoystickID *joysticks = SDL_GetJoysticks(&ret);
    if (!joysticks) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    SDL_free(joysticks);
    return PyLong_FromLong(ret);
#else
    return PyLong_FromLong(SDL_NumJoysticks());
#endif
}

static PyObject *
joy_init(PyObject *self, PyObject *_null)
{
    if (PyErr_WarnEx(
            PyExc_DeprecationWarning,
            "pygame.joystick.Joystick.init() is deprecated since 2.4.0. "
            "Bear in mind in future it will not be possible to "
            "reinitialise a closed Joystick object.",
            1) == -1) {
        return NULL;
    }

    pgJoystickObject *jstick = (pgJoystickObject *)self;

    if (!jstick->joy) {
        jstick->joy = SDL_JoystickOpen(jstick->id);
        if (!jstick->joy) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }

    Py_RETURN_NONE;
}

static PyObject *
joy_quit(PyObject *self, PyObject *_null)
{
    pgJoystickObject *joy = (pgJoystickObject *)self;

    JOYSTICK_INIT_CHECK();
    if (joy->joy) {
        SDL_JoystickClose(joy->joy);
        joy->joy = NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
joy_get_init(PyObject *self, PyObject *_null)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);
    return PyBool_FromLong(joy != NULL);
}

static PyObject *
joy_get_id(PyObject *self, PyObject *_null)
{
    if (PyErr_WarnEx(
            PyExc_DeprecationWarning,
            "pygame.joystick.Joystick.get_id() is deprecated since 2.4.0. "
            "The original device index is not useful in pygame 2. "
            "Use get_instance_id() instead.",
            1) == -1) {
        return NULL;
    }

    int joy_id = pgJoystick_AsID(self);
    return PyLong_FromLong(joy_id);
}

static PyObject *
joy_get_instance_id(PyObject *self, PyObject *_null)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }

    return PyLong_FromLong(SDL_JoystickInstanceID(joy));
}

static PyObject *
joy_get_guid(PyObject *self, PyObject *_null)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);
    SDL_JoystickGUID guid;
    char strguid[33];

    JOYSTICK_INIT_CHECK();
    if (joy) {
        guid = SDL_JoystickGetGUID(joy);
    }
    else {
#if SDL_VERSION_ATLEAST(3, 0, 0)
        return RAISE(pgExc_SDLError, "Invalid/closed joystick object");
#else
        guid = SDL_JoystickGetDeviceGUID(pgJoystick_AsID(self));
#endif
    }

#if SDL_VERSION_ATLEAST(3, 0, 0)
    SDL_GUIDToString(guid, strguid, 33);
#else
    SDL_JoystickGetGUIDString(guid, strguid, 33);
#endif

    return PyUnicode_FromString(strguid);
}

const char *
_pg_powerlevel_string(SDL_Joystick *joy)
{
#if SDL_VERSION_ATLEAST(3, 0, 0)
    int percent = -1;
    SDL_PowerState state = SDL_GetJoystickPowerInfo(joy, &percent);
    if (state == SDL_POWERSTATE_ON_BATTERY) {
        /* These percentages are based on SDL_JoystickCurrentPowerLevel defined
         * in sdl2-compat */
        if (percent > 70) {
            return "full";
        }
        else if (percent > 20) {
            return "medium";
        }
        else if (percent > 5) {
            return "low";
        }
        else {
            return "empty";
        }
    }
    else if (state == SDL_POWERSTATE_UNKNOWN ||
             state == SDL_POWERSTATE_ERROR) {
        return "unknown";
    }
    else {
        return "wired";
    }
#else
    switch (SDL_JoystickCurrentPowerLevel(joy)) {
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
#endif
}

static PyObject *
joy_get_power_level(PyObject *self, PyObject *_null)
{
    const char *leveltext;
    SDL_Joystick *joy = pgJoystick_AsSDL(self);

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }

    leveltext = _pg_powerlevel_string(joy);

    return PyUnicode_FromString(leveltext);
}

static PyObject *
joy_rumble(pgJoystickObject *self, PyObject *args, PyObject *kwargs)
{
    SDL_Joystick *joy = self->joy;
    double lowf, highf;
    uint32_t low, high, duration;

    char *keywords[] = {
        "low_frequency",
        "high_frequency",
        "duration",
        NULL,
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ddI", keywords, &lowf,
                                     &highf, &duration)) {
        return NULL;
    }

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }

    if (lowf < 0) {
        lowf = 0.0;
    }
    else if (lowf > 1.0) {
        lowf = 1.0;
    }

    if (highf < 0) {
        highf = 0.f;
    }
    else if (highf > 1.0) {
        highf = 1.0;
    }
    low = (Uint32)(lowf * 0xFFFF);
    high = (Uint32)(highf * 0xFFFF);

#if SDL_VERSION_ATLEAST(3, 0, 0)
    if (!SDL_JoystickRumble(joy, low, high, duration)) {
#else
    if (SDL_JoystickRumble(joy, low, high, duration) == -1) {
#endif
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

static PyObject *
joy_stop_rumble(pgJoystickObject *self, PyObject *_null)
{
    SDL_Joystick *joy = self->joy;
    SDL_JoystickRumble(joy, 0, 0, 1);
    Py_RETURN_NONE;
}

static PyObject *
joy_get_name(PyObject *self, PyObject *_null)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);
    return PyUnicode_FromString(SDL_JoystickName(joy));
}

static PyObject *
joy_get_numaxes(PyObject *self, PyObject *_null)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);
    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }

    return PyLong_FromLong(SDL_JoystickNumAxes(joy));
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
joy_get_numbuttons(PyObject *self, PyObject *_null)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }

    return PyLong_FromLong(SDL_JoystickNumButtons(joy));
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
    return PyLong_FromLong(value);
}

static PyObject *
joy_get_numballs(PyObject *self, PyObject *_null)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }

    return PyLong_FromLong(SDL_JoystickNumBalls(joy));
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
    return pg_tuple_couple_from_values_int(dx, dy);
}

static PyObject *
joy_get_numhats(PyObject *self, PyObject *_null)
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
    return PyLong_FromLong(value);
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

    return pg_tuple_couple_from_values_int(px, py);
}

static PyObject *
joy_set_led(PyObject *self, PyObject *arg)
{
    SDL_Joystick *joy = pgJoystick_AsSDL(self);

    JOYSTICK_INIT_CHECK();
    if (!joy) {
        return RAISE(pgExc_SDLError, "Joystick not initialized");
    }

    Uint8 colors[4] = {0, 0, 0, 0};

    if (!pg_RGBAFromObjEx(arg, colors, PG_COLOR_HANDLE_ALL)) {
        // Exception already set
        return NULL;
    }

#if !SDL_VERSION_ATLEAST(3, 0, 0)
    if (SDL_JoystickSetLED(joy, colors[0], colors[1], colors[2]) < 0) {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
#else
    // SDL3 renames the function and sets an error message on failure
    bool result = SDL_SetJoystickLED(joy, colors[0], colors[1], colors[2]);
    if (!result) {
        // Clear the SDL error message that SDL set, for example if it didn't
        // have an addressable LED
        (void)SDL_GetError();
    }
    return PyBool_FromLong(result);
#endif
}

static PyMethodDef joy_methods[] = {
    {"init", joy_init, METH_NOARGS, DOC_JOYSTICK_JOYSTICK_INIT},
    {"quit", joy_quit, METH_NOARGS, DOC_JOYSTICK_JOYSTICK_QUIT},
    {"get_init", joy_get_init, METH_NOARGS, DOC_JOYSTICK_JOYSTICK_GETINIT},
    {"get_id", joy_get_id, METH_NOARGS, DOC_JOYSTICK_JOYSTICK_GETID},
    {"get_instance_id", joy_get_instance_id, METH_NOARGS,
     DOC_JOYSTICK_JOYSTICK_GETINSTANCEID},
    {"get_guid", joy_get_guid, METH_NOARGS, DOC_JOYSTICK_JOYSTICK_GETGUID},
    {"get_power_level", joy_get_power_level, METH_NOARGS,
     DOC_JOYSTICK_JOYSTICK_GETPOWERLEVEL},
    {"rumble", (PyCFunction)joy_rumble, METH_VARARGS | METH_KEYWORDS,
     DOC_JOYSTICK_JOYSTICK_RUMBLE},
    {"stop_rumble", (PyCFunction)joy_stop_rumble, METH_NOARGS,
     DOC_JOYSTICK_JOYSTICK_STOPRUMBLE},
    {"get_name", joy_get_name, METH_NOARGS, DOC_JOYSTICK_JOYSTICK_GETNAME},

    {"get_numaxes", joy_get_numaxes, METH_NOARGS,
     DOC_JOYSTICK_JOYSTICK_GETNUMAXES},
    {"get_axis", joy_get_axis, METH_VARARGS, DOC_JOYSTICK_JOYSTICK_GETAXIS},
    {"get_numbuttons", joy_get_numbuttons, METH_NOARGS,
     DOC_JOYSTICK_JOYSTICK_GETNUMBUTTONS},
    {"get_button", joy_get_button, METH_VARARGS,
     DOC_JOYSTICK_JOYSTICK_GETBUTTON},
    {"get_numballs", joy_get_numballs, METH_NOARGS,
     DOC_JOYSTICK_JOYSTICK_GETNUMBALLS},
    {"get_ball", joy_get_ball, METH_VARARGS, DOC_JOYSTICK_JOYSTICK_GETBALL},
    {"get_numhats", joy_get_numhats, METH_NOARGS,
     DOC_JOYSTICK_JOYSTICK_GETNUMHATS},
    {"get_hat", joy_get_hat, METH_VARARGS, DOC_JOYSTICK_JOYSTICK_GETHAT},
    {"set_led", joy_set_led, METH_O, DOC_JOYSTICK_JOYSTICK_SETLED},

    {NULL, NULL, 0, NULL}};

static PyTypeObject pgJoystick_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.joystick.Joystick",
    .tp_basicsize = sizeof(pgJoystickObject),
    .tp_dealloc = joy_dealloc,
    .tp_doc = DOC_JOYSTICK_JOYSTICK,
    .tp_methods = joy_methods,
};

static int
pgJoystick_GetDeviceIndexByInstanceID(int instance_id)
{
    pgJoystickObject *cur;
    cur = joylist_head;
    while (cur) {
        if (SDL_JoystickInstanceID(cur->joy) == instance_id) {
            return cur->id;
        }
        cur = cur->next;
    }
    return -1;
}

static PyObject *
pgJoystick_New(int id)
{
    pgJoystickObject *jstick, *cur;
    SDL_Joystick *joy;

    JOYSTICK_INIT_CHECK();

    /* Open the SDL device */
#if !SDL_VERSION_ATLEAST(3, 0, 0)
    /* This check should be redundant because SDL_JoystickOpen already checks
     * and errors if id is out of bounds on SDL3 */
    if (id >= SDL_NumJoysticks()) {
        return RAISE(pgExc_SDLError, "Invalid joystick device number");
    }
#endif
    joy = SDL_JoystickOpen(id);
    if (!joy) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    /* Search existing joystick objects to see if we already have this stick.
     */
    cur = joylist_head;
    while (cur) {
        if (cur->joy == joy) {
            Py_INCREF(cur);
            return (PyObject *)cur;
        }
        cur = cur->next;
    }

    /* Construct the Python object */
    jstick = PyObject_New(pgJoystickObject, &pgJoystick_Type);
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

    return (PyObject *)jstick;
}

static PyMethodDef _joystick_methods[] = {
    {"init", (PyCFunction)init, METH_NOARGS, DOC_JOYSTICK_INIT},
    {"quit", (PyCFunction)quit, METH_NOARGS, DOC_JOYSTICK_QUIT},
    {"get_init", (PyCFunction)get_init, METH_NOARGS, DOC_JOYSTICK_GETINIT},
    {"get_count", (PyCFunction)get_count, METH_NOARGS, DOC_JOYSTICK_GETCOUNT},
    {"Joystick", Joystick, METH_VARARGS, DOC_JOYSTICK_JOYSTICK},
    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(joystick)
{
    PyObject *module, *apiobj;
    static void *c_api[PYGAMEAPI_JOYSTICK_NUMSLOTS];

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "joystick",
                                         DOC_JOYSTICK_JOYSTICK,
                                         -1,
                                         _joystick_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }

    import_pygame_color();
    if (PyErr_Occurred()) {
        return NULL;
    }

    /* type preparation */
    if (PyType_Ready(&pgJoystick_Type) == -1) {
        return NULL;
    }

    /* create the module */
    module = PyModule_Create(&_module);
    if (module == NULL) {
        return NULL;
    }

    if (PyModule_AddObjectRef(module, "JoystickType",
                              (PyObject *)&pgJoystick_Type)) {
        Py_DECREF(module);
        return NULL;
    }

    /* export the c api */
    c_api[0] = &pgJoystick_Type;
    c_api[1] = pgJoystick_New;
    c_api[2] = pgJoystick_GetDeviceIndexByInstanceID;
    apiobj = encapsulate_api(c_api, "joystick");
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_XDECREF(apiobj);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
