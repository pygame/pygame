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
#define PYGAME_SDLJOYSTICK_INTERNAL

#include "joystickmod.h"
#include "pgsdl.h"
#include "sdljoystick_doc.h"

static int _joystick_clear (PyObject *mod);

static PyObject* _sdl_joyinit (PyObject *self);
static PyObject* _sdl_joywasinit (PyObject *self);
static PyObject* _sdl_joyquit (PyObject *self);
static PyObject* _sdl_joynumjoysticks (PyObject *self);
static PyObject* _sdl_joygetname (PyObject *self, PyObject *args);
static PyObject* _sdl_joyupdate (PyObject *self);
static PyObject* _sdl_joyeventstate (PyObject *self, PyObject *args);
static PyObject* _sdl_joyopened (PyObject *self, PyObject *args);

static PyMethodDef _joystick_methods[] = {
    { "init", (PyCFunction) _sdl_joyinit, METH_NOARGS, DOC_JOYSTICK_INIT },
    { "was_init", (PyCFunction) _sdl_joywasinit, METH_NOARGS,
      DOC_JOYSTICK_WAS_INIT },
    { "quit", (PyCFunction) _sdl_joyquit, METH_NOARGS, DOC_JOYSTICK_QUIT },
    { "num_joysticks", (PyCFunction) _sdl_joynumjoysticks, METH_NOARGS,
      DOC_JOYSTICK_NUM_JOYSTICKS },
    { "get_name", _sdl_joygetname, METH_O, DOC_JOYSTICK_GET_NAME },
    { "update", (PyCFunction) _sdl_joyupdate, METH_NOARGS,
      DOC_JOYSTICK_UPDATE },
    { "event_state", _sdl_joyeventstate, METH_O, DOC_JOYSTICK_EVENT_STATE },
    { "opened", _sdl_joyopened, METH_O, DOC_JOYSTICK_OPENED },
    { NULL, NULL, 0, NULL }
};

static PyObject*
_sdl_joyinit (PyObject *self)
{
    if (SDL_WasInit (SDL_INIT_JOYSTICK))
        Py_RETURN_NONE;
        
    if (SDL_InitSubSystem (SDL_INIT_JOYSTICK) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_sdl_joywasinit (PyObject *self)
{
    if (SDL_WasInit (SDL_INIT_JOYSTICK))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject*
_sdl_joyquit (PyObject *self)
{
    int i;
    _SDLJoystickState *state = SDLJOYSTICK_MOD_STATE (self);

    for (i = 0; i < MAX_JOYSTICKS; i++)
    {
        /* Close all open joysticks. */
        if (state->joysticks[i])
        {
            SDL_JoystickClose (state->joysticks[i]);
            state->joysticks[i] = NULL;
        }
    }

    if (SDL_WasInit (SDL_INIT_JOYSTICK))
        SDL_QuitSubSystem (SDL_INIT_JOYSTICK);
    Py_RETURN_NONE;
}

static PyObject*
_sdl_joynumjoysticks (PyObject *self)
{
    ASSERT_JOYSTICK_INIT(NULL);
    return PyInt_FromLong (SDL_NumJoysticks ());
}

static PyObject*
_sdl_joygetname (PyObject *self, PyObject *args)
{
    int joy;
    ASSERT_JOYSTICK_INIT(NULL);

    if (!IntFromObj (args, &joy))
        return NULL;

    if (joy < 0 || joy >= SDL_NumJoysticks())
    {
        PyErr_SetString (PyExc_ValueError, "invalid joystick index");
        return NULL;
    }
    return Text_FromUTF8 (SDL_JoystickName (joy));
}

static PyObject*
_sdl_joyupdate (PyObject *self)
{
    SDL_JoystickUpdate ();
    Py_RETURN_NONE;
}

static PyObject*
_sdl_joyeventstate (PyObject *self, PyObject *args)
{
    int state;

    /* TODO: is that necessary? */
    /*ASSERT_VIDEO_INIT(NULL);*/

    if (!IntFromObj (args, &state))
        return NULL;

    return PyInt_FromLong (SDL_JoystickEventState (state));
}

static PyObject*
_sdl_joyopened (PyObject *self, PyObject *args)
{
    int joy;
    ASSERT_JOYSTICK_INIT(NULL);

    if (!IntFromObj (args, &joy))
        return NULL;
    if (joy < 0 || joy >= SDL_NumJoysticks())
    {
        PyErr_SetString (PyExc_ValueError, "invalid joystick index");
        return NULL;
    }
    return PyBool_FromLong (SDL_JoystickOpened (joy));
}

static int
_joystick_clear (PyObject *mod)
{
    int i;
    _SDLJoystickState *state = SDLJOYSTICK_MOD_STATE (mod);
    for (i = 0; i < MAX_JOYSTICKS; i++)
    {
        /* Close all open joysticks. */
        if (state->joysticks[i])
        {
            SDL_JoystickClose (state->joysticks[i]);
            state->joysticks[i] = NULL;
        }
    }
    return 0;
}

void
joystickmod_add_joystick (int _index, SDL_Joystick *joystick)
{
    if (_index < 0 || _index >= MAX_JOYSTICKS)
        return;
    SDLJOYSTICK_STATE->joysticks[_index] = joystick;
}

void
joystickmod_remove_joystick (int _index)
{
    if (_index < 0 || _index >= MAX_JOYSTICKS)
        return;
    SDLJOYSTICK_STATE->joysticks[_index] = NULL;
}

SDL_Joystick*
joystickmod_get_joystick (int _index)
{
    if (_index < 0 || _index >= MAX_JOYSTICKS)
        return NULL;
    return SDLJOYSTICK_STATE->joysticks[_index];
}


#ifdef IS_PYTHON_3
struct PyModuleDef _joystickmodule = {
    PyModuleDef_HEAD_INIT,
    "joystick",
    DOC_JOYSTICK,
    sizeof (_SDLJoystickState),
    _joystick_methods,
    NULL,
    NULL,
    _joystick_clear,
    NULL
    };
#else
_SDLJoystickState _modstate;
#endif

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_joystick (void)
#else
PyMODINIT_FUNC initjoystick (void)
#endif
{
    PyObject *mod = NULL;
    PyObject *c_api_obj;
    int i;
    _SDLJoystickState *state;
    static void *c_api[PYGAME_SDLJOYSTICK_SLOTS];

    /* Complete types */
    if (PyType_Ready (&PyJoystick_Type) < 0)
        goto fail;
    Py_INCREF (&PyJoystick_Type);


#ifdef IS_PYTHON_3
    mod = PyModule_Create (&_joystickmodule);
#else
    mod = Py_InitModule3 ("joystick", _joystick_methods, DOC_JOYSTICK);
#endif
    if (!mod)
        goto fail;

    state = SDLJOYSTICK_MOD_STATE(mod);
    for (i = 0; i < MAX_JOYSTICKS; i++)
        state->joysticks[i] = NULL;

    PyModule_AddObject (mod, "Joystick", (PyObject *) &PyJoystick_Type);

    joystick_export_capi (c_api);

    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
        PyModule_AddObject (mod, PYGAME_SDLJOYSTICK_ENTRY, c_api_obj);    

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;

    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
