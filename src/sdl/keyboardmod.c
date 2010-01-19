/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners

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
#define PYGAME_SDLKEYBOARD_INTERNAL

#include "pgsdl.h"
#include "sdlkeyboard_doc.h"

static PyObject* _sdl_keygetkeystate (PyObject *self);
static PyObject* _sdl_keygetmodstate (PyObject *self);
static PyObject* _sdl_keysetmodstate (PyObject *self, PyObject *args);
static PyObject* _sdl_keygetkeyname (PyObject *self, PyObject *args);
static PyObject* _sdl_keyenablekeyrepeat (PyObject *self, PyObject *args);
static PyObject* _sdl_keygetkeyrepeat (PyObject *self);
static PyObject* _sdl_keyenableunicode (PyObject *self, PyObject *args);

static PyMethodDef _key_methods[] = {
    { "get_state", (PyCFunction) _sdl_keygetkeystate, METH_NOARGS, 
      DOC_KEYBOARD_GET_STATE },
    { "get_mod_state", (PyCFunction) _sdl_keygetmodstate, METH_NOARGS,
      DOC_KEYBOARD_GET_MOD_STATE },
    { "set_mod_state", _sdl_keysetmodstate, METH_O,
      DOC_KEYBOARD_SET_MOD_STATE },
    { "get_key_name", _sdl_keygetkeyname, METH_O, DOC_KEYBOARD_GET_KEY_NAME },
    { "enable_repeat", _sdl_keyenablekeyrepeat, METH_VARARGS,
      DOC_KEYBOARD_ENABLE_REPEAT },
    { "get_repeat", (PyCFunction) _sdl_keygetkeyrepeat, METH_NOARGS,
      DOC_KEYBOARD_GET_REPEAT },
    { "enable_unicode", _sdl_keyenableunicode, METH_VARARGS,
      DOC_KEYBOARD_ENABLE_UNICODE },
    { NULL, NULL, 0, NULL }
};

static PyObject*
_sdl_keygetkeystate (PyObject *self)
{
    Uint8 *array;
    int count, i;
    PyObject *dict, *key, *val;
    
    ASSERT_VIDEO_INIT (NULL);
    
    dict = PyDict_New ();
    if (!dict)
        return NULL;
    SDL_PumpEvents ();
    array = SDL_GetKeyState (&count);
    
    for (i = 0; i < count; i++)
    {
        key = PyInt_FromLong (i);
        val = PyBool_FromLong (array[i]);
        if (PyDict_SetItem (dict, key, val) == -1)
        {
            Py_DECREF (key);
            Py_DECREF (val);
            Py_DECREF (dict);
            return NULL;
        }
        Py_DECREF (key);
        Py_DECREF (val);
    }
    return dict;
}

static PyObject*
_sdl_keygetmodstate (PyObject *self)
{
    ASSERT_VIDEO_INIT (NULL);
    return PyLong_FromUnsignedLong (SDL_GetModState ());
}

static PyObject*
_sdl_keysetmodstate (PyObject *self, PyObject *args)
{
    SDLMod mod;
    ASSERT_VIDEO_INIT (NULL);
    
    if (!IntFromObj (args, (int*)&mod))
        return NULL;

    SDL_SetModState (mod);
    Py_RETURN_NONE;
}

static PyObject*
_sdl_keygetkeyname (PyObject *self, PyObject *args)
{
    int key;
    
    ASSERT_VIDEO_INIT (NULL);
    
    if (!IntFromObj (args, &key))
        return NULL;

    if (key < SDLK_FIRST || key >= SDLK_LAST)
    {
        PyErr_SetString (PyExc_ValueError, "invalid key value");
        return 0;
    }
    return Text_FromUTF8 (SDL_GetKeyName ((SDLKey)key));
}

static PyObject*
_sdl_keyenablekeyrepeat (PyObject *self, PyObject *args)
{
    int delay, interval;
    
    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "ii:enable_key_repeat", &delay, &interval))
        return NULL;
    if (delay < 0 || interval < 0)
    {
        PyErr_SetString (PyExc_ValueError,
            "delay and interval must not be negative"); 
        return NULL;
    }
    if (SDL_EnableKeyRepeat (delay, interval) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_sdl_keygetkeyrepeat (PyObject *self)
{
    int delay, interval;
    ASSERT_VIDEO_INIT (NULL);
    SDL_GetKeyRepeat (&delay, &interval);
    return Py_BuildValue ("(ii)", delay, interval);
}

static PyObject*
_sdl_keyenableunicode (PyObject *self, PyObject *args)
{
    PyObject *val = NULL;
    int enable;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "|O:enable_unicode", &val))
        return NULL;

    if (val == NULL)
        enable = SDL_EnableUNICODE (-1);
    else
    {
        enable = PyObject_IsTrue (val);
        if (enable == -1)
            return NULL;
        enable = SDL_EnableUNICODE (enable);
    }        
    
    if (enable == 1)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_keyboard (void)
#else
PyMODINIT_FUNC initkeyboard (void)
#endif
{
    PyObject *mod;

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT, "keyboard", DOC_KEYBOARD, -1, _key_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("keyboard", _key_methods, DOC_KEYBOARD);
#endif
    if (!mod)
        goto fail;

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    if (import_pygame2_sdl_video () < 0)
        goto fail;
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
