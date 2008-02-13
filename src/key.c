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

/*
 *  pygame keyboard module
 */
#include "pygame.h"
#include "pygamedocs.h"

/* keyboard module functions */
static PyObject*
key_set_repeat (PyObject* self, PyObject* args)
{
    int delay = 0, interval = 0;

    if (!PyArg_ParseTuple (args, "|ii", &delay, &interval))
        return NULL;

    VIDEO_INIT_CHECK ();

    if (delay && !interval)
        interval = delay;

    if (SDL_EnableKeyRepeat (delay, interval) == -1)
        return RAISE (PyExc_SDLError, SDL_GetError ());

    Py_RETURN_NONE;
}



static PyObject*
key_get_repeat (PyObject* self, PyObject* args)
{
    int delay = 0, interval = 0;

    VIDEO_INIT_CHECK ();
#if SDL_VERSION_ATLEAST(1, 2, 10)
    SDL_GetKeyRepeat (&delay, &interval);
    return Py_BuildValue ("(ii)", delay, interval);
#else
    Py_RETURN_NONE;
#endif
}









static PyObject*
key_get_pressed (PyObject* self)
{
    int num_keys;
    Uint8* key_state;
    PyObject* key_tuple;
    int i;

    VIDEO_INIT_CHECK ();

    key_state = SDL_GetKeyState (&num_keys);

    if (!key_state || !num_keys)
        Py_RETURN_NONE;

    if (!(key_tuple = PyTuple_New (num_keys)))
        return NULL;

    for (i = 0;i < num_keys;i++)
    {
        PyObject* key_elem;
        key_elem = PyInt_FromLong (key_state[i]);
        if (!key_elem)
        {
            Py_DECREF (key_tuple);
            return NULL;
        }		
        PyTuple_SET_ITEM (key_tuple, i, key_elem);
    }
    return key_tuple;
}

static PyObject*
key_name (PyObject* self, PyObject* args)
{
    int key;
	
    if (!PyArg_ParseTuple (args, "i", &key))
        return NULL;

    return PyString_FromString (SDL_GetKeyName (key));	
}

static PyObject*
key_get_mods (PyObject* self)
{
    VIDEO_INIT_CHECK ();

    return PyInt_FromLong (SDL_GetModState ());
}

static PyObject*
key_set_mods (PyObject* self, PyObject* args)
{
    int mods;

    if (!PyArg_ParseTuple (args, "i", &mods))
        return NULL;

    VIDEO_INIT_CHECK ();

    SDL_SetModState (mods);
    Py_RETURN_NONE;
}

static PyObject*
key_get_focused (PyObject* self)
{
    VIDEO_INIT_CHECK ();

    return PyInt_FromLong ((SDL_GetAppState () & SDL_APPINPUTFOCUS) != 0);
}

static PyMethodDef key_builtins[] =
{
    { "set_repeat", key_set_repeat, METH_VARARGS, DOC_PYGAMEKEYSETREPEAT },
    { "get_repeat", key_get_repeat, METH_NOARGS, DOC_PYGAMEKEYGETREPEAT },
    { "get_pressed", (PyCFunction) key_get_pressed, METH_NOARGS,
      DOC_PYGAMEKEYGETPRESSED },
    { "name", key_name, METH_VARARGS, DOC_PYGAMEKEYNAME },
    { "get_mods", (PyCFunction) key_get_mods, METH_NOARGS,
      DOC_PYGAMEKEYGETMODS },
    { "set_mods", key_set_mods, METH_VARARGS, DOC_PYGAMEKEYSETMODS },
    { "get_focused", (PyCFunction) key_get_focused, METH_NOARGS,
      DOC_PYGAMEKEYGETFOCUSED },

    { NULL, NULL, 0, NULL }
};

PYGAME_EXPORT
void initkey (void)
{
    PyObject *module, *dict;

    /* create the module */
    module = Py_InitModule3 ("key", key_builtins, DOC_PYGAMEKEY);
    dict = PyModule_GetDict (module);

    /*imported needed apis*/
    import_pygame_base ();
}
