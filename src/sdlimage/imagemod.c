/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2006 Rene Dudfield

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
#define PYGAME_SDLIMAGE_INTERNAL

#include <SDL_image.h>
#include "pgsdl.h"

static PyObject* _image_geterror (PyObject *self);
static PyObject* _image_load (PyObject *self, PyObject *args);
static PyObject* _image_readxpmfromarray (PyObject *self, PyObject *args);

static PyMethodDef _image_methods[] = {
    { "get_error", (PyCFunction) _image_geterror, METH_NOARGS, "" },
    { "load", _image_load, METH_VARARGS, "" },
    { "read_xpm_from_array", _image_readxpmfromarray, METH_VARARGS, "" },
    { NULL, NULL, 0, NULL },
};

static PyObject*
_image_geterror (PyObject *self)
{
    char *err = IMG_GetError ();
    if (!err)
        Py_RETURN_NONE;
    return Text_FromUTF8 (err);
}

static PyObject*
_image_load (PyObject *self, PyObject *args)
{
    char *filename, *type = NULL;
    SDL_Surface *surface = NULL;
    PyObject *sf, *file;
    
    ASSERT_VIDEO_INIT (NULL);
    
    if (!PyArg_ParseTuple (args, "O|s:load", &file, &type))
        return NULL;

    if (IsTextObj (file))
    {
        PyObject *tmp;
        if (!UTF8FromObject (file, &filename, &tmp))
            return NULL;

        if (type)
        {
            /* If the type's set, it has precedence over the filename. */
            Py_BEGIN_ALLOW_THREADS;
            surface = IMG_LoadTyped_RW (SDL_RWFromFile (filename, "rb"), 1,
                type);
            Py_END_ALLOW_THREADS;
        }
        else
        {
            Py_BEGIN_ALLOW_THREADS;
            surface = IMG_Load (filename);
            Py_END_ALLOW_THREADS;
        }
        Py_XDECREF (tmp);
    }
#ifdef IS_PYTHON_3
    else if (PyObject_AsFileDescriptor (file) != -1)
#else
    else if (PyFile_Check (file))
#endif
    {
        SDL_RWops *rw = RWopsFromPython (file);
        if (!rw)
            return NULL;

        if (type)
        {
            Py_BEGIN_ALLOW_THREADS;
            surface = IMG_LoadTyped_RW (rw, 1, type);
            Py_END_ALLOW_THREADS;
        }
        else
        {
            Py_BEGIN_ALLOW_THREADS;
            surface = IMG_Load_RW (rw, 1);
            Py_END_ALLOW_THREADS;
        }
    }
    else
    {
#ifdef IS_PYTHON_3
        PyErr_Clear (); /* Set by PyObject_AsFileDescriptor() */
#endif
        PyErr_SetString (PyExc_TypeError, "file must be a string or file");
        return NULL;
    }

    if (!surface)
    {
        PyErr_SetString (PyExc_PyGameError, IMG_GetError ());
        return NULL;
    }

    sf = PySDLSurface_NewFromSDLSurface (surface);
    if (!sf)
    {
        SDL_FreeSurface (surface);
        PyErr_SetString (PyExc_PyGameError, IMG_GetError ());
        return NULL;
    }
    return sf;
}

static PyObject*
_image_readxpmfromarray (PyObject *self, PyObject *args)
{
    const char *buf;
    char *copy;
    Py_ssize_t len;
    PyObject *buffer, *sf;
    SDL_Surface *surface;

    if (!PyArg_ParseTuple (args, "O:read_xpm_from_array", &buffer))
        return NULL;

    if (PyObject_AsCharBuffer (buffer, &buf, &len) == -1)
        return NULL;

    copy = PyMem_Malloc ((size_t) len);
    if (!copy)
        return NULL;

    memcpy (copy, buf, (size_t) len);
    surface = IMG_ReadXPMFromArray (&copy);
    PyMem_Free (copy);

    if (!surface)
    {
        PyErr_SetString (PyExc_PyGameError, IMG_GetError ());
        return NULL;
    }

    sf = PySDLSurface_NewFromSDLSurface (surface);
    if (!sf)
    {
        SDL_FreeSurface (surface);
        PyErr_SetString (PyExc_PyGameError, IMG_GetError ());
        return NULL;
    }
    return sf;
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_base (void)
#else
PyMODINIT_FUNC initbase (void)
#endif
{
    PyObject *mod;

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "base",
        "",
        -1,
        _image_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("base", _image_methods, "");
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
