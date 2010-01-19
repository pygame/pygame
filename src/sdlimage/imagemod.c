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
#include "sdlimagebase_doc.h"

static PyObject* _image_init (PyObject *self, PyObject *args);
static PyObject* _image_quit (PyObject *self);
static PyObject* _image_geterror (PyObject *self);
static PyObject* _image_load (PyObject *self, PyObject *args);
static PyObject* _image_readxpmfromarray (PyObject *self, PyObject *args);

static PyMethodDef _image_methods[] = {
    { "init", _image_init, METH_VARARGS, DOC_BASE_INIT },
    { "quit", _image_quit, METH_VARARGS, DOC_BASE_QUIT },
    { "get_error", (PyCFunction) _image_geterror, METH_NOARGS,
      DOC_BASE_GET_ERROR },
    { "load", _image_load, METH_VARARGS, DOC_BASE_LOAD },
    { "read_xpm_from_array", _image_readxpmfromarray, METH_O,
      DOC_BASE_READ_XPM_FROM_ARRAY },
    { NULL, NULL, 0, NULL },
};

static PyObject*
_image_init (PyObject *self, PyObject *args)
{
    long flags = IMG_INIT_JPG | IMG_INIT_PNG | IMG_INIT_TIF;
    long retval = 0;

    if (!PyArg_ParseTuple (args, "|l:init", &flags))
        return NULL;
    retval = IMG_Init (flags);
    return PyInt_FromLong (flags);
}

static PyObject*
_image_quit (PyObject *self)
{
    IMG_Quit ();
    Py_RETURN_NONE;
}

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
    char *type = NULL;
    SDL_Surface *surface = NULL;
    PyObject *sf, *file;
    SDL_RWops *rw;
    int autoclose;
    
    ASSERT_VIDEO_INIT (NULL);
    
    if (!PyArg_ParseTuple (args, "O|s:load", &file, &type))
        return NULL;

    rw = PyRWops_NewRO_Threaded (file, &autoclose);
    if (!rw)
        return NULL;

    if (type)
    {
        /* If the type's set, it has precedence over the filename. */
        Py_BEGIN_ALLOW_THREADS;
        surface = IMG_LoadTyped_RW (rw, autoclose, type);
        Py_END_ALLOW_THREADS;
    }
    else
    {
        Py_BEGIN_ALLOW_THREADS;
        surface = IMG_Load_RW (rw, autoclose);
        Py_END_ALLOW_THREADS;
    }
    
    if (!autoclose)
        PyRWops_Close (rw, autoclose);
    
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
    PyObject *sf;
    SDL_Surface *surface;

    if (PyObject_AsCharBuffer (args, &buf, &len) == -1)
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
        DOC_BASE,
        -1,
        _image_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("base", _image_methods, DOC_BASE);
#endif
    if (!mod)
        goto fail;
    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    if (import_pygame2_sdl_rwops () < 0)
        goto fail;
    if (import_pygame2_sdl_video () < 0)
        goto fail;
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
