/*
  pygame - Python Game Library
  Copyright (C) 2008 Marcus von Appen

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

#include "pgsdl.h"
#include "sdlimage_doc.h"

static PyObject* _sdl_loadbmp (PyObject *self, PyObject *args);
static PyObject* _sdl_savebmp (PyObject *self, PyObject *args);

static PyMethodDef _image_methods[] = {
    { "load_bmp", _sdl_loadbmp, METH_O, DOC_IMAGE_LOAD_BMP },
    { "save_bmp", _sdl_savebmp, METH_VARARGS, DOC_IMAGE_SAVE_BMP },
    { NULL, NULL, 0, NULL }
};

static PyObject*
_sdl_loadbmp (PyObject *self, PyObject *args)
{
    SDL_Surface *surface;
    SDL_RWops *rw;
    int autoclose;
    PyObject *sf;

    rw = PyRWops_NewRO_Threaded (args, &autoclose);
    if (!rw)
        return NULL;

    Py_BEGIN_ALLOW_THREADS;
    surface = SDL_LoadBMP_RW (rw, autoclose);
    Py_END_ALLOW_THREADS;
    
    if (!autoclose)
        PyRWops_Close (rw, autoclose);

    if (!surface)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    sf = PySDLSurface_NewFromSDLSurface (surface);
    if (!sf)
    {
        SDL_FreeSurface (surface);
        return NULL;
    }
    return sf;
}

static PyObject*
_sdl_savebmp (PyObject *self, PyObject *args)
{
    PyObject *surface, *file;
    SDL_RWops *rw;
    int _stat, autoclose;
    
    if (!PyArg_ParseTuple (args, "OO:save_bmp", &surface, &file))
        return NULL;
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    rw = PyRWops_NewRW_Threaded (file, &autoclose);
    if (!rw)
        return NULL;
    
    Py_BEGIN_ALLOW_THREADS;
    _stat = SDL_SaveBMP_RW (((PySDLSurface*)surface)->surface, rw, autoclose);
    Py_END_ALLOW_THREADS;
    
    if (!autoclose)
        PyRWops_Close (rw, autoclose);
    
    if (_stat == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_image (void)
#else
PyMODINIT_FUNC initimage (void)
#endif
{
    PyObject *mod;

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT, "image", DOC_IMAGE, -1, _image_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("image", _image_methods, DOC_IMAGE);
#endif
    if (!mod)
        goto fail;

    if (import_pygame2_base () < 0)
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
