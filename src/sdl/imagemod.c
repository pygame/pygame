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

static PyObject* _sdl_loadbmp (PyObject *self, PyObject *args);
static PyObject* _sdl_savebmp (PyObject *self, PyObject *args);

static PyMethodDef _image_methods[] = {
    { "load_bmp", _sdl_loadbmp, METH_VARARGS, "" },
    { "save_bmp", _sdl_savebmp, METH_VARARGS, "" },
    { NULL, NULL, 0, NULL }
};

static PyObject*
_sdl_loadbmp (PyObject *self, PyObject *args)
{
    SDL_Surface *surface;
    PyObject *sf, *file;
    char *filename;

    if (!PyArg_ParseTuple (args, "O:load_bmp", &file))
        return NULL;

    if (IsTextObj (file))
    {
        PyObject *tmp;
        if (!UTF8FromObject (file, &filename, &tmp))
            return NULL;

        Py_BEGIN_ALLOW_THREADS;
        surface = SDL_LoadBMP ((const char*)filename);
        Py_END_ALLOW_THREADS;

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

        Py_BEGIN_ALLOW_THREADS;
        surface = SDL_LoadBMP_RW (rw, 1);
        Py_END_ALLOW_THREADS;
    }
    else
    {
#ifdef IS_PYTHON_3
        PyErr_Clear (); /* Set by the PyObject_AsFileDescriptor() call */
#endif
        PyErr_SetString (PyExc_TypeError, "file must be a string or file");
        return NULL;
    }

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
    PyObject *surface;
    char *filename;
    int _stat;
    
    if (!PyArg_ParseTuple (args, "OO:save_bmp", &surface, &filename))
        return NULL;
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    _stat = SDL_SaveBMP (((PySDLSurface*)surface)->surface, filename);
    Py_END_ALLOW_THREADS;
    
    if (_stat == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

#if PY_VERSION_HEX >= 0x03000000
PyMODINIT_FUNC PyInit_image (void)
#else
PyMODINIT_FUNC initimage (void)
#endif
{
    PyObject *mod;

#if PY_VERSION_HEX >= 0x03000000
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT, "image", "", -1, _image_methods,
        NULL, NULL, NULL, NULL
    };
#endif

#if PY_VERSION_HEX < 0x03000000
    mod = Py_InitModule3 ("image", _image_methods, "");
#else
    mod = PyModule_Create (&_module);
#endif
    if (!mod)
        goto fail;

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_video () < 0)
        goto fail;
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
