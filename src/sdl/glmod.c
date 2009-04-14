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
#define PYGAME_SDLGL_INTERNAL

#include "pgsdl.h"
#include "sdlgl_doc.h"

static PyObject* _sdl_glloadlibrary (PyObject *self, PyObject *args);
static PyObject* _sdl_glgetprocaddress (PyObject *self, PyObject *args);
static PyObject* _sdl_glgetattribute (PyObject *self, PyObject *args);
static PyObject* _sdl_glsetattribute (PyObject *self, PyObject *args);
static PyObject* _sdl_glswapbuffers (PyObject *self);

static PyMethodDef _gl_methods[] = {
    { "load_library", _sdl_glloadlibrary, METH_VARARGS, DOC_GL_LOAD_LIBRARY },
    { "get_proc_address", _sdl_glgetprocaddress, METH_VARARGS,
      DOC_GL_GET_PROC_ADDRESS },
    { "get_attribute", _sdl_glgetattribute, METH_VARARGS,
      DOC_GL_GET_ATTRIBUTE },
    { "set_attribute", _sdl_glsetattribute, METH_VARARGS,
      DOC_GL_SET_ATTRIBUTE },
    { "swap_buffers", (PyCFunction) _sdl_glswapbuffers, METH_NOARGS,
      DOC_GL_SWAP_BUFFERS },
    { NULL, NULL, 0, NULL }
};

static PyObject*
_sdl_glloadlibrary (PyObject *self, PyObject *args)
{
    const char *path;
    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple (args, "s:load_library", &path))
        return NULL;

    if (SDL_GL_LoadLibrary (path) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject*
_sdl_glgetprocaddress (PyObject *self, PyObject *args)
{
    const char *proc;
    void *ptr;
    PyObject *retval;

    ASSERT_VIDEO_SURFACE_SET (NULL);

    if (!PyArg_ParseTuple (args, "s:get_proc_address", &proc))
        return NULL;

    ptr = SDL_GL_GetProcAddress (proc);
    if (!ptr)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    retval = PyCObject_FromVoidPtr (ptr, NULL);
    return retval;
}

static PyObject*
_sdl_glgetattribute (PyObject *self, PyObject *args)
{
    SDL_GLattr attr;
    int value;

    ASSERT_VIDEO_SURFACE_SET (NULL);

    if (!PyArg_ParseTuple (args, "i:get_attribute", &attr))
        return NULL;

    if (SDL_GL_GetAttribute (attr, &value) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    return PyInt_FromLong (value);
}

static PyObject*
_sdl_glsetattribute (PyObject *self, PyObject *args)
{
    SDL_GLattr attr;
    int value;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "ii:set_attribute", &attr, &value))
        return NULL;

    if (SDL_GL_SetAttribute (attr, value) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_sdl_glswapbuffers (PyObject *self)
{
    ASSERT_VIDEO_SURFACE_SET(NULL);
    SDL_GL_SwapBuffers ();
    Py_RETURN_NONE;
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_gl (void)
#else
PyMODINIT_FUNC initgl (void)
#endif

{
    PyObject *mod;

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "gl",
        DOC_GL,
        -1,
        _gl_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("gl", _gl_methods, DOC_GL);
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
