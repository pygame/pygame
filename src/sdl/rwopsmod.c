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
#define PYGAME_SDLRWOPS_INTERNAL

#include "pgsdl.h"

typedef struct
{
    PyObject *read;
    PyObject *write;
    PyObject *seek;
    PyObject *tell;
    PyObject *close;
} _RWWrapper;

static void _bind_python_methods (_RWWrapper *wrapper, PyObject *obj);
static int _pyobj_read (SDL_RWops *ops, void* ptr, int size, int num);
static int _pyobj_seek (SDL_RWops *ops, int offset, int whence);
static int _pyobj_write (SDL_RWops *ops, const void* ptr, int size, int num);
static int _pyobj_close (SDL_RWops *ops);

static void
_bind_python_methods (_RWWrapper *wrapper, PyObject *obj)
{
    wrapper->read = NULL;
    wrapper->write = NULL;
    wrapper->seek = NULL;
    wrapper->tell = NULL;
    wrapper->close = NULL;
    
    if (PyObject_HasAttrString (obj, "read"))
    {
        wrapper->read = PyObject_GetAttrString (obj, "read");
        if (wrapper->read && !PyCallable_Check (wrapper->read))
        {
            Py_DECREF (wrapper->read);
            wrapper->read = NULL;
        }
    }
    if (PyObject_HasAttrString (obj, "write"))
    {
        wrapper->write = PyObject_GetAttrString (obj, "write");
        if (wrapper->write&& !PyCallable_Check (wrapper->write))
        {
            Py_DECREF (wrapper->write);
            wrapper->write = NULL;
        }
    }
    if (PyObject_HasAttrString (obj, "seek"))
    {
        wrapper->seek = PyObject_GetAttrString (obj, "seek");
        if (wrapper->seek && !PyCallable_Check (wrapper->seek))
        {
            Py_DECREF (wrapper->seek);
            wrapper->seek = NULL;
        }
    }
    if (PyObject_HasAttrString (obj, "tell"))
    {
        wrapper->tell = PyObject_GetAttrString (obj, "tell");
        if (wrapper->tell && !PyCallable_Check (wrapper->tell))
        {
            Py_DECREF (wrapper->tell);
            wrapper->tell = NULL;
        }
    }
    if (PyObject_HasAttrString (obj, "close"))
    {
        wrapper->close = PyObject_GetAttrString (obj, "close");
        if (wrapper->close && !PyCallable_Check (wrapper->close))
        {
            Py_DECREF (wrapper->close);
            wrapper->close = NULL;
        }
    }
}

static int
_pyobj_read (SDL_RWops *ops, void* ptr, int size, int maxnum)
{
    _RWWrapper *wrapper = (_RWWrapper *) ops->hidden.unknown.data1;
    PyObject *result;
    int retval;
    
    if (!wrapper->read)
        return -1;
    result = PyObject_CallFunction (wrapper->read, "i", size * maxnum);
    if (!result)
        return -1;
    if (!Bytes_Check (result))
    {
        Py_DECREF (result);
        return -1;
    }
    retval = Bytes_GET_SIZE (result);
    memcpy (ptr, Bytes_AS_STRING (result), (size_t) retval);
    retval /= size;
    
    Py_DECREF (result);
    return retval;
}

static int
_pyobj_seek (SDL_RWops *ops, int offset, int whence)
{
    _RWWrapper *wrapper = (_RWWrapper *) ops->hidden.unknown.data1;
    PyObject* result;
    int retval;

    if (!wrapper->seek || !wrapper->tell)
        return -1;

    if (!(offset == 0 && whence == SEEK_CUR)) /*being called only for 'tell'*/
    {
        result = PyObject_CallFunction (wrapper->seek, "ii", offset, whence);
        if (!result)
            return -1;
        Py_DECREF (result);
    }

    result = PyObject_CallFunction (wrapper->tell, NULL);
    if (!result)
        return -1;

    retval = PyInt_AsLong (result);
    Py_DECREF (result);
    return retval;
}

static int
_pyobj_write (SDL_RWops *ops, const void* ptr, int size, int num)
{
    _RWWrapper *wrapper = (_RWWrapper *) ops->hidden.unknown.data1;
    PyObject *result;

    if (!wrapper->write)
        return -1;

    result = PyObject_CallFunction (wrapper->write, "s#", ptr, size * num);
    if (!result)
        return -1;

    Py_DECREF (result);
    return num;
}

static int
_pyobj_close (SDL_RWops *ops)
{
    _RWWrapper *wrapper = (_RWWrapper *) ops->hidden.unknown.data1;
    PyObject *result;
    int retval = 0;

    if (wrapper->close)
    {
        result = PyObject_CallFunction (wrapper->close, NULL);
        if (result)
            retval = -1;
        Py_XDECREF (result);
    }

    Py_XDECREF (wrapper->seek);
    Py_XDECREF (wrapper->tell);
    Py_XDECREF (wrapper->write);
    Py_XDECREF (wrapper->read);
    Py_XDECREF (wrapper->close);
    PyMem_Del (wrapper);
    SDL_FreeRW (ops);
    return retval;
}

/* C API */
static SDL_RWops*
PyRWops_NewRO (PyObject *obj, int *canautoclose)
{
    _RWWrapper *wrapper;
    SDL_RWops *ops;
    
    if (!obj || !canautoclose)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return NULL;
    }
    
    /* If we have a text object, assume it is a file, which is automatically
     * closed. */
    if (IsTextObj (obj))
    {
        PyObject *tmp;
        char *filename;
        if (!UTF8FromObject (obj, &filename, &tmp))
            return NULL;
        Py_XDECREF (tmp);
        *canautoclose = 1;
        return SDL_RWFromFile ((const char *)filename, "rb");
    }

    /* No text object, so its a buffer or something like that. Try to get the
     * necessary information. */
    ops = SDL_AllocRW ();
    if (!ops)
        return NULL;
    wrapper = PyMem_New (_RWWrapper, 1);
    if (!wrapper)
    {
        SDL_FreeRW (ops);
        return NULL;
    }
    _bind_python_methods (wrapper, obj);
    
    ops->read = _pyobj_read;
    ops->write = _pyobj_write;
    ops->seek = _pyobj_seek;
    ops->close = _pyobj_close;
    ops->hidden.unknown.data1 = (void*) wrapper;
    *canautoclose = 0;
    return ops;
}

static SDL_RWops*
PyRWops_NewRW (PyObject *obj, int *canautoclose)
{
    _RWWrapper *wrapper;
    SDL_RWops *ops;
    
    if (!obj || !canautoclose)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return NULL;
    }
    
    /* If we have a text object, assume it is a file, which is automatically
     * closed. */
    if (IsTextObj (obj))
    {
        PyObject *tmp;
        char *filename;
        if (!UTF8FromObject (obj, &filename, &tmp))
            return NULL;
        Py_XDECREF (tmp);
        *canautoclose = 1;
        return SDL_RWFromFile ((const char *)filename, "wb");
    }

    /* No text object, so its a buffer or something like that. Try to get the
     * necessary information. */
    ops = SDL_AllocRW ();
    if (!ops)
        return NULL;
    wrapper = PyMem_New (_RWWrapper, 1);
    if (!wrapper)
    {
        SDL_FreeRW (ops);
        return NULL;
    }
    _bind_python_methods (wrapper, obj);
    ops->read = _pyobj_read;
    ops->write = _pyobj_write;
    ops->seek = _pyobj_seek;
    ops->close = _pyobj_close;
    ops->hidden.unknown.data1 = (void*) wrapper;
    *canautoclose = 0;
    return ops;
}

static void
PyRWops_Close (SDL_RWops *ops, int canautoclose)
{
    /* internal _RWWrapper? */
    if (ops->close == _pyobj_close)
    {
        if (!canautoclose) /* Do not close the underlying object. */
        {
            _RWWrapper *wrapper = (_RWWrapper *) ops->hidden.unknown.data1;
            Py_DECREF (wrapper->close);
            wrapper->close = NULL;
        }
    }
    SDL_RWclose (ops);
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_rwops (void)
#else
PyMODINIT_FUNC initrwops (void)
#endif
{
    PyObject *mod;
    PyObject *c_api_obj;
    static void *c_api[PYGAME_SDLRWOPS_SLOTS];

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "rwops",
        "",
        -1,
        NULL,
        NULL, NULL, NULL, NULL
    };
#endif

#ifdef IS_PYTHON_3
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("rwops", NULL, "");
#endif
    if (!mod)
        goto fail;

    c_api[PYGAME_SDLRWOPS_FIRSTSLOT] = PyRWops_NewRO;
    c_api[PYGAME_SDLRWOPS_FIRSTSLOT+1] = PyRWops_NewRW;
    c_api[PYGAME_SDLRWOPS_FIRSTSLOT+2] = PyRWops_Close;
    
    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
        PyModule_AddObject (mod, PYGAME_SDLRWOPS_ENTRY, c_api_obj);    

    if (import_pygame2_base () < 0)
        goto fail;

    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
