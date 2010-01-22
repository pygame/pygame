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
#ifdef WITH_THREAD
    PyThreadState *thread;
#endif
} _RWWrapper;

static void _bind_python_methods (_RWWrapper *wrapper, PyObject *obj);
static int _pyobj_read (SDL_RWops *ops, void* ptr, int size, int num);
static int _pyobj_seek (SDL_RWops *ops, int offset, int whence);
static int _pyobj_write (SDL_RWops *ops, const void* ptr, int size, int num);
static int _pyobj_close (SDL_RWops *ops);
#ifdef WITH_THREAD
static int _pyobj_read_threaded (SDL_RWops *ops, void* ptr, int size, int num);
static int _pyobj_seek_threaded (SDL_RWops *ops, int offset, int whence);
static int _pyobj_write_threaded (SDL_RWops *ops, const void* ptr, int size,
    int num);
static int _pyobj_close_threaded (SDL_RWops *ops);
#endif

/* C API */
static SDL_RWops* PyRWops_NewRO (PyObject *obj, int *canautoclose);
static SDL_RWops* PyRWops_NewRO_Threaded (PyObject *obj, int *canautoclose);
static SDL_RWops* PyRWops_NewRW (PyObject *obj, int *canautoclose);
static SDL_RWops* PyRWops_NewRW_Threaded (PyObject *obj, int *canautoclose);
static void PyRWops_Close (SDL_RWops *ops, int canautoclose);

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
        if (wrapper->write && !PyCallable_Check (wrapper->write))
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
_pyobj_read (SDL_RWops *ops, void* ptr, int size, int num)
{
    _RWWrapper *wrapper = (_RWWrapper *) ops->hidden.unknown.data1;
    PyObject *result;
    int retval;
    
    if (!wrapper->read)
        return -1;
    result = PyObject_CallFunction (wrapper->read, "i", size * num);
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

#ifdef IS_PYTHON_3
    result = PyObject_CallFunction (wrapper->write, "y#", ptr, size * num);
#else
    result = PyObject_CallFunction (wrapper->write, "s#", ptr, size * num);
#endif
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
        if (!result)
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

#ifdef WITH_THREAD
static int
_pyobj_read_threaded (SDL_RWops *ops, void* ptr, int size, int num)
{
    _RWWrapper *wrapper = (_RWWrapper *) ops->hidden.unknown.data1;
    PyObject *result;
    int retval;
    PyThreadState* oldstate;
    
    if (!wrapper->read)
        return -1;
    
    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (wrapper->thread);
        
    result = PyObject_CallFunction (wrapper->read, "i", size * num);
    if (!result)
    {
        PyErr_Print ();
        retval = -1;
        goto end;
    }
    
    if (!Bytes_Check (result))
    {
        Py_DECREF (result);
        PyErr_Print ();
        retval = -1;
        goto end;
    }
    
    retval = Bytes_GET_SIZE (result);
    memcpy (ptr, Bytes_AS_STRING (result), (size_t) retval);
    retval /= size;
    
    Py_DECREF (result);

end:
    PyThreadState_Swap (oldstate);
    PyEval_ReleaseLock ();
    return retval;
}

static int
_pyobj_seek_threaded (SDL_RWops *ops, int offset, int whence)
{
    _RWWrapper *wrapper = (_RWWrapper *) ops->hidden.unknown.data1;
    PyObject* result;
    int retval;
    PyThreadState* oldstate;

    if (!wrapper->seek || !wrapper->tell)
        return -1;

    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (wrapper->thread);

    if (!(offset == 0 && whence == SEEK_CUR)) /*being called only for 'tell'*/
    {
        result = PyObject_CallFunction (wrapper->seek, "ii", offset, whence);
        if (!result)
        {
            PyErr_Print();
            retval = -1;
            goto end;
        }
        Py_DECREF (result);
    }

    result = PyObject_CallFunction (wrapper->tell, NULL);
    if (!result)
    {
        PyErr_Print ();
        retval = -1;
        goto end;
    }
    retval = PyInt_AsLong (result);
    Py_DECREF (result);

end:
    PyThreadState_Swap (oldstate);
    PyEval_ReleaseLock ();
    return retval;
}

static int
_pyobj_write_threaded (SDL_RWops *ops, const void* ptr, int size, int num)
{
    _RWWrapper *wrapper = (_RWWrapper *) ops->hidden.unknown.data1;
    PyObject *result;
    int retval;
    PyThreadState* oldstate;

    if (!wrapper->write)
        return -1;

    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (wrapper->thread);

#ifdef IS_PYTHON_3
    result = PyObject_CallFunction (wrapper->write, "y#", ptr, size * num);
#else
    result = PyObject_CallFunction (wrapper->write, "s#", ptr, size * num);
#endif
    if (!result)
    {
        PyErr_Print ();
        retval = -1;
        goto end;
    }
    Py_DECREF (result);
    retval = num;
    
end:
    PyThreadState_Swap (oldstate);
    PyEval_ReleaseLock ();
    return retval;
}

static int
_pyobj_close_threaded (SDL_RWops *ops)
{
    _RWWrapper *wrapper = (_RWWrapper *) ops->hidden.unknown.data1;
    PyObject *result;
    int retval = 0;
    PyThreadState* oldstate;

    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (wrapper->thread);

    if (wrapper->close)
    {
        result = PyObject_CallFunction (wrapper->close, NULL);
        if (!result)
        {
            PyErr_Print ();
            retval = -1;
        }
        Py_XDECREF (result);
    }

    Py_XDECREF (wrapper->seek);
    Py_XDECREF (wrapper->tell);
    Py_XDECREF (wrapper->write);
    Py_XDECREF (wrapper->read);
    Py_XDECREF (wrapper->close);
    
    PyThreadState_Swap (oldstate);
    PyThreadState_Clear (wrapper->thread);
    PyThreadState_Delete (wrapper->thread);
    PyMem_Del (wrapper);
    
    PyEval_ReleaseLock ();
    
    SDL_FreeRW (ops);
    return retval;
}
#endif

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
        ops = SDL_RWFromFile ((const char *)filename, "rb");
        if (!ops)
            PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return ops;
    }

    /* No text object, so its a buffer or something like that. Try to get the
     * necessary information. */
    ops = SDL_AllocRW ();
    if (!ops)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
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
        ops = SDL_RWFromFile ((const char *)filename, "wb");
        if (!ops)
            PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return ops;

    }

    /* No text object, so its a buffer or something like that. Try to get the
     * necessary information. */
    ops = SDL_AllocRW ();
    if (!ops)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
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
#ifdef WITH_THREAD
    if (ops->close == _pyobj_close || ops->close == _pyobj_close_threaded)
#else
    if (ops->close == _pyobj_close)
#endif
    {
        if (!canautoclose) /* Do not close the underlying object. */
        {
            _RWWrapper *wrapper = (_RWWrapper *) ops->hidden.unknown.data1;
            Py_DECREF (wrapper->close);
            wrapper->close = NULL;
        }
    }
#ifdef WITH_THREAD
    Py_BEGIN_ALLOW_THREADS;
    SDL_RWclose (ops);
    Py_END_ALLOW_THREADS;
#else
    SDL_RWclose (ops);
#endif
}

static SDL_RWops*
PyRWops_NewRO_Threaded (PyObject *obj, int *canautoclose)
{
#ifndef WITH_THREAD
    /* Fall back to the non-threaded implementation */
    return PyRWops_NewRO (obj, canautoclose);
#else
    _RWWrapper *wrapper;
    SDL_RWops *ops;
    PyInterpreterState* interp;
    PyThreadState* thread;
    
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
        ops = SDL_RWFromFile ((const char *)filename, "rb");
        if (!ops)
            PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return ops;

    }

    /* No text object, so its a buffer or something like that. Try to get the
     * necessary information. */
    ops = SDL_AllocRW ();
    if (!ops)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    wrapper = PyMem_New (_RWWrapper, 1);
    if (!wrapper)
    {
        SDL_FreeRW (ops);
        return NULL;
    }
    _bind_python_methods (wrapper, obj);
    
    ops->read = _pyobj_read_threaded;
    ops->write = _pyobj_write_threaded;
    ops->seek = _pyobj_seek_threaded;
    ops->close = _pyobj_close_threaded;
    ops->hidden.unknown.data1 = (void*) wrapper;
    
    PyEval_InitThreads ();
    thread = PyThreadState_Get ();
    interp = thread->interp;
    wrapper->thread = PyThreadState_New (interp);

    *canautoclose = 0;
    return ops;
#endif
}

static SDL_RWops*
PyRWops_NewRW_Threaded (PyObject *obj, int *canautoclose)
{
#ifndef WITH_THREAD
    /* Fall back to the non-threaded implementation */
    return PyRWops_NewRW (obj, canautoclose);
#else
    _RWWrapper *wrapper;
    SDL_RWops *ops;
    PyInterpreterState* interp;
    PyThreadState* thread;
    
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
        ops = SDL_RWFromFile ((const char *)filename, "wb");
        if (!ops)
            PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return ops;

    }

    /* No text object, so its a buffer or something like that. Try to get the
     * necessary information. */
    ops = SDL_AllocRW ();
    if (!ops)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    wrapper = PyMem_New (_RWWrapper, 1);
    if (!wrapper)
    {
        SDL_FreeRW (ops);
        return NULL;
    }
    _bind_python_methods (wrapper, obj);
    ops->read = _pyobj_read_threaded;
    ops->write = _pyobj_write_threaded;
    ops->seek = _pyobj_seek_threaded;
    ops->close = _pyobj_close_threaded;
    ops->hidden.unknown.data1 = (void*) wrapper;
    
    PyEval_InitThreads ();
    thread = PyThreadState_Get ();
    interp = thread->interp;
    wrapper->thread = PyThreadState_New (interp);
    
    *canautoclose = 0;
    return ops;
#endif
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
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("rwops", NULL, "");
#endif
    if (!mod)
        goto fail;

    c_api[PYGAME_SDLRWOPS_FIRSTSLOT+0] = PyRWops_NewRO;
    c_api[PYGAME_SDLRWOPS_FIRSTSLOT+1] = PyRWops_NewRW;
    c_api[PYGAME_SDLRWOPS_FIRSTSLOT+2] = PyRWops_Close;
    c_api[PYGAME_SDLRWOPS_FIRSTSLOT+3] = PyRWops_NewRO_Threaded;
    c_api[PYGAME_SDLRWOPS_FIRSTSLOT+4] = PyRWops_NewRW_Threaded;

    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
    {
        if (PyModule_AddObject (mod, PYGAME_SDLRWOPS_ENTRY, c_api_obj) == -1)
        {
            Py_DECREF (c_api_obj);
            goto fail;
        }
    }

    if (import_pygame2_base () < 0)
        goto fail;

    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
