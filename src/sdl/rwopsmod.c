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
    PyObject* read;
    PyObject* write;
    PyObject* seek;
    PyObject* tell;
    PyObject* close;
#ifdef WITH_THREAD
    PyThreadState* thread;
#endif
} _RWHelper;

static SDL_RWops* get_standard_rwop (PyObject* obj);
static void fetch_object_methods (_RWHelper* helper, PyObject* obj);

static int rw_seek (SDL_RWops* context, int offset, int whence);
static int rw_read (SDL_RWops* context, void* ptr, int size, int maxnum);
static int rw_write (SDL_RWops* context, const void* ptr, int size, int maxnum);
static int rw_close (SDL_RWops* context);

#ifdef WITH_THREAD
static int rw_seek_th (SDL_RWops* context, int offset, int whence);
static int rw_read_th (SDL_RWops* context, void* ptr, int size, int maxnum);
static int rw_write_th (SDL_RWops* context, const void* ptr, int size,
                        int maxnum);
static int rw_close_th (SDL_RWops* context);
#endif

/* C API */
static SDL_RWops* RWopsFromPython (PyObject* obj);
static int RWopsCheckPython (SDL_RWops* rw);
static SDL_RWops* RWopsFromPythonThreaded (PyObject* obj);
static int RWopsCheckPythonThreaded (SDL_RWops* rw);

static SDL_RWops*
get_standard_rwop (PyObject* obj)
{
#ifdef IS_PYTHON_3
    int fd;
#endif
    if (IsTextObj (obj))
    {
        int result;
        char* name;
        PyObject* tuple = PyTuple_New (1);
        PyTuple_SET_ITEM (tuple, 0, obj);
        Py_INCREF (obj);
        if (!tuple)
            return NULL;
        result = PyArg_ParseTuple (tuple, "s", &name);
        Py_DECREF (tuple);
        if (!result)
            return NULL;
        /* TODO: allow wb ops! */
        return SDL_RWFromFile (name, "rb");
    }
#ifdef IS_PYTHON_3
    else if ((fd = PyObject_AsFileDescriptor (obj)) != -1)
    {
        FILE *fp = fdopen (fd, "rb"); /* TODO: is that safe? */
        if (!fp)
        {
            PyErr_SetString (PyExc_IOError, "could not open file");
            return NULL;
        }
        return SDL_RWFromFP (fp, 1);
    }
#else
    else if (PyFile_Check(obj))
        return SDL_RWFromFP (PyFile_AsFile (obj), 0);
#endif
    return NULL;
}

static void
fetch_object_methods (_RWHelper* helper, PyObject* obj)
{
    helper->read = helper->write = helper->seek = helper->tell =
        helper->close = NULL;

    if (PyObject_HasAttrString (obj, "read"))
    {
        helper->read = PyObject_GetAttrString (obj, "read");
        if(helper->read && !PyCallable_Check (helper->read))
        {
            Py_DECREF (helper->read);
            helper->read = NULL;
        }
    }
    if (PyObject_HasAttrString (obj, "write"))
    {
        helper->write = PyObject_GetAttrString (obj, "write");
        if (helper->write && !PyCallable_Check (helper->write))
        {
            Py_DECREF (helper->write);
            helper->write = NULL;
        }
    }
    if (PyObject_HasAttrString (obj, "seek"))
    {
        helper->seek = PyObject_GetAttrString (obj, "seek");
        if (helper->seek && !PyCallable_Check (helper->seek))
        {
            Py_DECREF (helper->seek);
            helper->seek = NULL;
        }
    }
    if (PyObject_HasAttrString (obj, "tell"))
    {
        helper->tell = PyObject_GetAttrString (obj, "tell");
        if (helper->tell && !PyCallable_Check (helper->tell))
        {
            Py_DECREF (helper->tell);
            helper->tell = NULL;
        }
    }
    if(PyObject_HasAttrString(obj, "close"))
    {
        helper->close = PyObject_GetAttrString (obj, "close");
        if (helper->close && !PyCallable_Check (helper->close))
        {
            Py_DECREF (helper->close);
            helper->close = NULL;
        }
    }
}

static int
rw_seek (SDL_RWops* context, int offset, int whence)
{
    _RWHelper* helper = (_RWHelper*) context->hidden.unknown.data1;
    PyObject* result;
    int retval;

    if (!helper->seek || !helper->tell)
        return -1;

    if (!(offset == 0 && whence == SEEK_CUR)) /*being called only for 'tell'*/
    {
        result = PyObject_CallFunction (helper->seek, "ii", offset, whence);
        if (!result)
            return -1;
        Py_DECREF (result);
    }

    result = PyObject_CallFunction (helper->tell, NULL);
    if (!result)
        return -1;

    retval = PyInt_AsLong (result);
    Py_DECREF (result);

    return retval;
}

static int
rw_read (SDL_RWops* context, void* ptr, int size, int maxnum)
{
    _RWHelper* helper = (_RWHelper*) context->hidden.unknown.data1;
    PyObject* result;
    int retval;

    if (!helper->read)
        return -1;

    result = PyObject_CallFunction (helper->read, "i", size * maxnum);
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
rw_write (SDL_RWops* context, const void* ptr, int size, int num)
{
    _RWHelper* helper = (_RWHelper*) context->hidden.unknown.data1;
    PyObject* result;

    if (!helper->write)
        return -1;

    result = PyObject_CallFunction (helper->write, "s#", ptr, size * num);
    if(!result)
        return -1;

    Py_DECREF (result);
    return num;
}

static int
rw_close (SDL_RWops* context)
{
    _RWHelper* helper = (_RWHelper*) context->hidden.unknown.data1;
    PyObject* result;
    int retval = 0;

    if (helper->close)
    {
        result = PyObject_CallFunction (helper->close, NULL);
        if (result)
            retval = -1;
        Py_XDECREF (result);
    }

    Py_XDECREF (helper->seek);
    Py_XDECREF (helper->tell);
    Py_XDECREF (helper->write);
    Py_XDECREF (helper->read);
    Py_XDECREF (helper->close);
    PyMem_Del (helper);
    SDL_FreeRW (context);
    return retval;
}

#ifdef WITH_THREAD
static int
rw_seek_th (SDL_RWops* context, int offset, int whence)
{
    _RWHelper* helper = (_RWHelper*) context->hidden.unknown.data1;
    PyObject* result;
    int retval;
    PyThreadState* oldstate;

    if (!helper->seek || !helper->tell)
        return -1;

    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (helper->thread);

    /* being seek'd, not just tell'd */
    if (!(offset == 0 && whence == SEEK_CUR))
    {
        result = PyObject_CallFunction (helper->seek, "ii", offset, whence);
        if(!result)
        {
            PyErr_Print();
            retval = -1;
            goto end;
        }
        Py_DECREF (result);
    }

    result = PyObject_CallFunction (helper->tell, NULL);
    if (!result)
    {
        PyErr_Print();
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
rw_read_th (SDL_RWops* context, void* ptr, int size, int maxnum)
{
    _RWHelper* helper = (_RWHelper*) context->hidden.unknown.data1;
    PyObject* result;
    int retval;
    PyThreadState* oldstate;

    if (!helper->read)
        return -1;

    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (helper->thread);

    result = PyObject_CallFunction (helper->read, "i", size * maxnum);
    if (!result)
    {
        PyErr_Print();
        retval = -1;
        goto end;
    }

    if (!Bytes_Check (result))
    {
        Py_DECREF (result);
        PyErr_Print();
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
rw_write_th (SDL_RWops* context, const void* ptr, int size, int num)
{
    _RWHelper* helper = (_RWHelper*) context->hidden.unknown.data1;
    PyObject* result;
    int retval;
    PyThreadState* oldstate;

    if (!helper->write)
        return -1;

    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (helper->thread);

    result = PyObject_CallFunction (helper->write, "s#", ptr, size * num);
    if (!result)
    {
        PyErr_Print();
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
rw_close_th (SDL_RWops* context)
{
    _RWHelper* helper = (_RWHelper*) context->hidden.unknown.data1;
    PyObject* result;
    int retval = 0;
    PyThreadState* oldstate;

    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (helper->thread);

    if (helper->close)
    {
        result = PyObject_CallFunction (helper->close, NULL);
        if (!result)
        {
            PyErr_Print();
            retval = -1;
        }
        Py_XDECREF (result);
    }

    Py_XDECREF (helper->seek);
    Py_XDECREF (helper->tell);
    Py_XDECREF (helper->write);
    Py_XDECREF (helper->read);
    Py_XDECREF (helper->close);

    PyThreadState_Swap (oldstate);
    PyThreadState_Clear (helper->thread);
    PyThreadState_Delete (helper->thread);

    PyMem_Del (helper);

    PyEval_ReleaseLock ();

    SDL_FreeRW (context);
    return retval;
}
#endif

/* C API */
static SDL_RWops*
RWopsFromPython (PyObject* obj)
{
    SDL_RWops* rw;
    _RWHelper* helper;

    if (!obj)
    {
        PyErr_SetString (PyExc_TypeError, "Invalid filetype object");
        return NULL;
    }
    rw = get_standard_rwop (obj);
    if (rw)
        return rw;

    helper = PyMem_New (_RWHelper, 1);
    fetch_object_methods (helper, obj);

    rw = SDL_AllocRW ();
    rw->hidden.unknown.data1 = (void*) helper;
    rw->seek = rw_seek;
    rw->read = rw_read;
    rw->write = rw_write;
    rw->close = rw_close;

    return rw;
}

static int
RWopsCheckPython (SDL_RWops* rw)
{
    if (!rw)
    {
        PyErr_SetString (PyExc_TypeError, "rw must not be NULL");
        return -1;
    }
    return rw->close == rw_close;
}

static SDL_RWops*
RWopsFromPythonThreaded (PyObject* obj)
{
    SDL_RWops* rw;
    _RWHelper* helper;
    PyInterpreterState* interp;
    PyThreadState* thread;

    if (!obj)
    {
        PyErr_SetString (PyExc_TypeError, "Invalid filetype object");
        return NULL;
    }

#ifndef WITH_THREAD
    PyErr_SetString (PyExc_NotImplementedError,
        "Python built without thread support");
    return NULL;
#else
    helper = PyMem_New (_RWHelper, 1);
    fetch_object_methods (helper, obj);

    rw = SDL_AllocRW ();
    rw->hidden.unknown.data1 = (void*) helper;
    rw->seek = rw_seek_th;
    rw->read = rw_read_th;
    rw->write = rw_write_th;
    rw->close = rw_close_th;

    PyEval_InitThreads ();
    thread = PyThreadState_Get ();
    interp = thread->interp;
    helper->thread = PyThreadState_New (interp);

    return rw;
#endif
}

static int
RWopsCheckPythonThreaded (SDL_RWops* rw)
{
#ifdef WITH_THREAD
    if (!rw)
    {
        PyErr_SetString (PyExc_TypeError, "rw must not be NULL");
        return -1;
    }
    return rw->close == rw_close_th;
#else
    return 0;
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
#endif

#ifdef IS_PYTHON_3
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("rwops", NULL, "");
#endif
    if (!mod)
        goto fail;

    c_api[PYGAME_SDLRWOPS_FIRSTSLOT+0] = RWopsFromPython;
    c_api[PYGAME_SDLRWOPS_FIRSTSLOT+1] = RWopsCheckPython;
    c_api[PYGAME_SDLRWOPS_FIRSTSLOT+2] = RWopsFromPythonThreaded;
    c_api[PYGAME_SDLRWOPS_FIRSTSLOT+3] = RWopsCheckPythonThreaded;

    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
        PyModule_AddObject (mod, PYGAME_SDLRWOPS_ENTRY, c_api_obj);    

    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
