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
 *  SDL_RWops support for python objects
 */
#define NO_PYGAME_C_API
#define PYGAMEAPI_RWOBJECT_INTERNAL
#include "pygame.h"
#include "pgcompat.h"

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
} RWHelper;

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

static SDL_RWops*
get_standard_rwop (PyObject* obj)
{
    if (PyUnicode_Check (obj)) {
        PyObject *bytes = Unicode_AsEncodedPath (obj);
        const char *name = NULL;
        SDL_RWops *rw = NULL;
        
        if (!bytes)
        {
            PyErr_Clear ();
            SDL_SetError ("Unable to encode unicode file name");
            return NULL;
        }
        name = Bytes_AS_STRING (bytes);
        rw = SDL_RWFromFile (name, "rb");
        Py_DECREF (bytes);
        return rw;
    }
    if (Bytes_Check (obj))
    {
        const char *name = Bytes_AS_STRING (obj);
        
        return SDL_RWFromFile (name, "rb");
    }
#if PY2
    if (PyFile_Check (obj))
    {
        return SDL_RWFromFP (PyFile_AsFile (obj), 0);
    }
#endif
    return NULL;
}

static void
fetch_object_methods (RWHelper* helper, PyObject* obj)
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

static SDL_RWops*
RWopsFromPython (PyObject* obj)
{
    SDL_RWops* rw;
    RWHelper* helper;

    if (!obj)
    {
        return (SDL_RWops*) RAISE (PyExc_TypeError, "Invalid filetype object");
    }
    rw = get_standard_rwop (obj);
    if (rw)
    {
        return rw;
    }
    SDL_ClearError ();

    helper = PyMem_New (RWHelper, 1);
    if (!helper)
    {
        return (SDL_RWops*) PyErr_NoMemory ();
    }
    fetch_object_methods (helper, obj);

    rw = SDL_AllocRW ();
    if (!rw)
    {
        return (SDL_RWops *) PyErr_NoMemory ();
    }
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
    return rw->close == rw_close;
}


static int
rw_seek (SDL_RWops* context, int offset, int whence)
{
    RWHelper* helper = (RWHelper*) context->hidden.unknown.data1;
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
    RWHelper* helper = (RWHelper*) context->hidden.unknown.data1;
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
    memcpy (ptr, Bytes_AsString (result), retval);
    retval /= size;

    Py_DECREF (result);
    return retval;
}

static int
rw_write (SDL_RWops* context, const void* ptr, int size, int num)
{
    RWHelper* helper = (RWHelper*) context->hidden.unknown.data1;
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
    RWHelper* helper = (RWHelper*) context->hidden.unknown.data1;
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

static SDL_RWops*
RWopsFromPythonThreaded (PyObject* obj)
{
    SDL_RWops* rw;
    RWHelper* helper;
    PyInterpreterState* interp;
    PyThreadState* thread;

    if (!obj)
        return (SDL_RWops*) RAISE (PyExc_TypeError, "Invalid filetype object");

#ifndef WITH_THREAD
    return (SDL_RWops*) RAISE (PyExc_NotImplementedError,
                               "Python built without thread support");
#else
    helper = PyMem_New (RWHelper, 1);
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
    return rw->close == rw_close_th;
#else
    return 0;
#endif
}

#ifdef WITH_THREAD
static int
rw_seek_th (SDL_RWops* context, int offset, int whence)
{
    RWHelper* helper = (RWHelper*) context->hidden.unknown.data1;
    PyObject* result;
    int retval;
    PyThreadState* oldstate;

    if (!helper->seek || !helper->tell)
        return -1;

    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (helper->thread);

    if (!(offset == 0 && whence == SEEK_CUR)) /* being seek'd, not just tell'd */
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
    RWHelper* helper = (RWHelper*) context->hidden.unknown.data1;
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
    memcpy (ptr, Bytes_AsString (result), retval);
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
    RWHelper* helper = (RWHelper*) context->hidden.unknown.data1;
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
    RWHelper* helper = (RWHelper*) context->hidden.unknown.data1;
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

static PyMethodDef _rwobject_methods[] =
{
    { NULL, NULL }
};

/*DOC*/ static char _rwobject_doc[] =
/*DOC*/    "SDL_RWops support";

MODINIT_DEFINE (rwobject)
{
    PyObject *module, *dict, *apiobj;
    int ecode;
    static void* c_api[PYGAMEAPI_RWOBJECT_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "rwobject",
        _rwobject_doc,
        -1,
        _rwobject_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    /* Create the module and add the functions */
#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 (MODPREFIX "rwobject", 
                             _rwobject_methods,
                             _rwobject_doc);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict (module);

    /* export the c api */
    c_api[0] = RWopsFromPython;
    c_api[1] = RWopsCheckPython;
    c_api[2] = RWopsFromPythonThreaded;
    c_api[3] = RWopsCheckPythonThreaded;
    apiobj = PyCObject_FromVoidPtr (c_api, NULL);
    if (apiobj == NULL) {
        DECREF_MOD (module);
	MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString (dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF (apiobj);
    if (ecode == -1) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN (module);
}
