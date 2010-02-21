/*
  pygame - Python Game Library
  Copyright (C) 2010 Marcus von Appen

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
#define PYGAME_STREAMWRAPPER_INTERNAL

#include <Python.h>
#include "internals.h"
#include "pgbase.h"

static CPyStreamWrapper* CPyStreamWrapper_New (PyObject *obj);
static void CPyStreamWrapper_Free (CPyStreamWrapper *wrapper);
static int CPyStreamWrapper_Read_Threaded (CPyStreamWrapper *wrapper, void *buf,
    pguint32 offset, pguint32 count, pguint32 *read_);
static int CPyStreamWrapper_Read (CPyStreamWrapper *wrapper, void *buf,
    pguint32 offset, pguint32 count, pguint32 *read_);
static int CPyStreamWrapper_Write_Threaded (CPyStreamWrapper *wrapper,
    const void *buf, pguint32 num, pguint32 size, pguint32 *written);
static int CPyStreamWrapper_Write (CPyStreamWrapper *wrapper,
    const void *buf, pguint32 num, pguint32 size, pguint32 *written);
static int CPyStreamWrapper_Seek_Threaded (CPyStreamWrapper *wrapper,
    pgint32 offset, int whence);
static int CPyStreamWrapper_Seek (CPyStreamWrapper *wrapper, pgint32 offset,
    int whence);
static pgint32 CPyStreamWrapper_Tell_Threaded (CPyStreamWrapper *wrapper);
static pgint32 CPyStreamWrapper_Tell (CPyStreamWrapper *wrapper);
static int CPyStreamWrapper_Close_Threaded (CPyStreamWrapper *wrapper);
static int CPyStreamWrapper_Close (CPyStreamWrapper *wrapper);

static int IsReadableStreamObj (PyObject *obj);
static int IsWriteableStreamObj (PyObject *obj);
static int IsReadWriteableStreamObj (PyObject *obj);

static CPyStreamWrapper*
CPyStreamWrapper_New (PyObject *obj)
{
    CPyStreamWrapper *wrapper;
#ifdef WITH_THREAD
    PyInterpreterState* interp;
    PyThreadState* thread;
#endif

    if (!obj)
    {
        PyErr_SetString (PyExc_ValueError, "obj must not be NULL");
        return NULL;
    }

    wrapper = PyMem_New (CPyStreamWrapper, 1);
    if (!wrapper)
        return NULL;

    wrapper->read = NULL;
    wrapper->write = NULL;
    wrapper->seek = NULL;
    wrapper->tell = NULL;
    wrapper->close = NULL;

#ifdef WITH_THREAD    
    PyEval_InitThreads ();
    thread = PyThreadState_Get ();
    interp = thread->interp;
    wrapper->thread = PyThreadState_New (interp);
#endif

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
    return wrapper;
}

static void
CPyStreamWrapper_Free (CPyStreamWrapper *wrapper)
{
    if (!wrapper)
    {
        PyErr_SetString (PyExc_ValueError, "wrapper must not be NULL");
        return;
    }
#ifdef WITH_THREAD
    PyThreadState_Clear (wrapper->thread);
    PyThreadState_Delete (wrapper->thread);
#endif
    Py_XDECREF (wrapper->seek);
    Py_XDECREF (wrapper->tell);
    Py_XDECREF (wrapper->write);
    Py_XDECREF (wrapper->read);
    Py_XDECREF (wrapper->close);
    PyMem_Free (wrapper);
    return;
}

static int
CPyStreamWrapper_Read_Threaded (CPyStreamWrapper *wrapper, void *buf,
    pguint32 offset, pguint32 count, pguint32 *read_)
{
    PyObject *result;
    pguint32 off, _read = 0;
    int retval = 1;
    char *tmp;
    char *ptr = (char*) buf;
#ifdef WITH_THREAD
    PyThreadState* oldstate;
#endif

    if (!wrapper->read || !wrapper->seek || !read_)
        return 0;

#ifdef WITH_THREAD    
    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (wrapper->thread);
#endif

    if (offset != 0)
    {
        /* Move to the correct offset. */
        result = PyObject_CallFunction (wrapper->seek, "li", offset, SEEK_SET);
        if (!result)
        {
            PyErr_Print();
            retval = 0;
            goto end;
        }
        Py_DECREF (result);
    }

    if (count == 0)
    {
        /* Just a seek was wanted */
        goto end;
    }

    result = PyObject_CallFunction (wrapper->read, "l", count);
    if (!result)
    {
        PyErr_Print ();
        retval = 0;
        goto end;
    }
    
    if (!Bytes_Check (result))
    {
        Py_DECREF (result);
        PyErr_Print ();
        retval = 0;
        goto end;
    }

    _read = (pguint32) Bytes_GET_SIZE (result);
    tmp = Bytes_AS_STRING (result);

    off = 0;
    while ((_read - off) > SIZE_MAX)
    {
        memcpy (ptr + off, tmp + off, SIZE_MAX);
        off += SIZE_MAX;
    }
    memcpy (ptr + off, tmp + off, (size_t) (_read - off));
    Py_DECREF (result);

end:
#ifdef WITH_THREAD    
    PyThreadState_Swap (oldstate);
    PyEval_ReleaseLock ();
#endif
    *read_ = _read;
    return retval;
}

static int
CPyStreamWrapper_Read (CPyStreamWrapper *wrapper, void *buf, pguint32 offset,
    pguint32 count, pguint32 *read_)
{
    PyObject *result;
    pguint32 off, _read = 0;
    int retval = 1;
    char *tmp;
    char *ptr = (char*) buf;

    if (!wrapper->read || !wrapper->seek || !read_)
        return 0;

    if (offset != 0)
    {
        /* Move to the correct offset. */
        result = PyObject_CallFunction (wrapper->seek, "li", offset, SEEK_SET);
        if (!result)
        {
            PyErr_Print();
            retval = 0;
            goto end;
        }
        Py_DECREF (result);
    }
    if (count == 0)
    {
        /* Just a seek was wanted */
        goto end;
    }

    result = PyObject_CallFunction (wrapper->read, "l", count);
    if (!result)
    {
        PyErr_Print ();
        retval = 0;
        goto end;
    }
    
    if (!Bytes_Check (result))
    {
        Py_DECREF (result);
        PyErr_Print ();
        retval = 0;
        goto end;
    }
    
    _read = (pguint32) Bytes_GET_SIZE (result);
    tmp = Bytes_AS_STRING (result);

    off = 0;
    while ((_read - off) > SIZE_MAX)
    {
        memcpy (ptr + off, tmp + off, SIZE_MAX);
        off += SIZE_MAX;
    }
    memcpy (ptr + off, tmp + off, (size_t) (_read - off));
    Py_DECREF (result);

end:
    *read_ = _read;
    return retval;
}

static int
CPyStreamWrapper_Write_Threaded (CPyStreamWrapper *wrapper, const void *buf,
    pguint32 num, pguint32 size, pguint32 *written)
{
    PyObject *result;
    int retval = 1;
    const char *ptr = buf;

#ifdef WITH_THREAD
    PyThreadState* oldstate;
#endif

    if (!wrapper->write || !buf || !written)
        return 0;

#ifdef WITH_THREAD
    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (wrapper->thread);
#endif

#ifdef IS_PYTHON_3
    result = PyObject_CallFunction (wrapper->write, "y#", ptr, size * num);
#else
    result = PyObject_CallFunction (wrapper->write, "s#", ptr, size * num);
#endif

    if (!result)
    {
        PyErr_Print ();
        retval = 0;
        goto end;
    }
    Py_DECREF (result);
    *written = num;
    
end:
#ifdef WITH_THREAD
    PyThreadState_Swap (oldstate);
    PyEval_ReleaseLock ();
#endif
    return retval;
}

static int
CPyStreamWrapper_Write (CPyStreamWrapper *wrapper, const void *buf,
    pguint32 num, pguint32 size, pguint32 *written)
{
    PyObject *result;
    int retval = 1;
    const char *ptr = buf;

    if (!wrapper->write || !buf || !written)
        return 0;

#ifdef IS_PYTHON_3
    result = PyObject_CallFunction (wrapper->write, "y#", ptr, size * num);
#else
    result = PyObject_CallFunction (wrapper->write, "s#", ptr, size * num);
#endif

    if (!result)
    {
        PyErr_Print ();
        retval = 0;
        goto end;
    }
    Py_DECREF (result);
    *written = num;
    
end:
    return retval;
}

static int
CPyStreamWrapper_Seek_Threaded (CPyStreamWrapper *wrapper, pgint32 offset,
    int whence)
{
    PyObject* result;
    int retval = 1;
#ifdef WITH_THREAD
    PyThreadState* oldstate;
#endif

    if (!wrapper->seek)
        return 0;

#ifdef WITH_THREAD
    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (wrapper->thread);
#endif

    result = PyObject_CallFunction (wrapper->seek, "li", offset, whence);
    if (!result)
    {
        PyErr_Print();
        retval = 0;
        goto end;
    }
    Py_DECREF (result);
end:
#ifdef WITH_THREAD
    PyThreadState_Swap (oldstate);
    PyEval_ReleaseLock ();
#endif
    return retval;
}

static int
CPyStreamWrapper_Seek (CPyStreamWrapper *wrapper, pgint32 offset, int whence)
{
    PyObject* result;
    int retval = 1;

    if (!wrapper->seek)
        return 0;

    result = PyObject_CallFunction (wrapper->seek, "li", offset, whence);
    if (!result)
    {
        PyErr_Print();
        retval = 0;
        goto end;
    }
    Py_DECREF (result);
end:
    return retval;
}

static pgint32
CPyStreamWrapper_Tell_Threaded (CPyStreamWrapper *wrapper)
{
    PyObject* result;
    pgint32 retval = 0;
#ifdef WITH_THREAD
    PyThreadState* oldstate;
#endif

    if (!wrapper->tell)
        return -1;

#ifdef WITH_THREAD
    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (wrapper->thread);
#endif

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
#ifdef WITH_THREAD
    PyThreadState_Swap (oldstate);
    PyEval_ReleaseLock ();
#endif
    return retval;
}

static pgint32
CPyStreamWrapper_Tell (CPyStreamWrapper *wrapper)
{
    PyObject* result;
    pgint32 retval = 0;

    if (!wrapper->tell)
        return -1;

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
    return retval;
}

static int
CPyStreamWrapper_Close_Threaded (CPyStreamWrapper *wrapper)
{
    PyObject *result;
    int retval = 1;
#ifdef WITH_THREAD
    PyThreadState* oldstate;
#endif

    if (!wrapper->close)
        return -1;

#ifdef WITH_THREAD
    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (wrapper->thread);
#endif

    if (wrapper->close)
    {
        result = PyObject_CallFunction (wrapper->close, NULL);
        if (!result)
        {
            PyErr_Print ();
            retval = 0;
        }
        Py_XDECREF (result);
    }

#ifdef WITH_THREAD
    PyThreadState_Swap (oldstate);
    PyEval_ReleaseLock ();
#endif
    return retval;
}

static int
CPyStreamWrapper_Close (CPyStreamWrapper *wrapper)
{
    PyObject *result;
    int retval = 1;
    
    if (!wrapper->close)
        return -1;

    if (wrapper->close)
    {
        result = PyObject_CallFunction (wrapper->close, NULL);
        if (!result)
        {
            PyErr_Print ();
            retval = 0;
        }
        Py_XDECREF (result);
    }

    return retval;
}

static int
IsReadableStreamObj (PyObject *obj)
{
    PyObject *tmp;

    if (!obj)
    {
        PyErr_SetString (PyExc_ValueError, "obj must not be NULL");
        return 0;
    }

    if (PyObject_HasAttrString (obj, "read"))
    {
        tmp = PyObject_GetAttrString (obj, "read");
        if (tmp && !PyCallable_Check (tmp))
        {
            Py_DECREF (tmp);
            return 0;
        }
    }
    else
        return 0;

    if (PyObject_HasAttrString (obj, "seek"))
    {
        tmp = PyObject_GetAttrString (obj, "seek");
        if (tmp && !PyCallable_Check (tmp))
        {
            Py_DECREF (tmp);
            return 0;
        }
    }
    else
        return 0;

    if (PyObject_HasAttrString (obj, "tell"))
    {
        tmp = PyObject_GetAttrString (obj, "tell");
        if (tmp && !PyCallable_Check (tmp))
        {
            Py_DECREF (tmp);
            return 0;
        }
    }
    else
        return 0;

    return 1;
}

static int
IsWriteableStreamObj (PyObject *obj)
{
    PyObject *tmp;

    if (!obj)
    {
        PyErr_SetString (PyExc_ValueError, "obj must not be NULL");
        return 0;
    }

    if (PyObject_HasAttrString (obj, "write"))
    {
        tmp = PyObject_GetAttrString (obj, "write");
        if (tmp && !PyCallable_Check (tmp))
        {
            Py_DECREF (tmp);
            return 0;
        }
    }
    else
        return 0;

    if (PyObject_HasAttrString (obj, "seek"))
    {
        tmp = PyObject_GetAttrString (obj, "seek");
        if (tmp && !PyCallable_Check (tmp))
        {
            Py_DECREF (tmp);
            return 0;
        }
    }
    else
        return 0;

    if (PyObject_HasAttrString (obj, "tell"))
    {
        tmp = PyObject_GetAttrString (obj, "tell");
        if (tmp && !PyCallable_Check (tmp))
        {
            Py_DECREF (tmp);
            return 0;
        }
    }
    else
        return 0;
    return 1;
}

static int
IsReadWriteableStreamObj (PyObject *obj)
{
    PyObject *tmp;

    if (!obj)
    {
        PyErr_SetString (PyExc_ValueError, "obj must not be NULL");
        return 0;
    }

    if (PyObject_HasAttrString (obj, "read"))
    {
        tmp = PyObject_GetAttrString (obj, "read");
        if (tmp && !PyCallable_Check (tmp))
        {
            Py_DECREF (tmp);
            return 0;
        }
    }
    else
        return 0;

    if (PyObject_HasAttrString (obj, "write"))
    {
        tmp = PyObject_GetAttrString (obj, "write");
        if (tmp && !PyCallable_Check (tmp))
        {
            Py_DECREF (tmp);
            return 0;
        }
    }
    else
        return 0;

    if (PyObject_HasAttrString (obj, "seek"))
    {
        tmp = PyObject_GetAttrString (obj, "seek");
        if (tmp && !PyCallable_Check (tmp))
        {
            Py_DECREF (tmp);
            return 0;
        }
    }
    else
        return 0;

    if (PyObject_HasAttrString (obj, "tell"))
    {
        tmp = PyObject_GetAttrString (obj, "tell");
        if (tmp && !PyCallable_Check (tmp))
        {
            Py_DECREF (tmp);
            return 0;
        }
    }
    else
        return 0;
    return 1;
}

void
streamwrapper_export_capi (void **capi)
{
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT] = (void *)CPyStreamWrapper_New;
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT+1] = (void *)CPyStreamWrapper_Free;
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT+2] = (void *)CPyStreamWrapper_Read_Threaded;
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT+3] = (void *)CPyStreamWrapper_Read;
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT+4] = (void *)CPyStreamWrapper_Write_Threaded;
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT+5] = (void *)CPyStreamWrapper_Write;
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT+6] = (void *)CPyStreamWrapper_Seek_Threaded;
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT+7] = (void *)CPyStreamWrapper_Seek;
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT+8] = (void *)CPyStreamWrapper_Tell_Threaded;
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT+9] = (void *)CPyStreamWrapper_Tell;
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT+10] = (void *)CPyStreamWrapper_Close_Threaded;
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT+11] = (void *)CPyStreamWrapper_Close;
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT+12] = (void *)IsReadableStreamObj;
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT+13] = (void *)IsWriteableStreamObj;
    capi[PYGAME_STREAMWRAPPER_FIRSTSLOT+14] = (void *)IsReadWriteableStreamObj;
}
