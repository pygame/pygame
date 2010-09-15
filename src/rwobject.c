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
#include "doc/pygame_doc.h"

/* With Python 2.5 exception types became new-style classes and
 * PyExc_BaseException was introduced.
 */
#if PY_VERSION_HEX < 0x02050000
#define ExcClassType_Check(o) PyClass_Check(o)
#define PyExc_BaseException PyExc_Exception
#else
#define ExcClassType_Check(o) PyType_Check(o)
#endif

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

static const char const default_encoding[] = "unicode_escape";
static const char const default_errors[] = "backslashreplace";

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

static int
is_exception_class(PyObject *obj, void **optr)
{
    PyObject **rval = (PyObject **)optr;
    PyObject *oname;
#if PY3
    PyObject *tmp;
#endif

    if (!ExcClassType_Check(obj) || /* conditional or */
        !PyObject_IsSubclass(obj, PyExc_BaseException)) {
        oname = PyObject_Str(obj);
        if (oname == NULL) {
            return 0;
        }
#if PY3
        tmp = PyUnicode_AsEncodedString(oname, "ascii", "replace");
        Py_DECREF(tmp);
        if (tmp == NULL) {
            return 0;
        }
        oname = tmp;
#endif
        PyErr_Format(PyExc_TypeError,
                     "Expected an exception class: got %.1024s",
                     Bytes_AS_STRING(oname));
        Py_DECREF(oname);
        return 0;
    }
    *rval = obj;
    return 1;
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

static PyObject*
RWopsEncodeString(PyObject *obj,
                  const char *encoding,
                  const char *errors,
                  PyObject *eclass)
{
    PyObject *oencoded;
    PyObject *exc_type;
    PyObject *exc_value;
    PyObject *exc_trace;
    PyObject *str;

    if (obj == NULL) {
        /* Assume an error was raise; forward it */
        return NULL;
    }
    if (encoding == NULL) {
        encoding = default_encoding;
    }
    if (errors == NULL) {
        errors = default_errors;
    }
    if (PyUnicode_Check(obj)) {
        oencoded = PyUnicode_AsEncodedString(obj, encoding, errors);
        if (oencoded != NULL) {
            return oencoded;
        }
        else if (PyErr_ExceptionMatches(PyExc_MemoryError)) {
            /* Forward memory errors */
            return NULL;
        }
        else if (eclass != NULL) {
            /* Foward as eclass error */
            PyErr_Fetch(&exc_type, &exc_value, &exc_trace);
            Py_DECREF(exc_type);
            Py_XDECREF(exc_trace);
            if (exc_value == NULL) {
                PyErr_SetString(eclass,
                                "Unicode encoding error");
            }
            else {
                str = PyObject_Str(exc_value);
                Py_DECREF(exc_value);
                if (str != NULL) {
                    PyErr_SetObject(eclass, str);
                    Py_DECREF(str);
                }
            }
            return NULL;
        }
        else if (encoding == default_encoding && errors == default_errors) {
            /* The default encoding and error handling should not fail */
            return RAISE(PyExc_SystemError,
                         "Pygame bug (in RWopsEncodeString):"
                         " unexpected encoding error");
        }
        PyErr_Clear();
    }
    else if (Bytes_Check(obj)) {
        Py_INCREF(obj);
        return obj;
    }
    
    Py_RETURN_NONE;
}

static PyObject*
RWopsEncodeFilePath(PyObject *obj, PyObject *eclass)
{
    PyObject *result = RWopsEncodeString(obj,
                                         UNICODE_DEF_FS_CODEC, 
                                         UNICODE_DEF_FS_ERROR,
                                         eclass);
    if (result == NULL || result == Py_None) {
        return result;
    }
    if ((size_t)Bytes_GET_SIZE(result) != strlen(Bytes_AS_STRING(result))) {
        if (eclass != NULL) {
            Py_DECREF(result);
            result = RWopsEncodeString(obj, NULL, NULL, NULL);
            if (result == NULL) {
                return NULL;
            }
            PyErr_Format(eclass,
                         "File path '%.1024s' contains null characters",
                         Bytes_AS_STRING(result));
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(result);
        Py_RETURN_NONE;
    }
    return result;
}

static SDL_RWops*
RWopsFromFileObject(PyObject *obj)
{
    SDL_RWops *rw = NULL;
    RWHelper *helper;

    if (obj == NULL) {
        return (SDL_RWops *)RAISE(PyExc_TypeError, "Invalid filetype object");
    }
#if PY2
    if (PyFile_Check(obj))
    {
        rw = SDL_RWFromFP(PyFile_AsFile(obj), 0);
        if (rw) {
            return rw;
        }
        SDL_ClearError();
    }
#endif
    helper = PyMem_New(RWHelper, 1);
    if (helper == NULL) {
        return (SDL_RWops *)PyErr_NoMemory();
    }
    rw = SDL_AllocRW();
    if (rw == NULL) {
        PyMem_Del(helper);
        return (SDL_RWops *)PyErr_NoMemory();
    }
    fetch_object_methods(helper, obj);
    rw->hidden.unknown.data1 = (void *)helper;
    rw->seek = rw_seek;
    rw->read = rw_read;
    rw->write = rw_write;
    rw->close = rw_close;

    return rw;
}

static SDL_RWops*
RWopsFromObject(PyObject *obj)
{
    PyObject *oencoded;
    SDL_RWops *rw = NULL;

    if (obj != NULL) {
        oencoded = RWopsEncodeFilePath(obj, NULL);
        if (oencoded == NULL) {
            return NULL;
        }
        if (oencoded != Py_None) {
            rw = SDL_RWFromFile(Bytes_AS_STRING(oencoded), "rb");
        }
        Py_DECREF(oencoded);
        if (rw) {
            return rw;
        }
        SDL_ClearError();
    }
    return RWopsFromFileObject(obj);
}

static int
RWopsCheckObject (SDL_RWops* rw)
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
RWopsFromFileObjectThreaded(PyObject *obj)
{
    SDL_RWops *rw;
    RWHelper *helper;
    PyInterpreterState *interp;
    PyThreadState *thread;

    if (obj == NULL) {
        return (SDL_RWops *)RAISE(PyExc_TypeError, "Invalid filetype object");
    }

#ifndef WITH_THREAD
    return (SDL_RWops *)RAISE(PyExc_NotImplementedError,
                              "Python built without thread support");
#else
    helper = PyMem_New(RWHelper, 1);
    if (helper == NULL) {
        return (SDL_RWops *)PyErr_NoMemory();
    }
    rw = SDL_AllocRW();
    if (rw == NULL) {
        PyMem_Del(helper);
        return (SDL_RWops *)PyErr_NoMemory();
    }
    fetch_object_methods(helper, obj);
    rw->hidden.unknown.data1 = (void *)helper;
    rw->seek = rw_seek_th;
    rw->read = rw_read_th;
    rw->write = rw_write_th;
    rw->close = rw_close_th;

    PyEval_InitThreads();
    thread = PyThreadState_Get();
    interp = thread->interp;
    helper->thread = PyThreadState_New(interp);

    return rw;
#endif
}

static int
RWopsCheckObjectThreaded (SDL_RWops* rw)
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

static PyObject*
rwobject_encode_string(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *obj = NULL;
    PyObject *eclass = NULL;
    const char *encoding = NULL;
    const char *errors = NULL;
    static char *kwids[] = {"obj", "encoding", "errors", "etype", NULL};  

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|OssO&", kwids,
                                     &obj, &encoding, &errors,
                                     &is_exception_class, &eclass)) {
        return NULL;
    }
    
    if (obj == NULL) {
        RAISE(PyExc_SyntaxError, "Forwarded exception");
    }
    return RWopsEncodeString(obj, encoding, errors, eclass);
}

static PyObject*
rwobject_encode_file_path(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *obj = NULL;
    PyObject *eclass = NULL;
    static char *kwids[] = {"obj", "etype", NULL};  

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|OO&", kwids,
                                     &obj, &is_exception_class, &eclass)) {
        return NULL;
    }
    
    if (obj == NULL) {
        RAISE(PyExc_SyntaxError, "Forwarded exception");
    }
    return RWopsEncodeFilePath(obj, eclass);
}

static PyMethodDef _rwobject_methods[] =
{
    { "encode_string", (PyCFunction)rwobject_encode_string,
      METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEENCODESTRING },
    { "encode_file_path", (PyCFunction)rwobject_encode_file_path,
      METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEENCODEFILEPATH },
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
    c_api[0] = RWopsFromObject;
    c_api[1] = RWopsCheckObject;
    c_api[2] = RWopsFromFileObjectThreaded;
    c_api[3] = RWopsCheckObjectThreaded;
    c_api[4] = RWopsEncodeFilePath;
    c_api[5] = RWopsEncodeString;
    c_api[6] = RWopsFromFileObject;
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
