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

static const char pg_default_encoding[] = "unicode_escape";
static const char pg_default_errors[] = "backslashreplace";

#define PATHLIB "pathlib"
#define PUREPATH "PurePath"

/* Define a few type aliases for easy SDL1/2 compat */
#if IS_SDLv1
typedef int pg_int_t;
typedef int pg_size_t;
#else /* IS_SDLv2 */
typedef Sint64 pg_int_t;
typedef size_t pg_size_t;
#endif /* IS_SDLv2 */


#if IS_SDLv2
static pg_int_t
_pg_rw_size(SDL_RWops *);
#endif /* IS_SDLv2 */
static pg_int_t
_pg_rw_seek(SDL_RWops *, pg_int_t, int);
static pg_size_t
_pg_rw_read(SDL_RWops *, void *, pg_size_t, pg_size_t);
static pg_size_t
_pg_rw_write(SDL_RWops *, const void *, pg_size_t, pg_size_t);
static int
_pg_rw_close(SDL_RWops *);

/* Converter function used by PyArg_ParseTupleAndKeywords with the "O&" format.
 *
 * Returns: 1 on success
 *          0 on fail (with exception set)
 */
static int
_pg_is_exception_class(PyObject *obj, void **optr)
{
    PyObject **rval = (PyObject **)optr;
    PyObject *oname;
#if PY3
    PyObject *tmp;
#endif

    if (!PyType_Check(obj) || /* conditional or */
        !PyObject_IsSubclass(obj, PyExc_BaseException)) {
        oname = PyObject_Str(obj);
        if (oname == NULL) {
            PyErr_SetString(PyExc_TypeError,
                            "invalid exception class argument");
            return 0;
        }
#if PY3
        tmp = PyUnicode_AsEncodedString(oname, "ascii", "replace");
        Py_DECREF(oname);

        if (tmp == NULL) {
            PyErr_SetString(PyExc_TypeError,
                            "invalid exception class argument");
            return 0;
        }

        oname = tmp;
#endif /* PY3 */
        PyErr_Format(PyExc_TypeError,
                     "Expected an exception class: got %.1024s",
                     Bytes_AS_STRING(oname));
        Py_DECREF(oname);
        return 0;
    }
    *rval = obj;
    return 1;
}

static int
_is_filelike_obj(PyObject *obj)
{
    int ret = 1;
    PyObject *temp;
    if (!obj)
        return 0;

    /* These are the bare minimum conditions for an object to qualify
     * as a usable valid file-like object */
    if (!PyObject_HasAttrString(obj, "read")
        || !PyObject_HasAttrString(obj, "write")
        || !PyObject_HasAttrString(obj, "seek")
#if PY2
        || !PyObject_HasAttrString(obj, "tell")
#endif
    )
        return 0;

    /* SDL uses the seek method a lot, so bail out early if our object
     * does not support seeking, rather than getting lots of errors
     * later */
    if (PyObject_HasAttrString(obj, "seekable")) {
        temp = PyObject_CallMethod(obj, "seekable", NULL);
        if (!temp) {
            PyErr_Clear();
            return 0;
        }
        ret = PyObject_IsTrue(temp);
        Py_DECREF(temp);
    }
    return ret;
}

/* This function is meant to decode a pathlib object into its str/bytes representation.
 * It is based on PyOS_FSPath, and defines this function on python 3.4, 3.5 */
static PyObject *
_trydecode_pathlibobj(PyObject *obj)
{
#if PY_VERSION_HEX >= 0x03060000
    PyObject *ret = PyOS_FSPath(obj);
    if (!ret) {
        /* A valid object was not passed. But we do not consider it an error */
        PyErr_Clear();
        Py_INCREF(obj);
        return obj;
    }
    return ret;
#elif PY_VERSION_HEX >= 0x03040000
    /* Custom implementation for back-compat */
    int ret;
    PyObject *pathlib, *purepath;

    pathlib = PyImport_ImportModule(PATHLIB);
    if (!pathlib)
        return NULL;

    purepath = PyObject_GetAttrString(pathlib, PUREPATH);
    if (!purepath) {
        Py_DECREF(pathlib);
        return NULL;
    }

    ret = PyObject_IsInstance(obj, purepath);

    Py_DECREF(pathlib);
    Py_DECREF(purepath);

    if (ret == 1)
        return PyObject_Str(obj);
    else if (ret == 0) {
        Py_INCREF(obj);
        return obj;
    }
    else
        return NULL;
#else
    /* Pathlib module does not exist, just incref and return */
    Py_INCREF(obj);
    return obj;
#endif
}

static PyObject *
pg_EncodeString(PyObject *obj, const char *encoding, const char *errors,
                PyObject *eclass)
{
    PyObject *oencoded, *exc_type, *exc_value, *exc_trace, *str, *ret;

    if (obj == NULL) {
        /* Assume an error was raised; forward it */
        return NULL;
    }
    if (encoding == NULL) {
        encoding = pg_default_encoding;
    }
    if (errors == NULL) {
        errors = pg_default_errors;
    }

    ret = _trydecode_pathlibobj(obj);
    if (!ret)
        return NULL;

    if (PyUnicode_Check(ret)) {
        oencoded = PyUnicode_AsEncodedString(ret, encoding, errors);
        Py_DECREF(ret);

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
                PyErr_SetString(eclass, "Unicode encoding error");
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
        else if (encoding == pg_default_encoding &&
                 errors == pg_default_errors) {
            /* The default encoding and error handling should not fail */
            return RAISE(PyExc_SystemError,
                         "Pygame bug (in pg_EncodeString):"
                         " unexpected encoding error");
        }
        PyErr_Clear();
        Py_RETURN_NONE;
    }

    if (Bytes_Check(ret)) {
        return ret;
    }

    Py_DECREF(ret);
    Py_RETURN_NONE;
}

static PyObject *
pg_EncodeFilePath(PyObject *obj, PyObject *eclass)
{
    PyObject *result = pg_EncodeString(obj, UNICODE_DEF_FS_CODEC,
                                            UNICODE_DEF_FS_ERROR, eclass);
    if (result == NULL || result == Py_None) {
        return result;
    }
    if ((size_t)Bytes_GET_SIZE(result) != strlen(Bytes_AS_STRING(result))) {
        if (eclass != NULL) {
            Py_DECREF(result);
            result = pg_EncodeString(obj, NULL, NULL, NULL);
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

static int
pgRWops_IsFileObject(SDL_RWops *rw)
{
    return rw->close == _pg_rw_close;
}

#if IS_SDLv2
static pg_int_t
_pg_rw_size(SDL_RWops *context)
{
    pg_int_t pos, ret;

    /* Current file position; need to restore it later. */
    pos = SDL_RWseek(context, 0, SEEK_CUR);
    if (pos == -1)
        return -1;

    /* Relocate to end of file, get size*/
    ret = SDL_RWseek(context, 0, SEEK_END);

    /* return to original position */
    if (SDL_RWseek(context, pos, SEEK_SET) == -1)
        return -1;

    return ret;
}
#endif /* IS_SDLv2 */

static pg_size_t
_pg_rw_write(SDL_RWops *context, const void *ptr, pg_size_t size, pg_size_t num)
{
    PyObject *result, *fileobj = (PyObject *)context->hidden.unknown.data1;
#ifdef WITH_THREAD
    PyGILState_STATE state = PyGILState_Ensure();
#endif

#if PY3
    result = PyObject_CallMethod(fileobj, "write", "y#", (const char *)ptr,
                                    (Py_ssize_t)(size * num));
#else  /* PY2 */
    result = PyObject_CallMethod(fileobj, "write", "s#", (const char *)ptr,
                                    (Py_ssize_t)(size * num));
#endif  /* PY2 */
    if (!result) {
        PyErr_Print();
        num = 0;
    }
    Py_XDECREF(result);
#ifdef WITH_THREAD
    PyGILState_Release(state);
#endif /* WITH_THREAD */
    return num;
}

static pg_int_t
_pg_rw_seek(SDL_RWops *context, pg_int_t offset, int whence)
{
    pg_int_t retval = -1;
    PyObject *result, *fileobj = (PyObject *)context->hidden.unknown.data1;
#ifdef WITH_THREAD
    PyGILState_STATE state = PyGILState_Ensure();
#endif

    /* Because offset can be a Sint64, cast it to a data type, that
     * can hold all the data */
    result = PyObject_CallMethod(fileobj, "seek", "Li",
                                    (long long)offset, whence);

#if PY2
    /* Some python2 file-like objects have a different seek implementation
     * that does not return the offset, and returns None instead. So use
     * tell method if that happens */
    if (result == Py_None) {
        Py_DECREF(result);
        result = PyObject_CallMethod(fileobj, "tell", NULL);
    }
#endif /* PY2 */
    if (result)
        retval = (pg_int_t)PyLong_AsLong(result);

    if (PyErr_Occurred())
        PyErr_Print();

    Py_XDECREF(result);
#ifdef WITH_THREAD
    PyGILState_Release(state);
#endif
    return retval;
}

static pg_size_t
_pg_rw_read(SDL_RWops *context, void *ptr, pg_size_t size, pg_size_t maxnum)
{
    pg_size_t retval = 0;
    PyObject *result, *fileobj = (PyObject *)context->hidden.unknown.data1;
#ifdef WITH_THREAD
    PyGILState_STATE state = PyGILState_Ensure();
#endif /* WITH_THREAD */

    result = PyObject_CallMethod(fileobj, "read", "k",
                                    (unsigned long)(size * maxnum));
    if (!result) {
        PyErr_Print();
        goto end;
    }

    if (!Bytes_Check(result)) {
        Py_DECREF(result);
        goto end;
    }

    retval = (pg_size_t)Bytes_GET_SIZE(result);
    if (retval) {
        memcpy(ptr, Bytes_AsString(result), retval);
        retval /= size;
    }

    Py_DECREF(result);

end:
#ifdef WITH_THREAD
    PyGILState_Release(state);
#endif /* WITH_THREAD */
    return retval;
}


static int
_pg_rw_close(SDL_RWops *context)
{
    PyObject *result, *fileobj = (PyObject *)context->hidden.unknown.data1;
    int retval = 0;
#ifdef WITH_THREAD
    PyGILState_STATE state = PyGILState_Ensure();
#endif /* WITH_THREAD */

    result = PyObject_CallMethod(fileobj, "close", NULL);
    if (!result) {
        PyErr_Print();
        retval = -1;
    }
    Py_XDECREF(result);
    Py_DECREF(fileobj);

#ifdef WITH_THREAD
    PyGILState_Release(state);
#endif /* WITH_THREAD */
    SDL_FreeRW(context);
    return retval;
}

static SDL_RWops *
pgRWops_FromFileObject(PyObject *obj)
{
    SDL_RWops *rw;

    if (!_is_filelike_obj(obj))
        return (SDL_RWops *)RAISE(PyExc_TypeError, "Invalid filetype object");

    rw = SDL_AllocRW();
    if (!rw)
        return (SDL_RWops *)PyErr_NoMemory();

    Py_INCREF(obj);
    rw->hidden.unknown.data1 = (void *)obj;
#if IS_SDLv2
    rw->type = SDL_RWOPS_UNKNOWN;
    rw->size = _pg_rw_size;
#endif /* IS_SDLv2 */
    rw->seek = _pg_rw_seek;
    rw->read = _pg_rw_read;
    rw->write = _pg_rw_write;
    rw->close = _pg_rw_close;

/* https://docs.python.org/3/c-api/init.html#c.PyEval_InitThreads */
/* ^ in Python >= 3.7, we don't have to call this function, and in 3.11 
 * it will be removed */
#if PY_VERSION_HEX < 0x03070000
#ifdef WITH_THREAD
    PyEval_InitThreads();
#endif /* WITH_THREAD */
#endif
    return rw;
}

static SDL_RWops *
_rwops_from_pystr(PyObject *obj)
{
    if (obj != NULL) {
        SDL_RWops *rw = NULL;
        PyObject *oencoded;
        oencoded = pg_EncodeString(obj, "UTF-8", NULL, NULL);
        if (oencoded == NULL) {
            return NULL;
        }
        if (oencoded != Py_None) {
            rw = SDL_RWFromFile(Bytes_AS_STRING(oencoded), "rb");
        }
        Py_DECREF(oencoded);
        if (rw) {
            return rw;
        } else {
#if PY3
            if (PyUnicode_Check(obj)) {
                SDL_ClearError();
                PyErr_SetString(PyExc_FileNotFoundError,
                                "No such file or directory.");
#else
            if (PyUnicode_Check(obj) || PyString_Check(obj)) {
                SDL_ClearError();
                PyErr_SetString(PyExc_IOError, "No such file or directory.");
#endif
                return NULL;
            }
        }
        SDL_ClearError();
    }
    return NULL;
}

static SDL_RWops *
pgRWops_FromObject(PyObject *obj)
{
    SDL_RWops *rw = _rwops_from_pystr(obj);
    if (!rw) {
        if (PyErr_Occurred())
            return NULL;
    } else {
        return rw;
    }
    return pgRWops_FromFileObject(obj);
}

static PyObject *
pg_encode_string(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *obj = NULL;
    PyObject *eclass = NULL;
    const char *encoding = NULL;
    const char *errors = NULL;
    static char *kwids[] = {"obj", "encoding", "errors", "etype", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|OssO&", kwids, &obj,
                                     &encoding, &errors,
                                     &_pg_is_exception_class, &eclass)) {
        return NULL;
    }

    if (obj == NULL) {
        PyErr_SetString(PyExc_SyntaxError, "Forwarded exception");
    }
    return pg_EncodeString(obj, encoding, errors, eclass);
}

static PyObject *
pg_encode_file_path(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *obj = NULL;
    PyObject *eclass = NULL;
    static char *kwids[] = {"obj", "etype", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|OO&", kwids, &obj,
                                     &_pg_is_exception_class, &eclass)) {
        return NULL;
    }

    if (obj == NULL) {
        PyErr_SetString(PyExc_SyntaxError, "Forwarded exception");
    }
    return pg_EncodeFilePath(obj, eclass);
}

static PyMethodDef _pg_module_methods[] = {
    {"encode_string", (PyCFunction)pg_encode_string,
     METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEENCODESTRING},
    {"encode_file_path", (PyCFunction)pg_encode_file_path,
     METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEENCODEFILEPATH},
    {NULL, NULL, 0, NULL}};

/*DOC*/ static char _pg_module_doc[] =
    /*DOC*/ "SDL_RWops support";

MODINIT_DEFINE(rwobject)
{
    PyObject *module, *dict, *apiobj;
    int ecode;
    static void *c_api[PYGAMEAPI_RWOBJECT_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "rwobject",
                                         _pg_module_doc,
                                         -1,
                                         _pg_module_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif

    /* Create the module and add the functions */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "rwobject", _pg_module_methods,
                            _pg_module_doc);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict(module);

    /* export the c api */
    c_api[0] = pgRWops_FromObject;
    c_api[1] = pgRWops_IsFileObject;
    c_api[2] = pg_EncodeFilePath;
    c_api[3] = pg_EncodeString;
    c_api[4] = pgRWops_FromFileObject;
    apiobj = encapsulate_api(c_api, "rwobject");
    if (apiobj == NULL) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF(apiobj);
    if (ecode == -1) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
