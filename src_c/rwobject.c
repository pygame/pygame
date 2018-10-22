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

typedef struct {
    PyObject *read;
    PyObject *write;
    PyObject *seek;
    PyObject *tell;
    PyObject *close;
    int fileno;
} pgRWHelper;

/*static const char pg_default_encoding[] = "unicode_escape";*/
/*static const char pg_default_errors[] = "backslashreplace";*/
static const char pg_default_encoding[] = "unicode_escape";
static const char pg_default_errors[] = "backslashreplace";

#if IS_SDLv1
static int
_pg_rw_seek(SDL_RWops *, int, int);
static int
_pg_rw_read(SDL_RWops *, void *, int, int);
static int
_pg_rw_write(SDL_RWops *, const void *, int, int);
static int
_pg_rw_close(SDL_RWops *);

#ifdef WITH_THREAD
static int
_pg_rw_seek_th(SDL_RWops *, int, int);
static int
_pg_rw_read_th(SDL_RWops *, void *, int, int);
static int
_pg_rw_write_th(SDL_RWops *, const void *, int, int);
static int
_pg_rw_close_th(SDL_RWops *);
#endif
#else /* IS_SDLv2 */
static Sint64
_pg_rw_size(SDL_RWops *);
static Sint64
_pg_rw_seek(SDL_RWops *, Sint64, int);
static size_t
_pg_rw_read(SDL_RWops *, void *, size_t, size_t);
static size_t
_pg_rw_write(SDL_RWops *, const void *, size_t, size_t);
static int
_pg_rw_close(SDL_RWops *);

#ifdef WITH_THREAD
static Sint64
_pg_rw_size_th(SDL_RWops *);
static Sint64
_pg_rw_seek_th(SDL_RWops *, Sint64, int);
static size_t
_pg_rw_read_th(SDL_RWops *, void *, size_t, size_t);
static size_t
_pg_rw_write_th(SDL_RWops *, const void *, size_t, size_t);
static int
_pg_rw_close_th(SDL_RWops *);
#endif
#endif /* IS_SDLv2 */

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
fetch_object_methods(pgRWHelper *helper, PyObject *obj)
{
    helper->read = helper->write = helper->seek = helper->tell =
        helper->close = NULL;

    if (PyObject_HasAttrString(obj, "read")) {
        helper->read = PyObject_GetAttrString(obj, "read");
        if (helper->read && !PyCallable_Check(helper->read)) {
            Py_DECREF(helper->read);
            helper->read = NULL;
        }
    }
    if (PyObject_HasAttrString(obj, "write")) {
        helper->write = PyObject_GetAttrString(obj, "write");
        if (helper->write && !PyCallable_Check(helper->write)) {
            Py_DECREF(helper->write);
            helper->write = NULL;
        }
    }
    if (PyObject_HasAttrString(obj, "seek")) {
        helper->seek = PyObject_GetAttrString(obj, "seek");
        if (helper->seek && !PyCallable_Check(helper->seek)) {
            Py_DECREF(helper->seek);
            helper->seek = NULL;
        }
    }
    if (PyObject_HasAttrString(obj, "tell")) {
        helper->tell = PyObject_GetAttrString(obj, "tell");
        if (helper->tell && !PyCallable_Check(helper->tell)) {
            Py_DECREF(helper->tell);
            helper->tell = NULL;
        }
    }
    if (PyObject_HasAttrString(obj, "close")) {
        helper->close = PyObject_GetAttrString(obj, "close");
        if (helper->close && !PyCallable_Check(helper->close)) {
            Py_DECREF(helper->close);
            helper->close = NULL;
        }
    }
}

static PyObject *
pgRWopsEncodeString(PyObject *obj, const char *encoding, const char *errors,
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
        encoding = pg_default_encoding;
    }
    if (errors == NULL) {
        errors = pg_default_errors;
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
                         "Pygame bug (in pgRWopsEncodeString):"
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

static PyObject *
pgRWopsEncodeFilePath(PyObject *obj, PyObject *eclass)
{
    PyObject *result = pgRWopsEncodeString(obj, UNICODE_DEF_FS_CODEC,
                                           UNICODE_DEF_FS_ERROR, eclass);
    if (result == NULL || result == Py_None) {
        return result;
    }
    if ((size_t)Bytes_GET_SIZE(result) != strlen(Bytes_AS_STRING(result))) {
        if (eclass != NULL) {
            Py_DECREF(result);
            result = pgRWopsEncodeString(obj, NULL, NULL, NULL);
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

static SDL_RWops *
pgRWopsFromFileObject(PyObject *obj)
{
    SDL_RWops *rw = NULL;
    pgRWHelper *helper;

    if (obj == NULL) {
        return (SDL_RWops *)RAISE(PyExc_TypeError, "Invalid filetype object");
    }
    helper = PyMem_New(pgRWHelper, 1);
    if (helper == NULL) {
        return (SDL_RWops *)PyErr_NoMemory();
    }
    rw = SDL_AllocRW();
    if (rw == NULL) {
        PyMem_Del(helper);
        return (SDL_RWops *)PyErr_NoMemory();
    }
    helper->fileno = PyObject_AsFileDescriptor(obj);
    if (helper->fileno == -1)
        PyErr_Clear();
    fetch_object_methods(helper, obj);
    rw->hidden.unknown.data1 = (void *)helper;
#if IS_SDLv2
    rw->size = _pg_rw_size;
#endif /* IS_SDLv2 */
    rw->seek = _pg_rw_seek;
    rw->read = _pg_rw_read;
    rw->write = _pg_rw_write;
    rw->close = _pg_rw_close;

    return rw;
}

static SDL_RWops *
pgRWopsFromObject(PyObject *obj)
{
    PyObject *oencoded;
    SDL_RWops *rw = NULL;

    if (obj != NULL) {
        oencoded = pgRWopsEncodeFilePath(obj, NULL);
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
    return pgRWopsFromFileObject(obj);
}

static int
pgRWopsCheckObject(SDL_RWops *rw)
{
    return rw->close == _pg_rw_close;
}

#if IS_SDLv2
static Sint64
_pg_rw_size(SDL_RWops *context)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *pos = NULL;
    PyObject *tmp = NULL;
    Sint64 size;
    Sint64 retval = -1;

    if (!helper->seek || !helper->tell)
        return retval;

    /* Current file position; need to restore it later.
     */
    pos = PyObject_CallFunction(helper->tell, NULL);
    if (!pos) {
        PyErr_Print();
        goto end;
    }

    /* Relocate to end of file.
     */
    tmp = PyObject_CallFunction(helper->seek, "ii", 0, SEEK_END);
    if (!tmp) {
        PyErr_Print();
        goto end;
    }
    Py_DECREF(tmp);

    /* Record file size.
     */
    tmp = PyObject_CallFunction(helper->tell, NULL);
    if (!tmp) {
        PyErr_Print();
        goto end;
    }
    size = PyInt_AsLong(tmp);
    if (size == -1 && PyErr_Occurred() != NULL) {
        PyErr_Print();
        goto end;
    }
    Py_DECREF(tmp);

    /* Return to original position.
     */
    tmp = PyObject_CallFunctionObjArgs(helper->seek, pos, NULL);
    if (!tmp) {
        PyErr_Print();
        goto end;
    }

    /* Success.
     */
    retval = size;

end:
    /* Cleanup.
     */
    Py_XDECREF(pos);
    Py_XDECREF(tmp);
    return retval;
}
#endif /* IS_SDLv2 */

#if IS_SDLv1
static int
_pg_rw_seek(SDL_RWops *context, int offset, int whence)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *result;
    int retval;
#else /* IS_SDLv2 */
static Sint64
_pg_rw_seek(SDL_RWops *context, Sint64 offset, int whence)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *result;
    Sint64 retval;
#endif

    if (helper->fileno != -1) {
        return lseek(helper->fileno, offset, whence);
    }

    if (!helper->seek || !helper->tell)
        return -1;

    if (!(offset == 0 && whence == SEEK_CUR)) /*being called only for 'tell'*/
    {
        result = PyObject_CallFunction(helper->seek, "ii", offset, whence);
        if (!result)
            return -1;
        Py_DECREF(result);
    }

    result = PyObject_CallFunction(helper->tell, NULL);
    if (!result)
        return -1;

    retval = PyInt_AsLong(result);
    Py_DECREF(result);

    return retval;
}

#if IS_SDLv1
static int
_pg_rw_read(SDL_RWops *context, void *ptr, int size, int maxnum)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *result;
    int retval;
#else  /* IS_SDLv2 */
static size_t
_pg_rw_read(SDL_RWops *context, void *ptr, size_t size, size_t maxnum)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *result;
    size_t retval;
#endif /* IS_SDLv2 */

    if (helper->fileno != -1) {
        retval = read(helper->fileno, ptr, size * maxnum);
        if (retval == -1) {
            return -1;
        }
        retval /= size;
        return retval;
    }

    if (!helper->read)
        return -1;

    result = PyObject_CallFunction(helper->read, "i", size * maxnum);
    if (!result)
        return -1;

    if (!Bytes_Check(result)) {
        Py_DECREF(result);
        return -1;
    }

    retval = Bytes_GET_SIZE(result);
    memcpy(ptr, Bytes_AsString(result), retval);
    retval /= size;

    Py_DECREF(result);
    return retval;
}

#if IS_SDLv1
static int
_pg_rw_write(SDL_RWops *context, const void *ptr, int size, int num)
#else  /* IS_SDLv2 */
static size_t
_pg_rw_write(SDL_RWops *context, const void *ptr, size_t size, size_t num)
#endif /* IS_SDLv2 */
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *result;

    if (!helper->write)
        return -1;

    result = PyObject_CallFunction(helper->write, "s#", ptr, size * num);
    if (!result)
        return -1;

    Py_DECREF(result);
    return num;
}

static int
_pg_rw_close(SDL_RWops *context)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *result;
    int retval = 0;

    if (helper->close) {
        result = PyObject_CallFunction(helper->close, NULL);
        if (result)
            retval = -1;
        Py_XDECREF(result);
    }

    Py_XDECREF(helper->seek);
    Py_XDECREF(helper->tell);
    Py_XDECREF(helper->write);
    Py_XDECREF(helper->read);
    Py_XDECREF(helper->close);
    PyMem_Del(helper);
    SDL_FreeRW(context);
    return retval;
}

static SDL_RWops *
pgRWopsFromFileObjectThreaded(PyObject *obj)
{
    SDL_RWops *rw;
    pgRWHelper *helper;

    if (obj == NULL) {
        return (SDL_RWops *)RAISE(PyExc_TypeError, "Invalid filetype object");
    }

#ifndef WITH_THREAD
    return (SDL_RWops *)RAISE(PyExc_NotImplementedError,
                              "Python built without thread support");
#else
    helper = PyMem_New(pgRWHelper, 1);
    if (helper == NULL) {
        return (SDL_RWops *)PyErr_NoMemory();
    }
    rw = SDL_AllocRW();
    if (rw == NULL) {
        PyMem_Del(helper);
        return (SDL_RWops *)PyErr_NoMemory();
    }
    helper->fileno = PyObject_AsFileDescriptor(obj);
    if (helper->fileno == -1)
        PyErr_Clear();
    fetch_object_methods(helper, obj);
    rw->hidden.unknown.data1 = (void *)helper;
#if IS_SDLv2
    rw->size = _pg_rw_size_th;
#endif /* IS_SDLv2 */
    rw->seek = _pg_rw_seek_th;
    rw->read = _pg_rw_read_th;
    rw->write = _pg_rw_write_th;
    rw->close = _pg_rw_close_th;

    PyEval_InitThreads();

    return rw;
#endif
}

static int
pgRWopsCheckObjectThreaded(SDL_RWops *rw)
{
#ifdef WITH_THREAD
    return rw->close == _pg_rw_close_th;
#else
    return 0;
#endif
}

static int
pgRWopsFreeFromObject(SDL_RWops *context)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *result;
    int retval = 0;

    Py_XDECREF(helper->seek);
    Py_XDECREF(helper->tell);
    Py_XDECREF(helper->write);
    Py_XDECREF(helper->read);
    Py_XDECREF(helper->close);
    PyMem_Del(helper);
    SDL_FreeRW(context);
    return retval;
}

#ifdef WITH_THREAD
#if IS_SDLv2
static Sint64
_pg_rw_size_th(SDL_RWops *context)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *pos = NULL;
    PyObject *tmp = NULL;
    Sint64 size;
    Sint64 retval = -1;
    PyGILState_STATE state;

    if (!helper->seek || !helper->tell)
        return retval;

    state = PyGILState_Ensure();

    /* Current file position; need to restore it later.
     */
    pos = PyObject_CallFunction(helper->tell, NULL);
    if (!pos) {
        PyErr_Print();
        goto end;
    }

    /* Relocate to end of file.
     */
    tmp = PyObject_CallFunction(helper->seek, "ii", 0, SEEK_END);
    if (!tmp) {
        PyErr_Print();
        goto end;
    }
    Py_DECREF(tmp);

    /* Record file size.
     */
    tmp = PyObject_CallFunction(helper->tell, NULL);
    if (!tmp) {
        PyErr_Print();
        goto end;
    }
    size = PyInt_AsLong(tmp);
    if (size == -1 && PyErr_Occurred() != NULL) {
        PyErr_Print();
        goto end;
    }
    Py_DECREF(tmp);

    /* Return to original position.
     */
    tmp = PyObject_CallFunctionObjArgs(helper->seek, pos, NULL);
    if (!tmp) {
        PyErr_Print();
        goto end;
    }

    /* Success.
     */
    retval = size;

end:
    /* Cleanup.
     */
    Py_XDECREF(pos);
    Py_XDECREF(tmp);
    PyGILState_Release(state);
    return retval;
}
#endif /* IS_SDLv2 */

#if IS_SDLv1
static int
_pg_rw_seek_th(SDL_RWops *context, int offset, int whence)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *result;
    int retval;
#else  /* IS_SDLv2 */
static Sint64
_pg_rw_seek_th(SDL_RWops *context, Sint64 offset, int whence)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *result;
    Sint64 retval;
#endif /* IS_SDLv2 */
    PyGILState_STATE state;

    if (helper->fileno != -1) {
        return lseek(helper->fileno, offset, whence);
    }

    if (!helper->seek || !helper->tell)
        return -1;

    state = PyGILState_Ensure();

    if (!(offset == 0 &&
          whence == SEEK_CUR)) /* being seek'd, not just tell'd */
    {
        result = PyObject_CallFunction(helper->seek, "ii", offset, whence);
        if (!result) {
            PyErr_Print();
            retval = -1;
            goto end;
        }
        Py_DECREF(result);
    }

    result = PyObject_CallFunction(helper->tell, NULL);
    if (!result) {
        PyErr_Print();
        retval = -1;
        goto end;
    }

    retval = PyInt_AsLong(result);
    Py_DECREF(result);

end:
    PyGILState_Release(state);

    return retval;
}

#if IS_SDLv1
static int
_pg_rw_read_th(SDL_RWops *context, void *ptr, int size, int maxnum)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *result;
    int retval;
#else  /* IS_SDLv2 */
static size_t
_pg_rw_read_th(SDL_RWops *context, void *ptr, size_t size, size_t maxnum)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *result;
    size_t retval;
#endif /* IS_SDLv2 */
    PyGILState_STATE state;

    if (helper->fileno != -1) {
        retval = read(helper->fileno, ptr, size * maxnum);
        if (retval == -1) {
            return -1;
        }
        retval /= size;
        return retval;
    }

    if (!helper->read)
        return -1;

    state = PyGILState_Ensure();
    result = PyObject_CallFunction(helper->read, "i", size * maxnum);
    if (!result) {
        PyErr_Print();
        retval = -1;
        goto end;
    }

    if (!Bytes_Check(result)) {
        Py_DECREF(result);
        PyErr_Print();
        retval = -1;
        goto end;
    }

    retval = Bytes_GET_SIZE(result);
    memcpy(ptr, Bytes_AsString(result), retval);
    retval /= size;

    Py_DECREF(result);

end:
    PyGILState_Release(state);

    return retval;
}

#if IS_SDLv1
static int
_pg_rw_write_th(SDL_RWops *context, const void *ptr, int size, int num)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *result;
    int retval;
#else  /* IS_SDLv2 */
static size_t
_pg_rw_write_th(SDL_RWops *context, const void *ptr, size_t size, size_t num)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *result;
    size_t retval;
#endif /* IS_SDLv2 */
    PyGILState_STATE state;

    if (!helper->write)
        return -1;

    state = PyGILState_Ensure();

    result = PyObject_CallFunction(helper->write, "s#", ptr, size * num);
    if (!result) {
        PyErr_Print();
        retval = -1;
        goto end;
    }

    Py_DECREF(result);
    retval = num;

end:
    PyGILState_Release(state);

    return retval;
}

static int
_pg_rw_close_th(SDL_RWops *context)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    PyObject *result;
    int retval = 0;
    PyGILState_STATE state;

    state = PyGILState_Ensure();

    if (helper->close) {
        result = PyObject_CallFunction(helper->close, NULL);
        if (!result) {
            PyErr_Print();
            retval = -1;
        }
        Py_XDECREF(result);
    }

    Py_XDECREF(helper->seek);
    Py_XDECREF(helper->tell);
    Py_XDECREF(helper->write);
    Py_XDECREF(helper->read);
    Py_XDECREF(helper->close);

    PyMem_Del(helper);

    PyGILState_Release(state);

    SDL_FreeRW(context);
    return retval;
}
#endif

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
        RAISE(PyExc_SyntaxError, "Forwarded exception");
    }
    return pgRWopsEncodeString(obj, encoding, errors, eclass);
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
        RAISE(PyExc_SyntaxError, "Forwarded exception");
    }
    return pgRWopsEncodeFilePath(obj, eclass);
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
    c_api[0] = pgRWopsFromObject;
    c_api[1] = pgRWopsCheckObject;
    c_api[2] = pgRWopsFromFileObjectThreaded;
    c_api[3] = pgRWopsCheckObjectThreaded;
    c_api[4] = pgRWopsEncodeFilePath;
    c_api[5] = pgRWopsEncodeString;
    c_api[6] = pgRWopsFromFileObject;
    c_api[7] = pgRWopsFreeFromObject;
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
