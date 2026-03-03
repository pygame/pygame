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
} pgRWHelper;

/*static const char pg_default_encoding[] = "unicode_escape";*/
/*static const char pg_default_errors[] = "backslashreplace";*/
static const char pg_default_encoding[] = "unicode_escape";
static const char pg_default_errors[] = "backslashreplace";

static PyObject *os_module = NULL;

#if SDL_VERSION_ATLEAST(3, 0, 0)
static Sint64
_pg_rw_size(void *);
static Sint64
_pg_rw_seek(void *, Sint64, SDL_IOWhence);
static size_t
_pg_rw_read(void *, void *, size_t, SDL_IOStatus *);
static size_t
_pg_rw_write(void *, const void *, size_t, SDL_IOStatus *);
static bool
_pg_rw_close(void *);
#else
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
#endif

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
    PyObject *tmp;

    if (!PyType_Check(obj) || /* conditional or */
        !PyObject_IsSubclass(obj, PyExc_BaseException)) {
        oname = PyObject_Str(obj);
        if (oname == NULL) {
            PyErr_SetString(PyExc_TypeError,
                            "invalid exception class argument");
            return 0;
        }
        tmp = PyUnicode_AsEncodedString(oname, "ascii", "replace");
        Py_DECREF(oname);

        if (tmp == NULL) {
            PyErr_SetString(PyExc_TypeError,
                            "invalid exception class argument");
            return 0;
        }

        oname = tmp;
        PyErr_Format(PyExc_TypeError,
                     "Expected an exception class: got %.1024s",
                     PyBytes_AS_STRING(oname));
        Py_DECREF(oname);
        return 0;
    }
    *rval = obj;
    return 1;
}

static int
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
    if (!helper->read && !helper->write) {
        PyErr_SetString(PyExc_TypeError, "not a file object");
        return -1;
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
    return 0;
}

/* This function is meant to decode a pathlib object into its str/bytes
 * representation. */
static PyObject *
_trydecode_pathlibobj(PyObject *obj)
{
    PyObject *ret = PyOS_FSPath(obj);
    if (!ret) {
        /* A valid object was not passed. But we do not consider it an error */
        PyErr_Clear();
        Py_INCREF(obj);
        return obj;
    }
    return ret;
}

static PyObject *
pg_EncodeString(PyObject *obj, const char *encoding, const char *errors,
                PyObject *eclass)
{
    PyObject *oencoded, *exc_type, *exc_value, *exc_trace, *str, *ret;

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

    ret = _trydecode_pathlibobj(obj);
    if (!ret) {
        return NULL;
    }

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
            /* Forward as eclass error */
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

    if (PyBytes_Check(ret)) {
        return ret;
    }

    Py_DECREF(ret);
    Py_RETURN_NONE;
}

static PyObject *
pg_EncodeFilePath(PyObject *obj, PyObject *eclass)
{
    /* All of this code is a replacement for Py_FileSystemDefaultEncoding,
     * which is deprecated in Python 3.12
     *
     * But I'm not sure of the use of this function, so maybe it should be
     * deprecated. */

    PyObject *sys_module = PyImport_ImportModule("sys");
    if (sys_module == NULL) {
        return NULL;
    }
    PyObject *system_encoding_obj =
        PyObject_CallMethod(sys_module, "getfilesystemencoding", NULL);
    if (system_encoding_obj == NULL) {
        Py_DECREF(sys_module);
        return NULL;
    }
    Py_DECREF(sys_module);
    const char *encoding = PyUnicode_AsUTF8(system_encoding_obj);
    if (encoding == NULL) {
        Py_DECREF(system_encoding_obj);
        return NULL;
    }

    /* End code replacement section */

    if (obj == NULL) {
        PyErr_SetString(PyExc_SyntaxError, "Forwarded exception");
    }

    PyObject *result =
        pg_EncodeString(obj, encoding, UNICODE_DEF_FS_ERROR, eclass);
    Py_DECREF(system_encoding_obj);

    if (result == NULL || result == Py_None) {
        return result;
    }
    if ((size_t)PyBytes_GET_SIZE(result) !=
        strlen(PyBytes_AS_STRING(result))) {
        if (eclass != NULL) {
            Py_DECREF(result);
            result = pg_EncodeString(obj, NULL, NULL, NULL);
            if (result == NULL) {
                return NULL;
            }
            PyErr_Format(eclass,
                         "File path '%.1024s' contains null characters",
                         PyBytes_AS_STRING(result));
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
#if SDL_VERSION_ATLEAST(3, 0, 0)
    SDL_PropertiesID props = SDL_GetIOProperties(rw);
    if (!props) {
        // pgRWops_IsFileObject doesn't have any error checking facility
        // so when in doubt let's say it isn't a file object.
        return 0;
    }
    return SDL_GetBooleanProperty(props, "_pygame_is_file_object", 0);
#else
    return rw->close == _pg_rw_close;
#endif
}

#if SDL_VERSION_ATLEAST(3, 0, 0)
static Sint64
_pg_rw_size(void *userdata)
{
    pgRWHelper *helper = (pgRWHelper *)userdata;
#else
static Sint64
_pg_rw_size(SDL_RWops *context)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
#endif

    PyObject *pos = NULL;
    PyObject *tmp = NULL;
    Sint64 size;
    Sint64 retval = -1;

    if (!helper->seek || !helper->tell) {
        return retval;
    }

    PyGILState_STATE state = PyGILState_Ensure();

    /* Current file position; need to restore it later.
     */
    pos = PyObject_CallNoArgs(helper->tell);
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
    tmp = PyObject_CallNoArgs(helper->tell);
    if (!tmp) {
        PyErr_Print();
        goto end;
    }

    size = PyLong_AsLongLong(tmp);
    if (size == -1 && PyErr_Occurred() != NULL) {
        PyErr_Print();
        goto end;
    }
    Py_DECREF(tmp);

    /* Return to original position.
     */
    tmp = PyObject_CallOneArg(helper->seek, pos);
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

#if SDL_VERSION_ATLEAST(3, 0, 0)
static size_t
_pg_rw_write(void *userdata, const void *ptr, size_t size,
             SDL_IOStatus *status)
{
    pgRWHelper *helper = (pgRWHelper *)userdata;
    size_t num = 1;
#else
static size_t
_pg_rw_write(SDL_RWops *context, const void *ptr, size_t size, size_t num)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
#endif
    PyObject *result;
    size_t retval;

    if (!helper->write) {
        return -1;
    }

    PyGILState_STATE state = PyGILState_Ensure();

    result = PyObject_CallFunction(helper->write, "y#", (const char *)ptr,
                                   (Py_ssize_t)size * num);
    if (!result) {
        PyErr_Print();
        retval = -1;
        goto end;
    }

    Py_DECREF(result);
#if SDL_VERSION_ATLEAST(3, 0, 0)
    retval = size;
#else
    retval = num;
#endif

end:
    PyGILState_Release(state);
    return retval;
}

#if SDL_VERSION_ATLEAST(3, 0, 0)
static bool
_pg_rw_close(void *userdata)
{
    pgRWHelper *helper = (pgRWHelper *)userdata;
    bool retval = true;
#else
static int
_pg_rw_close(SDL_RWops *context)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
    int retval = 0;
#endif
    PyObject *result;
    PyGILState_STATE state = PyGILState_Ensure();

    if (helper->close) {
        result = PyObject_CallNoArgs(helper->close);
        if (!result) {
            PyErr_Print();
#if SDL_VERSION_ATLEAST(3, 0, 0)
            retval = false;
#else
            retval = -1;
#endif
        }
        Py_XDECREF(result);
    }

    Py_XDECREF(helper->seek);
    Py_XDECREF(helper->tell);
    Py_XDECREF(helper->write);
    Py_XDECREF(helper->read);
    Py_XDECREF(helper->close);

    PyMem_Free(helper);
    PyGILState_Release(state);
#if !SDL_VERSION_ATLEAST(3, 0, 0)
    SDL_FreeRW(context);
#endif
    return retval;
}

static SDL_RWops *
pgRWops_FromFileObject(PyObject *obj)
{
    SDL_RWops *rw;
    pgRWHelper *helper;

    if (obj == NULL) {
        return (SDL_RWops *)RAISE(PyExc_TypeError, "Invalid filetype object");
    }

    helper = PyMem_New(pgRWHelper, 1);
    if (helper == NULL) {
        return (SDL_RWops *)PyErr_NoMemory();
    }
    if (fetch_object_methods(helper, obj)) {
        PyMem_Free(helper);
        return NULL;
    }

#if SDL_VERSION_ATLEAST(3, 0, 0)
    SDL_IOStreamInterface iface;
    SDL_INIT_INTERFACE(&iface);
    iface.size = _pg_rw_size;
    iface.seek = _pg_rw_seek;
    iface.read = _pg_rw_read;
    iface.write = _pg_rw_write;
    iface.close = _pg_rw_close;

    // TODO: These should raise SDLError probably?
    // rwobject.c hasn't required pygame.base before (the source of SDLError)
    // so omitting that for now.

    rw = SDL_OpenIO(&iface, helper);
    if (rw == NULL) {
        iface.close(helper);
        return (SDL_RWops *)RAISE(PyExc_IOError, SDL_GetError());
    }

    SDL_PropertiesID props = SDL_GetIOProperties(rw);
    if (!props) {
        iface.close(helper);
        return (SDL_RWops *)RAISE(PyExc_IOError, SDL_GetError());
    }

    if (!SDL_SetBooleanProperty(props, "_pygame_is_file_object", 1)) {
        iface.close(helper);
        return (SDL_RWops *)RAISE(PyExc_IOError, SDL_GetError());
    }

#else
    rw = SDL_AllocRW();
    if (rw == NULL) {
        /* TODO: here member attributes of helper are getting leaked and close
         * isn't being called. Refactor to use _pg_rw_close logic on helper
         * here */
        PyMem_Free(helper);
        return (SDL_RWops *)PyErr_NoMemory();
    }

    /* Adding a helper to the hidden data to support file-like object RWops */
    rw->hidden.unknown.data1 = (void *)helper;
    rw->size = _pg_rw_size;
    rw->seek = _pg_rw_seek;
    rw->read = _pg_rw_read;
    rw->write = _pg_rw_write;
    rw->close = _pg_rw_close;
#endif

    return rw;
}

#if SDL_VERSION_ATLEAST(3, 0, 0)
static Sint64
_pg_rw_seek(void *userdata, Sint64 offset, SDL_IOWhence whence)
{
    pgRWHelper *helper = (pgRWHelper *)userdata;
#else
static Sint64
_pg_rw_seek(SDL_RWops *context, Sint64 offset, int whence)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
#endif
    PyObject *result;
    Sint64 retval;

    if (!helper->seek || !helper->tell) {
        return -1;
    }

    PyGILState_STATE state = PyGILState_Ensure();

    if (!(offset == 0 &&
          whence == SEEK_CUR)) /* being seek'd, not just tell'd */
    {
        result = PyObject_CallFunction(helper->seek, "Li", (long long)offset,
                                       whence);
        if (!result) {
            PyErr_Print();
            retval = -1;
            goto end;
        }
        Py_DECREF(result);
    }

    result = PyObject_CallNoArgs(helper->tell);
    if (!result) {
        PyErr_Print();
        retval = -1;
        goto end;
    }

    retval = PyLong_AsLongLong(result);
    if (retval == -1 && PyErr_Occurred()) {
        PyErr_Clear();
    }

    Py_DECREF(result);

end:
    PyGILState_Release(state);

    return retval;
}

#if SDL_VERSION_ATLEAST(3, 0, 0)
static size_t
_pg_rw_read(void *userdata, void *ptr, size_t size, SDL_IOStatus *status)
{
    pgRWHelper *helper = (pgRWHelper *)userdata;
    size_t maxnum = 1;
#else
static size_t
_pg_rw_read(SDL_RWops *context, void *ptr, size_t size, size_t maxnum)
{
    pgRWHelper *helper = (pgRWHelper *)context->hidden.unknown.data1;
#endif
    PyObject *result;
    Py_ssize_t retval;

    if (!helper->read) {
        return -1;
    }

    PyGILState_STATE state = PyGILState_Ensure();
    result = PyObject_CallFunction(helper->read, "K",
                                   (unsigned long long)size * maxnum);
    if (!result) {
        PyErr_Print();
        retval = -1;
        goto end;
    }

    if (!PyBytes_Check(result)) {
        Py_DECREF(result);
        PyErr_Print();
        retval = -1;
        goto end;
    }

    retval = PyBytes_GET_SIZE(result);
    if (retval) {
        memcpy(ptr, PyBytes_AsString(result), retval);
#if !SDL_VERSION_ATLEAST(3, 0, 0)
        retval /= size;
#endif
    }

    Py_DECREF(result);

end:
    PyGILState_Release(state);

    return retval;
}

static SDL_RWops *
_rwops_from_pystr(PyObject *obj, char **extptr)
{
    SDL_RWops *rw = NULL;
    PyObject *oencoded;
    char *encoded = NULL;

    /* If a valid extptr has been passed, we want it to default to NULL
     * to show that an extension hasn't been procured (if it has it will
     * get set to that later) */
    if (extptr) {
        *extptr = NULL;
    }

    if (!obj) {
        // forward any errors
        return NULL;
    }

    oencoded = pg_EncodeString(obj, "UTF-8", NULL, NULL);
    if (!oencoded || oencoded == Py_None) {
        /* if oencoded is NULL, we are forwarding an error. If it is None, the
         * object passed was not a bytes/string/pathlib object so handling of
         * that is done after this function, exit early here */
        Py_XDECREF(oencoded);
        return NULL;
    }

    encoded = PyBytes_AS_STRING(oencoded);
    rw = SDL_RWFromFile(encoded, "rb");

    if (rw) {
        /* If a valid extptr has been passed, populate it with a dynamically
         * allocated field for the file extension. */
        if (extptr) {
            char *ext = strrchr(encoded, '.');
            if (ext && strlen(ext) > 1) {
                ext++;
                *extptr = malloc(strlen(ext) + 1);
                if (!(*extptr)) {
                    /* If out of memory, decref oencoded to be safe, and try
                     * to close out `rw` as well. */
                    Py_DECREF(oencoded);
#if SDL_VERSION_ATLEAST(3, 0, 0)
                    if (!SDL_RWclose(rw)) {
#else
                    if (SDL_RWclose(rw) < 0) {
#endif
                        PyErr_SetString(PyExc_IOError, SDL_GetError());
                    }
                    return (SDL_RWops *)PyErr_NoMemory();
                }
                strcpy(*extptr, ext);
            }
        }

        Py_DECREF(oencoded);
        return rw;
    }

    Py_DECREF(oencoded);
    /* Clear SDL error and set our own error message for filenotfound errors
     * TODO: Check SDL error here and forward any non filenotfound related
     * errors correctly here */
    SDL_ClearError();

    PyObject *cwd = NULL, *path = NULL, *isabs = NULL;
    if (!os_module) {
        goto simple_case;
    }

    cwd = PyObject_CallMethod(os_module, "getcwd", NULL);
    if (!cwd) {
        goto simple_case;
    }

    path = PyObject_GetAttrString(os_module, "path");
    if (!path) {
        goto simple_case;
    }

    isabs = PyObject_CallMethod(path, "isabs", "O", obj);
    Py_DECREF(path);
    if (!isabs || isabs == Py_True) {
        goto simple_case;
    }

    PyErr_Format(PyExc_FileNotFoundError,
                 "No file '%S' found in working directory '%S'.", obj, cwd);

    Py_DECREF(cwd);
    Py_DECREF(isabs);
    return NULL;

simple_case:
    Py_XDECREF(cwd);
    Py_XDECREF(isabs);
    PyErr_Format(PyExc_FileNotFoundError, "No such file or directory: '%S'.",
                 obj);
    return NULL;
}

static SDL_RWops *
pgRWops_FromObject(PyObject *obj, char **extptr)
{
#if __EMSCRIPTEN__
    SDL_RWops *rw;
    int retry = 0;
again:
    rw = _rwops_from_pystr(obj, extptr);
    if (retry) {
        Py_XDECREF(obj);
    }
    if (!rw) {
        if (PyErr_Occurred()) {
            return NULL;
        }
    }
    else {
        return rw;
    }

fail:
    if (retry) {
        return RAISE(PyExc_RuntimeError, "can't access resource on platform");
    }

    retry = 1;
    PyObject *name = PyObject_GetAttrString(obj, "name");
    if (name) {
        obj = name;
        goto again;
    }
    goto fail;
    // unreachable.
#else
    SDL_RWops *rw = _rwops_from_pystr(obj, extptr);
    if (!rw) {
        if (PyErr_Occurred()) {
            return NULL;
        }
    }
    else {
        return rw;
    }
    return pgRWops_FromFileObject(obj);
#endif
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

    return pg_EncodeFilePath(obj, eclass);
}

static PyMethodDef _pg_rwobject_methods[] = {
    {"encode_string", (PyCFunction)pg_encode_string,
     METH_VARARGS | METH_KEYWORDS, DOC_ENCODESTRING},
    {"encode_file_path", (PyCFunction)pg_encode_file_path,
     METH_VARARGS | METH_KEYWORDS, DOC_ENCODEFILEPATH},
    {NULL, NULL, 0, NULL}};

/*DOC*/ static char _pg_rwobject_doc[] =
    /*DOC*/ "SDL_RWops support";

MODINIT_DEFINE(rwobject)
{
    PyObject *module, *apiobj;
    static void *c_api[PYGAMEAPI_RWOBJECT_NUMSLOTS];

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "rwobject",
                                         _pg_rwobject_doc,
                                         -1,
                                         _pg_rwobject_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* Create the module and add the functions */
    module = PyModule_Create(&_module);
    if (module == NULL) {
        return NULL;
    }

    /* export the c api */
    c_api[0] = pgRWops_FromObject;
    c_api[1] = pgRWops_IsFileObject;
    c_api[2] = pg_EncodeFilePath;
    c_api[3] = pg_EncodeString;
    c_api[4] = pgRWops_FromFileObject;
    apiobj = encapsulate_api(c_api, "rwobject");
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_XDECREF(apiobj);
        Py_DECREF(module);
        return NULL;
    }

    /* import os, don't sweat if it errors, it will be checked before use */
    os_module = PyImport_ImportModule("os");
    if (os_module == NULL) {
        PyErr_Clear();
    }

    return module;
}
