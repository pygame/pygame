/*
  pygame - Python Game Library
  Module adapted from bufferproxy.c, Copyright (C) 2007  Marcus von Appen

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

/*
  This module exports a proxy object that exposes another object's
  data through the Python buffer protocol or the array interface.
  The new buffer protocol is available for Python 3.x. For Python 2.x
  only the old protocol is implemented (for PyPy compatibility).
  Both the C level array structure - __array_struct__ - interface and
  Python level - __array_interface__ - are exposed.
 */

#define PY_SSIZE_T_CLEAN
#define PYGAMEAPI_BUFPROXY_INTERNAL
#include "pygame.h"

#include "pgcompat.h"
#include "pgbufferproxy.h"
#include "doc/bufferproxy_doc.h"

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define BUFPROXY_MY_ENDIAN '<'
#define BUFPROXY_OTHER_ENDIAN '>'
#else
#define BUFPROXY_MY_ENDIAN '>'
#define BUFPROXY_OTHER_ENDIAN '<'
#endif

#define PROXY_MODNAME "bufferproxy"
#define PROXY_TYPE_NAME "BufferProxy"

#ifdef NDEBUG
#define PyBUF_PG_VIEW PyBUF_RECORDS
#define PyBUF_PG_VIEW_RO PyBUF_RECORDS_RO
#else
#define PyBUF_PG_VIEW (PyBUF_RECORDS | PyBUF_PYGAME)
#define PyBUF_PG_VIEW_RO (PyBUF_RECORDS_RO | PyBUF_PYGAME)
#endif

typedef struct pgBufproxyObject_s {
    PyObject_HEAD PyObject *obj; /* Wrapped object (parent)     */
    pg_buffer *pg_view_p;        /* For array interface export  */
    getbufferproc get_buffer;    /* pg_buffer get callback      */
    PyObject *dict;              /* Allow arbitrary attributes  */
    PyObject *weakrefs;          /* Reference cycles can happen */
} pgBufproxyObject;

static int
pgBufproxy_Trip(PyObject *);
static Py_buffer *
_proxy_get_view(pgBufproxyObject *);
static int
proxy_getbuffer(pgBufproxyObject *, Py_buffer *, int);
static void
proxy_releasebuffer(pgBufproxyObject *, Py_buffer *);

static void
_release_buffer_from_dict(Py_buffer *);

static int
_get_buffer_from_dict(PyObject *dict, Py_buffer *view_p, int flags)
{
    PyObject *obj;
    pg_buffer *pg_dict_view_p;
    Py_buffer *dict_view_p;
    PyObject *py_callback;

    assert(dict && PyDict_Check(dict));
    assert(view_p);
    view_p->obj = 0;
    pg_dict_view_p = PyMem_New(pg_buffer, 1);
    if (!pg_dict_view_p) {
        PyErr_NoMemory();
        return -1;
    }
    pg_dict_view_p->consumer = ((pg_buffer *)view_p)->consumer;
    if (pgDict_AsBuffer(pg_dict_view_p, dict, flags)) {
        PyMem_Free(pg_dict_view_p);
        return -1;
    }
    dict_view_p = (Py_buffer *)pg_dict_view_p;
    obj = PyDict_GetItemString(dict, "parent");
    if (!obj) {
        obj = Py_None;
    }
    Py_INCREF(obj);
    py_callback = PyDict_GetItemString(dict, "before");

    if (py_callback) {
        PyObject *py_rval;

        Py_INCREF(py_callback);
        py_rval = PyObject_CallFunctionObjArgs(py_callback, obj, NULL);
        Py_DECREF(py_callback);
        if (!py_rval) {
            pgBuffer_Release(pg_dict_view_p);
            Py_DECREF(obj);
            return -1;
        }
        Py_DECREF(py_rval);
    }
    Py_INCREF(dict);
    dict_view_p->obj = dict;
    view_p->obj = obj;
    view_p->buf = dict_view_p->buf;
    view_p->len = dict_view_p->len;
    view_p->readonly = dict_view_p->readonly;
    view_p->itemsize = dict_view_p->itemsize;
    view_p->format = dict_view_p->format;
    view_p->ndim = dict_view_p->ndim;
    view_p->shape = dict_view_p->shape;
    view_p->strides = dict_view_p->strides;
    view_p->suboffsets = dict_view_p->suboffsets;
    view_p->internal = pg_dict_view_p;
    ((pg_buffer *)view_p)->release_buffer = _release_buffer_from_dict;
    return 0;
}

/* This will need changes */
static void
_release_buffer_from_dict(Py_buffer *view_p)
{
    Py_buffer *dict_view_p;
    PyObject *dict;
    PyObject *obj;
    PyObject *py_callback;

    assert(view_p && view_p->internal);
    obj = view_p->obj;
    dict_view_p = (Py_buffer *)view_p->internal;
    dict = dict_view_p->obj;
    assert(dict && PyDict_Check(dict));
    py_callback = PyDict_GetItemString(dict, "after");

    if (py_callback) {
        PyObject *py_rval;

        Py_INCREF(py_callback);
        py_rval = PyObject_CallFunctionObjArgs(py_callback, obj, NULL);
        if (py_rval) {
            Py_DECREF(py_rval);
        }
        else {
            PyErr_Clear();
        }
        Py_DECREF(py_callback);
    }
    pgBuffer_Release((pg_buffer *)dict_view_p);
    PyMem_Free(dict_view_p);
    view_p->obj = 0;
    Py_DECREF(obj);
}

/* Stub functions */
static PyObject *
proxy_get_raw(pgBufproxyObject *, PyObject *);

/* End transitional stuff */

static PyObject *
_proxy_subtype_new(PyTypeObject *type, PyObject *obj, getbufferproc get_buffer)
{
    pgBufproxyObject *self = (pgBufproxyObject *)type->tp_alloc(type, 0);

    if (!self) {
        return 0;
    }
    Py_XINCREF(obj);
    self->obj = obj;
    self->get_buffer = get_buffer;
    return (PyObject *)self;
}

static Py_buffer *
_proxy_get_view(pgBufproxyObject *proxy)
{
    pg_buffer *pg_view_p = proxy->pg_view_p;

    if (!pg_view_p) {
        pg_view_p = PyMem_New(pg_buffer, 1);
        if (!pg_view_p) {
            PyErr_NoMemory();
            return 0;
        }
        pg_view_p->consumer = (PyObject *)proxy;
        if (proxy->get_buffer(proxy->obj, (Py_buffer *)pg_view_p,
                              PyBUF_PG_VIEW_RO)) {
            PyMem_Free(pg_view_p);
            return 0;
        }
        proxy->pg_view_p = pg_view_p;
    }
    assert(((Py_buffer *)pg_view_p)->len &&
           ((Py_buffer *)pg_view_p)->itemsize);
    return (Py_buffer *)pg_view_p;
}

static void
_proxy_release_view(pgBufproxyObject *proxy)
{
    pg_buffer *pg_view_p = proxy->pg_view_p;

    if (pg_view_p) {
        proxy->pg_view_p = 0;
        pgBuffer_Release(pg_view_p);
        PyMem_Free(pg_view_p);
    }
}

static int
_proxy_zombie_get_buffer(PyObject *obj, Py_buffer *view_p, int flags)
{
    PyObject *proxy = ((pg_buffer *)view_p)->consumer;

    view_p->obj = 0;
    PyErr_Format(PyExc_RuntimeError,
                 "Attempted buffer export on <%s at %p, parent=<%s at %p>> "
                 "while deallocating it",
                 Py_TYPE(proxy)->tp_name, (void *)proxy, Py_TYPE(obj)->tp_name,
                 (void *)obj);
    return -1;
}

/**
 * Return a new pgBufproxyObject (Python level constructor).
 */
static PyObject *
proxy_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    getbufferproc get_buffer = (getbufferproc)pgObject_GetBuffer;

    if (!PyArg_ParseTuple(args, "O:Bufproxy", &obj)) {
        return 0;
    }
    if (PyDict_Check(obj)) {
        get_buffer = _get_buffer_from_dict;
    }
    return _proxy_subtype_new(type, obj, get_buffer);
}

/**
 * Deallocates the pgBufproxyObject and its members.
 * Is reentrant.
 */
static void
proxy_dealloc(pgBufproxyObject *self)
{
    /* Prevent infinite recursion from a reentrant call */
    if (self->get_buffer == _proxy_zombie_get_buffer) {
        return;
    }
    self->get_buffer = _proxy_zombie_get_buffer;

    /* Non reentrant call; deallocate */
    PyObject_GC_UnTrack(self);
    _proxy_release_view(self);
    Py_XDECREF(self->obj);
    Py_XDECREF(self->dict);
    if (self->weakrefs) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }
    Py_TYPE(self)->tp_free(self);
}

static int
proxy_traverse(pgBufproxyObject *self, visitproc visit, void *arg)
{
    if (self->obj) {
        Py_VISIT(self->obj);
    }
    if (self->pg_view_p && /* conditional && */
        ((Py_buffer *)self->pg_view_p)->obj) {
        Py_VISIT(((Py_buffer *)self->pg_view_p)->obj);
    }
    if (self->dict) {
        Py_VISIT(self->dict);
    }
    return 0;
}

/**** Getter and setter access ****/
static PyObject *
proxy_get_arraystruct(pgBufproxyObject *self, PyObject *closure)
{
    Py_buffer *view_p = _proxy_get_view(self);
    PyObject *capsule;

    if (!view_p) {
        return 0;
    }
    capsule = pgBuffer_AsArrayStruct(view_p);
    if (!capsule) {
        _proxy_release_view(self);
    }
    return capsule;
}

static PyObject *
proxy_get_arrayinterface(pgBufproxyObject *self, PyObject *closure)
{
    Py_buffer *view_p = _proxy_get_view(self);
    PyObject *dict;

    if (!view_p) {
        return 0;
    }
    dict = pgBuffer_AsArrayInterface(view_p);
    if (!dict) {
        _proxy_release_view(self);
    }
    return dict;
}

static PyObject *
proxy_get_parent(pgBufproxyObject *self, PyObject *closure)
{
    Py_buffer *view_p = _proxy_get_view(self);
    PyObject *obj;

    if (!view_p) {
        return 0;
    }
    obj = view_p->obj ? view_p->obj : Py_None;
    Py_INCREF(obj);
    return obj;
}

static PyObject *
proxy_get___dict__(pgBufproxyObject *self, PyObject *closure)
{
    if (!self->dict) {
        self->dict = PyDict_New();
        if (!self->dict) {
            return 0;
        }
    }
    Py_INCREF(self->dict);
    return self->dict;
}

static PyObject *
proxy_get_raw(pgBufproxyObject *self, PyObject *closure)
{
    Py_buffer *view_p = _proxy_get_view(self);
    PyObject *py_raw = 0;

    if (!view_p) {
        return 0;
    }
    if (!PyBuffer_IsContiguous(view_p, 'A')) {
        _proxy_release_view(self);
        PyErr_SetString(PyExc_ValueError, "the bytes are not contiguous");
        return 0;
    }
    py_raw = PyBytes_FromStringAndSize((char *)view_p->buf, view_p->len);
    if (!py_raw) {
        _proxy_release_view(self);
        return 0;
    }
    return py_raw;
}

static PyObject *
proxy_get_length(pgBufproxyObject *self, PyObject *closure)
{
    Py_buffer *view_p = _proxy_get_view(self);
    PyObject *py_length = 0;

    if (view_p) {
        py_length = PyLong_FromSsize_t(view_p->len);
        if (!py_length) {
            _proxy_release_view(self);
        }
    }
    return py_length;
}

/**** Methods ****/

/**
 * Representation method.
 */
static PyObject *
proxy_repr(pgBufproxyObject *self)
{
    Py_buffer *view_p = _proxy_get_view(self);

    if (!view_p) {
        return 0;
    }
    return PyUnicode_FromFormat("<BufferProxy(%zd)>", view_p->len);
}

/**
 * Writes raw data to the buffer.
 */
static PyObject *
proxy_write(pgBufproxyObject *self, PyObject *args, PyObject *kwds)
{
    Py_buffer view;
    const char *buf = 0;
    Py_ssize_t buflen = 0;
    Py_ssize_t offset = 0;
    char *keywords[] = {"buffer", "offset", 0};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s#|n", keywords, &buf,
                                     &buflen, &offset)) {
        return 0;
    }

    if (proxy_getbuffer(self, &view, PyBUF_PG_VIEW)) {
        return 0;
    }
    if (!PyBuffer_IsContiguous(&view, 'A')) {
        proxy_releasebuffer(self, &view);
        Py_DECREF(self);
        PyErr_SetString(PyExc_ValueError,
                        "the BufferProxy bytes are not contiguous");
        return 0;
    }
    if (buflen > view.len) {
        proxy_releasebuffer(self, &view);
        Py_DECREF(self);
        PyErr_SetString(PyExc_ValueError,
                        "'buffer' object length is too large");
        return 0;
    }
    if (offset < 0 || buflen + offset > view.len) {
        proxy_releasebuffer(self, &view);
        Py_DECREF(self);
        PyErr_SetString(PyExc_IndexError, "'offset' is out of range");
        return 0;
    }
    memcpy((char *)view.buf + offset, buf, (size_t)buflen);
    proxy_releasebuffer(self, &view);
    Py_DECREF(self);
    Py_RETURN_NONE;
}

static struct PyMethodDef proxy_methods[] = {
    {"write", (PyCFunction)proxy_write, METH_VARARGS | METH_KEYWORDS,
     DOC_BUFFERPROXYWRITE},
    {0, 0, 0, 0}};

/**
 * Getters and setters for the pgBufproxyObject.
 */
static PyGetSetDef proxy_getsets[] = {
    {"__array_struct__", (getter)proxy_get_arraystruct, 0,
     "Version 3 array interface, C level", 0},
    {"__array_interface__", (getter)proxy_get_arrayinterface, 0,
     "Version 3 array interface, Python level", 0},
    {"parent", (getter)proxy_get_parent, 0, DOC_BUFFERPROXYPARENT, 0},
    {"__dict__", (getter)proxy_get___dict__, 0,
     "The object's attribute dictionary, read-only", 0},
    {"raw", (getter)proxy_get_raw, 0, DOC_BUFFERPROXYRAW, 0},
    {"length", (getter)proxy_get_length, 0, DOC_BUFFERPROXYLENGTH, 0},
    {0, 0, 0, 0, 0}};

static int
proxy_getbuffer(pgBufproxyObject *self, Py_buffer *view_p, int flags)
{
    Py_buffer *obj_view_p = PyMem_Malloc(sizeof(pg_buffer));

#ifndef NDEBUG
    flags |= PyBUF_PYGAME;
#endif
    view_p->obj = 0;
    if (!obj_view_p) {
        PyErr_NoMemory();
        return -1;
    }
    ((pg_buffer *)obj_view_p)->consumer = (PyObject *)self;
    if (self->get_buffer(self->obj, obj_view_p, flags)) {
        PyMem_Free(obj_view_p);
        return -1;
    }
    Py_INCREF(self);
    view_p->obj = (PyObject *)self;
    view_p->buf = obj_view_p->buf;
    view_p->len = obj_view_p->len;
    view_p->readonly = obj_view_p->readonly;
    view_p->itemsize = obj_view_p->itemsize;
    view_p->format = obj_view_p->format;
    view_p->ndim = obj_view_p->ndim;
    view_p->shape = obj_view_p->shape;
    view_p->strides = obj_view_p->strides;
    view_p->suboffsets = obj_view_p->suboffsets;
    view_p->internal = obj_view_p;
    return 0;
}

static void
proxy_releasebuffer(pgBufproxyObject *self, Py_buffer *view_p)
{
    pgBuffer_Release((pg_buffer *)view_p->internal);
    PyMem_Free(view_p->internal);
}

#define PROXY_BUFFERPROCS (&proxy_bufferprocs)

static PyBufferProcs proxy_bufferprocs = {
    (getbufferproc)proxy_getbuffer, (releasebufferproc)proxy_releasebuffer};

#define PROXY_TPFLAGS \
    (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC)

static PyTypeObject pgBufproxy_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.bufferproxy.BufferProxy",
    .tp_basicsize = sizeof(pgBufproxyObject),
    .tp_dealloc = (destructor)proxy_dealloc,
    .tp_repr = (reprfunc)proxy_repr,
    .tp_as_buffer = PROXY_BUFFERPROCS,
    .tp_flags = PROXY_TPFLAGS,
    .tp_doc = DOC_PYGAMEBUFFERPROXY,
    .tp_traverse = (traverseproc)proxy_traverse,
    .tp_weaklistoffset = offsetof(pgBufproxyObject, weakrefs),
    .tp_methods = proxy_methods,
    .tp_getset = proxy_getsets,
    .tp_dictoffset = offsetof(pgBufproxyObject, dict),
    .tp_alloc = PyType_GenericAlloc,
    .tp_new = proxy_new,
    .tp_free = PyObject_GC_Del,
};

/**** Module methods ***/

static PyMethodDef bufferproxy_methods[] = {{0, 0, 0, 0}};

/**** Public C api ***/

static PyObject *
pgBufproxy_New(PyObject *obj, getbufferproc get_buffer)
{
    if (!get_buffer) {
        if (!obj) {
            PyErr_SetString(PyExc_ValueError,
                            "One of arguments 'obj' or 'get_buffer' is "
                            "required: both NULL instead");
            return 0;
        }
        get_buffer = (getbufferproc)pgObject_GetBuffer;
    }
    return _proxy_subtype_new(&pgBufproxy_Type, obj, get_buffer);
}

static PyObject *
pgBufproxy_GetParent(PyObject *obj)
{
    if (!PyObject_IsInstance(obj, (PyObject *)&pgBufproxy_Type)) {
        PyErr_Format(PyExc_TypeError,
                     "Expected a BufferProxy object: got %s instance instead",
                     Py_TYPE(obj)->tp_name);
        return 0;
    }
    return proxy_get_parent((pgBufproxyObject *)obj, 0);
}

static int
pgBufproxy_Trip(PyObject *obj)
{
    if (!PyObject_IsInstance(obj, (PyObject *)&pgBufproxy_Type)) {
        PyErr_Format(PyExc_TypeError,
                     "Expected a BufferProxy object: got %s instance instead",
                     Py_TYPE(obj)->tp_name);
        return -1;
    }
    return _proxy_get_view((pgBufproxyObject *)obj) ? 0 : -1;
}

/*DOC*/ static char bufferproxy_doc[] = DOC_PYGAMEBUFFERPROXY;

MODINIT_DEFINE(bufferproxy)
{
    PyObject *module, *apiobj;
    static void *c_api[PYGAMEAPI_BUFPROXY_NUMSLOTS];

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         PROXY_MODNAME,
                                         bufferproxy_doc,
                                         -1,
                                         bufferproxy_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* imported needed apis */
    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }

    /* prepare exported types */
    if (PyType_Ready(&pgBufproxy_Type) < 0) {
        return NULL;
    }

#define bufferproxy_docs ""

    /* create the module */
    module = PyModule_Create(&_module);
    if (!module) {
        return NULL;
    }

    Py_INCREF(&pgBufproxy_Type);
    if (PyModule_AddObject(module, PROXY_TYPE_NAME,
                           (PyObject *)&pgBufproxy_Type)) {
        Py_DECREF(&pgBufproxy_Type);
        Py_DECREF(module);
        return NULL;
    }
#if PYGAMEAPI_BUFPROXY_NUMSLOTS != 4
#error export slot count mismatch
#endif
    c_api[0] = &pgBufproxy_Type;
    c_api[1] = pgBufproxy_New;
    c_api[2] = pgBufproxy_GetParent;
    c_api[3] = pgBufproxy_Trip;
    apiobj = encapsulate_api(c_api, PROXY_MODNAME);
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_XDECREF(apiobj);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
