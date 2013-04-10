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
  data throught the Python buffer protocol or the array interface.
  The new buffer protocol is available for Python 3.x. For Python 2.x
  only the old protocol is implemented (for PyPy compatibility).
  Both the C level array structure - __array_struct__ - interface and
  Python level - __array_interface__ - are exposed.
 */

#define PYGAMEAPI_BUFPROXY_INTERNAL
#include "pygame.h"
#include "pgcompat.h"
#include "pgbufferproxy.h"

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define BUFPROXY_MY_ENDIAN '<'
#define BUFPROXY_OTHER_ENDIAN '>'
#else
#define BUFPROXY_MY_ENDIAN '>'
#define BUFPROXY_OTHER_ENDIAN '<'
#endif

#define PROXY_MODNAME "bufferproxy"
#define PROXY_TYPE_NAME "BufferProxy"
#define PROXY_TYPE_FULLNAME (IMPPREFIX PROXY_MODNAME "." PROXY_TYPE_NAME)

typedef struct PgBufproxyObject_s {
    PyObject_HEAD
    PyObject *obj;                             /* Wrapped object              */
    Pg_buffer *view_p;                         /* For array interface export  */
    pg_getbufferfunc get_buffer;               /* Pg_buffer get callback      */
    PyObject *dict;                            /* Allow arbitrary attributes  */
    PyObject *weakrefs;                        /* Reference cycles can happen */
} PgBufproxyObject;

typedef struct Pg_buffer_d_s {
    Pg_buffer view;
    PyObject *dict;
} Pg_buffer_d;

static int PgBufproxy_Trip(PyObject *);

/* $$ Transitional stuff */
#warning Transitional stuff: must disappear!

#define NOTIMPLEMENTED(rcode) \
    PyErr_Format(PyExc_NotImplementedError, \
                 "Not ready yet. (line %i in %s)", __LINE__, __FILE__); \
    return rcode

/* Use Dict_AsView alternative with flags arg. */
static void _release_buffer_from_dict(Py_buffer *);

static int
_get_buffer_from_dict(PyObject *dict, Pg_buffer *pg_view_p, int flags) {
    PyObject *obj;
    Py_buffer *view_p = (Py_buffer *)pg_view_p;
    Pg_buffer *pg_dict_view_p;
    Py_buffer *dict_view_p;
    PyObject *py_callback;
    PyObject *py_rval;

    assert(dict && PyDict_Check(dict));
    assert(view_p);
    view_p->obj = 0;
    pg_dict_view_p = PyMem_New(Pg_buffer, 1);
    if (!pg_dict_view_p) {
        PyErr_NoMemory();
        return -1;
    }
    if (PgDict_AsBuffer(pg_dict_view_p, dict, flags)) {
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
        Py_INCREF(py_callback);
        py_rval = PyObject_CallFunctionObjArgs(py_callback, obj, NULL);
        Py_DECREF(py_callback);
        if (!py_rval) {
            PgBuffer_Release(pg_dict_view_p);
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
    pg_view_p->release_buffer = _release_buffer_from_dict;
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
    PyObject *py_rval;

    assert(view_p && view_p->internal);
    obj = view_p->obj;
    dict_view_p = (Py_buffer *)view_p->internal;
    dict = dict_view_p->obj;
    assert(dict && PyDict_Check(dict));
    py_callback = PyDict_GetItemString(dict, "after");
    if (py_callback) {
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
    PgBuffer_Release((Pg_buffer *)dict_view_p);
    PyMem_Free(dict_view_p);
    view_p->obj = 0;
    Py_DECREF(obj);
}

/* Stub functions */
static PyObject *proxy_get_raw(PgBufproxyObject *, PyObject *);

/* End transitional stuff */

static PyObject *
_proxy_subtype_new(PyTypeObject *type,
                   PyObject *obj,
                   pg_getbufferfunc get_buffer)
{
    PgBufproxyObject *self = (PgBufproxyObject *)type->tp_alloc(type, 0);

    if (!self) {
        return 0;
    }
    Py_XINCREF(obj);
    self->obj = obj;
    self->get_buffer = get_buffer;
    return (PyObject *)self;
}

static Py_buffer *
_proxy_get_view(PgBufproxyObject *proxy) {
    Pg_buffer *view_p = proxy->view_p;

    if (!view_p) {
        view_p = PyMem_New(Pg_buffer, 1);
        if (!view_p) {
            PyErr_NoMemory();
        }
        else if (proxy->get_buffer(proxy->obj, view_p, PyBUF_RECORDS)) {
            PyMem_Free(view_p);
            view_p = 0;
        }
        else {
            proxy->view_p = view_p;
        }
    }
    return (Py_buffer *)view_p;
}

static void
_proxy_release_view(PgBufproxyObject *proxy) {
    Pg_buffer *view_p = proxy->view_p;

    if (view_p) {
        proxy->view_p = 0;
        PgBuffer_Release(view_p);
        PyMem_Free(view_p);
    }
}

/**
 * Return a new PgBufproxyObject (Python level constructor).
 */
static PyObject *
proxy_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *obj = 0;
    pg_getbufferfunc get_buffer = PgObject_GetBuffer;

    if (!PyArg_ParseTuple(args, "O:Bufproxy", &obj)) {
        return 0;
    }
    if (PyDict_Check(obj)) {
        get_buffer = _get_buffer_from_dict;
    }
    return _proxy_subtype_new(type, obj, get_buffer);
}

/**
 * Deallocates the PgBufproxyObject and its members.
 */
static void
proxy_dealloc(PgBufproxyObject *self)
{
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
proxy_traverse(PgBufproxyObject *self, visitproc visit, void *arg) {
    if (self->obj) {
        Py_VISIT(self->obj);
    }
    if (self->view_p && ((Py_buffer *)self->view_p)->obj) /* conditional && */ {
        Py_VISIT(((Py_buffer *)self->view_p)->obj);
    }
    if (self->dict) {
        Py_VISIT(self->dict);
    }
    return 0;
}

/**** Getter and setter access ****/
static PyObject *
proxy_get_arraystruct(PgBufproxyObject *self, PyObject *closure)
{
    Py_buffer *view_p = _proxy_get_view(self);
    PyObject *capsule;

    if (!view_p) {
        return 0;
    }
    capsule = PgBuffer_AsArrayStruct(view_p);
    if (!capsule) {
        _proxy_release_view(self);
    }
    return capsule;
}

static PyObject *
proxy_get_arrayinterface(PgBufproxyObject *self, PyObject *closure)
{
    Py_buffer *view_p = _proxy_get_view(self);
    PyObject *dict;

    if (!view_p) {
        return 0;
    }
    dict = PgBuffer_AsArrayInterface(view_p);
    if (!dict) {
        _proxy_release_view(self);
    }
    return dict;
}

static PyObject *
proxy_get_parent(PgBufproxyObject *self, PyObject *closure)
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
proxy_get___dict__(PgBufproxyObject *self, PyObject *closure)
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
proxy_get_raw(PgBufproxyObject *self, PyObject *closure)
{
    Py_buffer *view_p = _proxy_get_view(self);

    if (!view_p) {
        return 0;
    }
    if (!PyBuffer_IsContiguous(view_p, 'A')) {
        PyErr_SetString(PyExc_ValueError, "the bytes are not contiguous");
        return 0;
    }
    return Bytes_FromStringAndSize((char *)view_p->buf, view_p->len);
}

static PyObject *
proxy_get_length(PgBufproxyObject *self, PyObject *closure)
{
    Py_buffer *view_p = _proxy_get_view(self);

    return view_p ? PyInt_FromSsize_t(view_p->len) : 0;
}

/**** Methods ****/

/**
 * Representation method.
 */
static PyObject *
proxy_repr (PgBufproxyObject *self)
{
    return Text_FromFormat("<%s(%p)>", Py_TYPE(self)->tp_name, self);
}

/**
 * Writes raw data to the buffer.
 */
static PyObject *
proxy_write(PgBufproxyObject *buffer, PyObject *args, PyObject *kwds)
{
    NOTIMPLEMENTED(0);
}

static struct PyMethodDef proxy_methods[] = {
    {"write", (PyCFunction)proxy_write, METH_VARARGS | METH_KEYWORDS,
     "write raw bytes to object buffer"},
    {0, 0, 0, 0}
};

/**
 * Getters and setters for the PgBufproxyObject.
 */
static PyGetSetDef proxy_getsets[] =
{
    {"__array_struct__", (getter)proxy_get_arraystruct, 0, 0, 0},
    {"__array_interface__", (getter)proxy_get_arrayinterface, 0, 0, 0},
    {"parent", (getter)proxy_get_parent, 0, 0, 0},
    {"__dict__", (getter)proxy_get___dict__, 0, 0, 0},
    {"raw", (getter)proxy_get_raw, 0, 0, 0},
    {"length", (getter)proxy_get_length, 0, 0, 0},
    {0, 0, 0, 0, 0}
};

#if PG_ENABLE_NEWBUF
static int
proxy_getbuffer(PgBufproxyObject *self, Py_buffer *view_p, int flags)
{
    Pg_buffer *pg_obj_view_p = PyMem_New(Pg_buffer, 1);
    Py_buffer *obj_view_p = (Py_buffer *)pg_obj_view_p;

    view_p->obj = 0;
    if (!pg_obj_view_p) {
        PyErr_NoMemory();
        return -1;
    }
    if (self->get_buffer(self->obj, pg_obj_view_p, flags)) {
        PyMem_Free(pg_obj_view_p);
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
proxy_releasebuffer(PgBufproxyObject *self, Py_buffer *view_p)
{
    PgBuffer_Release((Pg_buffer *)view_p->internal);
    PyMem_Free(view_p->internal);
}

static PyBufferProcs proxy_bufferprocs = {
    (getbufferproc)proxy_getbuffer,
    (releasebufferproc)proxy_releasebuffer
};

#else

static Py_ssize_t
proxy_getreadbuf(PgBufproxyObject *buffer, Py_ssize_t _index, const void **ptr)
{
    NOTIMPLEMENTED(-1);
}

static Py_ssize_t
proxy_getwritebuf(PgBufproxyObject *buffer, Py_ssize_t _index, const void **ptr)
{
    NOTIMPLEMENTED(-1);
}

static Py_ssize_t
proxy_getsegcount(PgBufproxyObject *buffer, Py_ssize_t *lenp)
{
    NOTIMPLEMENTED(-1);
}

static PyBufferProcs proxy_bufferprocs = {
    (readbufferproc)proxy_getreadbuf,
    (writebufferproc)proxy_getwritebuf,
    (segcountproc)proxy_getsegcount,
    0
#if PY_VERSION_HEX >= 0x02060000
    ,
    0,
    0
#endif
};

#endif /* #if PG_ENABLE_NEWBUF */

#define PROXY_TPFLAGS \
    (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC)

static PyTypeObject PgBufproxy_Type =
{
    TYPE_HEAD(NULL, 0)
    PROXY_TYPE_FULLNAME,        /* tp_name */
    sizeof (PgBufproxyObject),  /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor)proxy_dealloc,  /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc)proxy_repr,       /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    &proxy_bufferprocs,         /* tp_as_buffer */
    PROXY_TPFLAGS,              /* tp_flags */
    "Object bufproxy as an array struct\n",
    (traverseproc)proxy_traverse,  /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    offsetof(PgBufproxyObject, weakrefs),  /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    proxy_methods,              /* tp_methods */
    0,                          /* tp_members */
    proxy_getsets,              /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    offsetof(PgBufproxyObject, dict),  /* tp_dictoffset */
    0,                          /* tp_init */
    PyType_GenericAlloc,        /* tp_alloc */
    proxy_new,                  /* tp_new */
    PyObject_GC_Del,            /* tp_free */
#ifndef __SYMBIAN32__
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0                           /* tp_del */
#endif
};

/**** Module methods ***/

static PyMethodDef bufferproxy_methods[] = {
    {0, 0, 0, 0}
};

/**** Public C api ***/

static PyObject *
PgBufproxy_New(PyObject *obj, pg_getbufferfunc get_buffer)
{
    if (!get_buffer) {
        if (!obj) {
            PyErr_SetString(PyExc_ValueError,
                            "One of arguments 'obj' or 'get_buffer' is "
                            "required: both NULL instead");
            return 0;
        }
        get_buffer = PgObject_GetBuffer;
    }
    return _proxy_subtype_new(&PgBufproxy_Type, obj, get_buffer);
}

static PyObject *
PgBufproxy_GetParent(PyObject *obj)
{
    if (!PyObject_IsInstance (obj, (PyObject *)&PgBufproxy_Type)) {
        PyErr_Format(PyExc_TypeError,
                     "Expected a BufferProxy object: got %s instance instead",
                     Py_TYPE(obj)->tp_name);
        return 0;
    }
    return proxy_get_parent((PgBufproxyObject *)obj, 0);
}

static int
PgBufproxy_Trip(PyObject *obj)
{
    if (!PyObject_IsInstance (obj, (PyObject *)&PgBufproxy_Type)) {
        PyErr_Format(PyExc_TypeError,
                     "Expected a BufferProxy object: got %s instance instead",
                     Py_TYPE(obj)->tp_name);
        return -1;
    }
    return _proxy_get_view((PgBufproxyObject *)obj) ? 0 : -1;
}

/*DOC*/ static char bufferproxy_doc[] =
/*DOC*/    "exports BufferProxy, a generic wrapper object for an py_buffer";

MODINIT_DEFINE(bufferproxy)
{
    PyObject *module;
    PyObject *apiobj;
    static void* c_api[PYGAMEAPI_BUFPROXY_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        PROXY_MODNAME,
        bufferproxy_doc,
        -1,
        bufferproxy_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    /* imported needed apis */
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* prepare exported types */
    if (PyType_Ready(&PgBufproxy_Type) < 0) {
        MODINIT_ERROR;
    }

#define bufferproxy_docs ""

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX PROXY_MODNAME, bufferproxy_methods,
                            bufferproxy_doc);
#endif

    Py_INCREF(&PgBufproxy_Type);
    if (PyModule_AddObject(module,
                           PROXY_TYPE_NAME,
                           (PyObject *)&PgBufproxy_Type)) {
        Py_DECREF(&PgBufproxy_Type);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
#if PYGAMEAPI_BUFPROXY_NUMSLOTS != 4
#error export slot count mismatch
#endif
    c_api[0] = &PgBufproxy_Type;
    c_api[1] = PgBufproxy_New;
    c_api[2] = PgBufproxy_GetParent;
    c_api[3] = PgBufproxy_Trip;
    apiobj = encapsulate_api(c_api, PROXY_MODNAME);
    if (apiobj == NULL) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_DECREF(apiobj);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
