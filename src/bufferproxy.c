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
    Py_buffer *view_p;                         /* For array interface export  */
    PgBufproxy_CallbackGet get_buffer;         /* Py_buffer get callback      */
    PgBufproxy_CallbackRelease release_buffer; /* Py_buffer release callback  */
    PyObject *dict;                            /* Allow arbitrary attributes  */
    PyObject *weakrefs;                        /* Reference cycles can happen */
} PgBufproxyObject;

static int PgBufproxy_Trip(PyObject *);
static void _free_view(Py_buffer *);

/* $$ Transitional stuff */
#warning Transitional stuff: must disappear!

#define NOTIMPLEMENTED(rcode) \
    PyErr_Format(PyExc_NotImplementedError, \
                 "Not ready yet. (line %i in %s)", __LINE__, __FILE__); \
    return rcode


#if PY_VERSION_HEX < 0x02060000
static int
_IsFortranContiguous(Py_buffer *view)
{
    Py_ssize_t sd, dim;
    int i;

    if (view->ndim == 0) return 1;
    if (view->strides == NULL) return (view->ndim == 1);

    sd = view->itemsize;
    if (view->ndim == 1) return (view->shape[0] == 1 ||
                               sd == view->strides[0]);
    for (i=0; i<view->ndim; i++) {
        dim = view->shape[i];
        if (dim == 0) return 1;
        if (view->strides[i] != sd) return 0;
        sd *= dim;
    }
    return 1;
}

static int
_IsCContiguous(Py_buffer *view)
{
    Py_ssize_t sd, dim;
    int i;

    if (view->ndim == 0) return 1;
    if (view->strides == NULL) return 1;

    sd = view->itemsize;
    if (view->ndim == 1) return (view->shape[0] == 1 ||
                               sd == view->strides[0]);
    for (i=view->ndim-1; i>=0; i--) {
        dim = view->shape[i];
        if (dim == 0) return 1;
        if (view->strides[i] != sd) return 0;
        sd *= dim;
    }
    return 1;
}

int
PyBuffer_IsContiguous(Py_buffer *view, char fort)
{

    if (view->suboffsets != NULL) return 0;

    if (fort == 'C')
        return _IsCContiguous(view);
    else if (fort == 'F')
        return _IsFortranContiguous(view);
    else if (fort == 'A')
        return (_IsCContiguous(view) || _IsFortranContiguous(view));
    return 0;
}
#endif /* #if PY_VERSION_HEX < 0x02060000 */

/* Stub for base.c function */
static int
PgObject_GetBuffer(PyObject *obj, Py_buffer *view_p, int flags)
{
#if PY3
    if (PyObject_CheckBuffer(obj)) {
        return PyObject_GetBuffer(obj, view_p, flags);
    }
#endif
    PyErr_SetString(PyExc_NotImplementedError, "TODO: implement in base.c");
    return -1;
}

/* Stub for base.c function */
static PyObject *
PgBuffer_AsArrayStruct(Py_buffer *view_p) {
    int flags = 0;

    if (PyBuffer_IsContiguous(view_p, 'C')) {
        flags |= VIEW_CONTIGUOUS | VIEW_C_ORDER;
    }
    if (PyBuffer_IsContiguous(view_p, 'F')) {
        flags |= VIEW_CONTIGUOUS | VIEW_F_ORDER;
    }
    return ViewAndFlagsAsArrayStruct(view_p, flags);
}

/* Stub for base.c function */
static PyObject *
PgBuffer_AsArrayInterface(Py_buffer *view_p) {
    return ViewAsDict(view_p);
}

/* Stub for base.c function */
static void
PgBuffer_Release(Py_buffer *view_p)
{
#if PY3
    if (view_p && view_p->obj && PyObject_CheckBuffer(view_p->obj)) {
        PyBuffer_Release(view_p);
    }
#endif
    printf("How did I get here? (line %i in %s)\n", __LINE__, __FILE__);
}

/* Use Dict_AsView alternative with flags arg. */
#warning This is a major hack that needs to be removed when base.c is updated.
static PyObject *_hack_01 = 0;

static int
_get_buffer_from_dict(PyObject *dict, Py_buffer *view_p, int flags) {
    PyObject *obj;
    Py_buffer *dict_view_p;
    PyObject *py_callback;
    PyObject *py_rval;

    assert(dict && PyDict_Check(dict));
    assert(view_p);
    view_p->obj = 0;
    dict_view_p = PyMem_New(Py_buffer, 1);
    if (!dict_view_p) {
        PyErr_NoMemory();
        return -1;
    }
    if (Dict_AsView(dict_view_p, dict)/* $$ Change here */) {
        PyMem_Free(dict_view_p);
        return -1;
    }
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
            ReleaseView(dict_view_p);
            Py_DECREF(obj);
            return -1;
        }
        Py_DECREF(py_rval);
    }
    Py_INCREF(dict);
    _hack_01 = dict_view_p->obj;  /* $$ hack */
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
    view_p->internal = dict_view_p;
    return 0;
}

/* This will need changes, including replacing ReleaseView call. */
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
    assert(dict_view_p);
    dict = dict_view_p->obj;
    assert(PyDict_Check(dict));
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
/*    dict_view_p->obj = 0; */
    dict_view_p->obj = _hack_01;  /* $$ hack */
    ReleaseView(dict_view_p);
    Py_DECREF(dict);
    PyMem_Free(dict_view_p);
    view_p->obj = 0;
    Py_DECREF(obj);
}

/* Backward compatible stub: replace me */
static int
mok_PgBufproxy_New(Py_buffer *view,
                   int flags,
                   PgBufproxy_CallbackBefore before,
                   PgBufproxy_CallbackAfter after)
{
    NOTIMPLEMENTED(-1);
}

/* Stub functions */
static PyObject *proxy_get_raw(PgBufproxyObject *, PyObject *);

/* End transitional stuff */

static PyObject *
_proxy_subtype_new(PyTypeObject *type,
                   PyObject *obj,
                   PgBufproxy_CallbackGet get_buffer,
                   PgBufproxy_CallbackRelease release_buffer)
{
    PgBufproxyObject *self = (PgBufproxyObject *)type->tp_alloc(type, 0);

    if (!self) {
        return 0;
    }
    Py_XINCREF(obj);
    self->obj = obj;
    self->get_buffer = get_buffer;
    self->release_buffer = release_buffer;
    return (PyObject *)self;
}

static Py_buffer *
_proxy_get_view(PgBufproxyObject *proxy) {
    Py_buffer *view_p = proxy->view_p;

    if (!view_p) {
        view_p = PyMem_New(Py_buffer, 1);
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
    return view_p;
}

static int
_tuple_as_ints(PyObject *o,
               const char *keyword,
               Py_ssize_t **array,
               int *length)
{
    /* Convert o as a C array of integers and return 1, otherwise
     * raise a Python exception and return 0. keyword is the function
     * argument name to use in the exception.
     */
    Py_ssize_t i, n;
    Py_ssize_t *a;

    if (!PyTuple_Check(o)) {
        PyErr_Format(PyExc_TypeError,
                     "Expected a tuple for argument %s: found %s",
                     keyword, Py_TYPE(o)->tp_name);
        return 0;
    }
    n = PyTuple_GET_SIZE(o);
    a = PyMem_New(Py_intptr_t, n);
    for (i = 0; i < n; ++i) {
        a[i] = PyInt_AsSsize_t(PyTuple_GET_ITEM(o, i));
        if (a[i] == -1 && PyErr_Occurred() /* conditional && */) {
            PyMem_Free(a);
            PyErr_Format(PyExc_TypeError,
                         "%s tuple has a non-integer at position %d",
                         keyword, (int)i);
            return 0;
        }
    }
    *array = a;
    *length = n;
    return 1;
}

static int
_shape_arg_convert(PyObject *o, void *a)
{
    Py_buffer *view = (Py_buffer *)a;

    if (!_tuple_as_ints(o, "shape", &view->shape, &view->ndim)) {
        return 0;
    }
    return 1;
}

static int
_typestr_arg_convert(PyObject *o, void *a)
{
    /* Due to incompatibilities between the array and new buffer interfaces,
     * as well as significant unicode changes in Python 3.3, this will
     * only handle integer types.
     */
    Py_buffer *view = (Py_buffer *)a;
    char type;
    int is_signed;
    int is_swapped;
    char byteorder = '\0';
    int itemsize;
    char *format;
    PyObject *s;
    const char *typestr;

    format = (char *)&view->internal;
    if (PyUnicode_Check(o)) {
        s = PyUnicode_AsASCIIString(o);
        if (!s) {
            return 0;
        }
    }
    else {
        Py_INCREF(o);
        s = o;
    }
    if (!Bytes_Check(s)) {
        PyErr_Format(PyExc_TypeError, "Expected a string for typestr: got %s",
                     Py_TYPE(s)->tp_name);
        Py_DECREF(s);
        return 0;
    }
    if (Bytes_GET_SIZE(s) != 3) {
        PyErr_SetString(PyExc_TypeError, "Expected typestr to be length 3");
        Py_DECREF(s);
        return 0;
    }
    typestr = Bytes_AsString(s);
    switch (typestr[0]) {

    case BUFPROXY_MY_ENDIAN:
        is_swapped = 0;
        break;
    case BUFPROXY_OTHER_ENDIAN:
        is_swapped = 1;
        break;
    case '|':
        is_swapped = 0;
        break;
    default:
        PyErr_Format(PyExc_ValueError,
                     "unknown byteorder character %c in typestr",
                     typestr[0]);
        Py_DECREF(s);
        return 0;
    }
    switch (typestr[1]) {

    case 'i':
        is_signed = 1;
        break;
    case 'u':
        is_signed = 0;
        break;
    default:
        PyErr_Format(PyExc_ValueError,
                     "unsupported typekind %c in typestr",
                     typestr[1]);
        Py_DECREF(s);
        return 0;
    }
    switch (typestr[2]) {

    case '1':
        type = is_signed ? 'c' : 'B';
        itemsize = 1;
        break;
    case '2':
        byteorder = is_swapped ? BUFPROXY_OTHER_ENDIAN : '=';
        type = is_signed ? 'h' : 'H';
        itemsize = 2;
        break;
    case '3':
        type = 's';
        itemsize = 3;
        break;
    case '4':
        byteorder = is_swapped ? BUFPROXY_OTHER_ENDIAN : '=';
        type = is_signed ? 'i' : 'I';
        itemsize = 4;
        break;
    case '6':
        type = 's';
        itemsize = 6;
        break;
    case '8':
        byteorder = is_swapped ? BUFPROXY_OTHER_ENDIAN : '=';
        type = is_signed ? 'q' : 'Q';
        itemsize = 8;
        break;
    default:
        PyErr_Format(PyExc_ValueError,
                     "unsupported size %c in typestr",
                     typestr[2]);
        Py_DECREF(s);
        return 0;
    }
    if (byteorder != '\0') {
        format[0] = byteorder;
        format[1] = type;
        format[2] = '\0';
    }
    else {
        format[0] = type;
        format[1] = '\0';
    }
    view->format = format;
    view->itemsize = itemsize;
    return 1;
}

static int
_strides_arg_convert(PyObject *o, void *a)
{
    /* Must be called after the array interface nd field has been filled in.
     */
    Py_buffer *view = (Py_buffer *)a;
    int n = 0;

    if (o == Py_None) {
        return 1; /* no strides (optional) given */
    }
    if (!_tuple_as_ints(o, "strides", &view->strides, &n)) {
        return 0;
    }
    if (n != view->ndim) {
        PyErr_SetString(PyExc_TypeError,
                        "strides and shape tuple lengths differ");
        return 0;
    }
    return 1;
}

static int
_data_arg_convert(PyObject *o, void *a)
{
    Py_buffer *view = (Py_buffer *)a;
    Py_ssize_t address;
    int readonly;

    if (!PyTuple_Check(o)) {
        PyErr_Format(PyExc_TypeError, "expected a tuple for data: got %s",
                     Py_TYPE(o)->tp_name);
        return 0;
    }
    if (PyTuple_GET_SIZE(o) != 2) {
        PyErr_SetString(PyExc_TypeError, "expected a length 2 tuple for data");
        return 0;
    }
    address = PyInt_AsSsize_t(PyTuple_GET_ITEM(o, 0));
    if (address == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,
                     "expected an integer address for data item 0: got %s",
                     Py_TYPE(PyTuple_GET_ITEM(o, 0))->tp_name);
        return 0;
    }
    readonly = PyObject_IsTrue(PyTuple_GET_ITEM(o, 1));
    if (readonly == -1) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,
                     "expected a boolean flag for data item 1: got %s",
                     Py_TYPE(PyTuple_GET_ITEM(o, 0))->tp_name);
        return 0;
    }
    view->buf = (void *)address;
    view->readonly = readonly;
    return 1;
}

static int
_parent_arg_convert(PyObject *o, void *a)
{
    Py_buffer *view = (Py_buffer *)a;

    if (o != Py_None) {
        view->obj = o;
        Py_INCREF(o);
    }
    return 1;
}

static void
_free_view(Py_buffer *view)
{
    if (view->shape) {
        PyMem_Free(view->shape);
    }
    if (view->strides) {
        PyMem_Free(view->strides);
    }
}

/**
 * Return a new PgBufproxyObject (Python level constructor).
 */
static int
_fill_view_from_array_interface(PyObject *dict, Py_buffer *view)
{
    PyObject *pyshape = PyDict_GetItemString(dict, "shape");
    PyObject *pytypestr = PyDict_GetItemString(dict, "typestr");
    PyObject *pydata = PyDict_GetItemString(dict, "data");
    PyObject *pystrides = PyDict_GetItemString(dict, "strides");
    int i;

    if (!pyshape) {
        PyErr_SetString(PyExc_ValueError,
                        "required \"shape\" item is missing");
        return -1;
    }
    if (!pytypestr) {
        PyErr_SetString(PyExc_ValueError,
                        "required \"typestr\" item is missing");
        return -1;
    }
    if (!pydata) {
        PyErr_SetString(PyExc_ValueError,
                        "required \"data\" item is missing");
        return -1;
    }
    /* The item processing order is important: "strides" must follow "shape". */
    if (!_shape_arg_convert(pyshape, view)) {
        return -1;
    }
    if (!_typestr_arg_convert(pytypestr, view)) {
        return -1;
    }
    if (!_data_arg_convert(pydata, view)) {
        return -1;
    }
    if (pystrides && !_strides_arg_convert(pystrides, view)) { /* cond. && */
        return -1;
    }
    view->len = view->itemsize;
    for (i = 0; i < view->ndim; ++i) {
        view->len *= view->shape[i];
    }
    return 0;
}

static PyObject *
proxy_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *obj = 0;
    PgBufproxy_CallbackGet get_buffer = 0;
    PgBufproxy_CallbackRelease release_buffer = 0;

    if (!PyArg_ParseTuple(args, "O:Bufproxy", &obj)) {
        return 0;
    }
    if (PyDict_Check(obj)) {
        get_buffer = _get_buffer_from_dict;
        release_buffer = _release_buffer_from_dict;
    }
    return _proxy_subtype_new(type, obj, get_buffer, release_buffer);
}

/**
 * Deallocates the PgBufproxyObject and its members.
 */
static void
proxy_dealloc(PgBufproxyObject *self)
{
    PyObject_GC_UnTrack(self);
    if (self->view_p) {
        self->release_buffer(self->view_p);
        PyMem_Free(self->view_p);
    }
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
    if (self->view_p && self->view_p->obj) /* conditional && */ {
        Py_VISIT(self->view_p->obj);
    }
    if (self->dict) {
        Py_VISIT(self->dict);
    }
    return 0;
}

/**** Getter and setter access ****/
#if PY3
#define Capsule_New(p) PyCapsule_New((p), 0, 0);
#else
#define Capsule_New(p) PyCObject_FromVoidPtr((p), 0);
#endif

static PyObject *
proxy_get_arraystruct(PgBufproxyObject *self, PyObject *closure)
{
    Py_buffer *view_p = self->view_p;
    PyObject *capsule;

    if (!view_p) {
        view_p = PyMem_New(Py_buffer, 1);
        if (!view_p) {
            return PyErr_NoMemory();
        }
        if (self->get_buffer(self->obj, view_p, PyBUF_RECORDS)) {
            PyMem_Free(view_p);
            return 0;
        }
    }
    capsule = PgBuffer_AsArrayStruct(view_p);
    if (!capsule) {
        if (!self->view_p) {
            self->release_buffer(view_p);
            PyMem_Free(view_p);
        }
        return 0;
    }
    self->view_p = view_p;
    return capsule;
}

static PyObject *
proxy_get_arrayinterface(PgBufproxyObject *self, PyObject *closure)
{
    Py_buffer *view_p = self->view_p;
    PyObject *dict;

    if (!view_p) {
        view_p = PyMem_New(Py_buffer, 1);
        if (!view_p) {
            return PyErr_NoMemory();
        }
        if (self->get_buffer(self->obj, view_p, PyBUF_RECORDS)) {
            PyMem_Free(view_p);
            return 0;
        }
    }
    dict = PgBuffer_AsArrayInterface(view_p);
    if (!dict) {
        if (!self->view_p) {
            self->release_buffer(view_p);
            PyMem_Free(view_p);
        }
        return 0;
    }
    self->view_p = view_p;
    return dict;
}

static PyObject *
proxy_get_parent(PgBufproxyObject *self, PyObject *closure)
{
    Py_buffer *view_p = _proxy_get_view(self);

    if (!view_p) {
        return 0;
    }
    if (!view_p->obj) {
        Py_RETURN_NONE;
    }
    Py_INCREF(view_p->obj);
    return view_p->obj;
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
    NOTIMPLEMENTED(0);
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

#if PY3
static int
proxy_getbuffer(PgBufproxyObject *self, Py_buffer *view_p, int flags)
{
    Py_buffer *obj_view_p = PyMem_New(Py_buffer, 1);

    view_p->obj = 0;
    if (!obj_view_p) {
        PyErr_NoMemory();
        return -1;
    }
    if (self->get_buffer(self->obj, view_p, flags)) {
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
proxy_releasebuffer(PgBufproxyObject *self, Py_buffer *view_p)
{
    self->release_buffer((Py_buffer *)view_p->internal);
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

#endif /* #if PY3 */

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
PgBufproxy_New(PyObject *obj,
               PgBufproxy_CallbackGet get_buffer,
               PgBufproxy_CallbackRelease release_buffer)
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
    if (!release_buffer) {
        if (!obj) {
            PyErr_SetString(PyExc_ValueError,
                            "One of arguments 'obj' or 'release_buffer' is "
                            "required: both NULL instead");
            return 0;
        }
        release_buffer = PgBuffer_Release;
    }
    return _proxy_subtype_new(&PgBufproxy_Type,
                              obj,
                              get_buffer,
                              release_buffer);
}

static PyObject *
PgBufproxy_GetParent(PyObject *bufproxy)
{
    PyObject *parent =
        (PyObject *)((PgBufproxyObject *) bufproxy)->obj;

    if (!parent) {
        parent = Py_None;
    }
    Py_INCREF(parent);
    return parent;
}

static int
PgBufproxy_Trip(PyObject *obj)
{
    NOTIMPLEMENTED(-1);
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
/*    c_api[1] = PgBufproxy_New; */
    c_api[1] = mok_PgBufproxy_New; /* $$ **Temporary** */
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
