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

typedef struct PgBufproxyObject_s {
    PyObject_HEAD
    Py_buffer view;
    Py_ssize_t imem[6];                /* shape/stride alloc for ndim <= 3 */
    char cmem[3];                      /* format alloc for simple types    */
    int flags;                         /* contiguity and array shape order */
    PgBufproxy_CallbackBefore before;  /* Lock callback                    */
    PgBufproxy_CallbackAfter after;    /* Release callback                 */
    PyObject *pybefore;                /* Python lock callable             */
    PyObject *pyafter;                 /* Python release callback          */
    int global_release;                /* dealloc callback flag            */
    PyObject *dict;                    /* Allow arbitrary attributes       */
    PyObject *weakrefs;                /* There can be reference cycles    */
} PgBufproxyObject;

static int PgBufproxy_Trip(PyObject *);

/**
 * Helper functions.
 */
static int
proxy_null_before(PyObject *bufproxy) {
    return 0;
}

static void
proxy_null_after(PyObject *bufproxy) {
    return;
}

static int
proxy_python_before(PyObject *bufproxy)
{
    PgBufproxyObject *v = (PgBufproxyObject *)bufproxy;
    PyObject *rvalue;
    PyObject *parent;
    int failed = 0;

    parent = (PyObject *)v->view.obj;
    if (!parent) {
        parent = Py_None;
    }
    Py_INCREF(parent);
    rvalue = PyObject_CallFunctionObjArgs(v->pybefore, parent, 0);
    Py_DECREF(parent);
    if (rvalue) {
        Py_DECREF(rvalue);
    }
    else {
        failed = -1;
    }
    return failed;
}

static void
proxy_python_after(PyObject *bufproxy)
{
    PgBufproxyObject *v = (PgBufproxyObject *)bufproxy;
    PyObject *rvalue;
    PyObject *parent;

    parent = (PyObject *)v->view.obj;
    if (!parent) {
        parent = Py_None;
    }
    Py_INCREF(parent);
    rvalue = PyObject_CallFunctionObjArgs(v->pyafter, parent, 0);
    PyErr_Clear();
    Py_XDECREF(rvalue);
    Py_DECREF(parent);
}

static PyObject *
proxy_new_from_type(PyTypeObject *type,
                    Py_buffer *view,
                    int flags,
                    PgBufproxy_CallbackBefore before,
                    PgBufproxy_CallbackAfter after,
                    PyObject *pybefore,
                    PyObject *pyafter)
{
    int ndim = view->ndim;
    PgBufproxyObject *self;
    Py_ssize_t *shape = 0;
    Py_ssize_t *strides = 0;
    Py_ssize_t format_len = 0;
    char *format = 0;

    if (view->suboffsets) {
        PyErr_SetString(PyExc_BufferError, "unable to handle suboffsets");
        return 0;
    }
    if (view->format) {
        format_len = strlen(view->format);
        if (format_len > 2) {
            format = PyMem_New(char, format_len + 1);
            if (!format) {
                return PyErr_NoMemory();
            }
        }
    }
    if (ndim > 3) {
        shape = PyMem_New(Py_ssize_t, 2 * ndim);
        if (!shape) {
            if (format) {
                PyMem_Free(format);
            }
            return PyErr_NoMemory();
        }
        strides = shape + ndim;
    }

    self = (PgBufproxyObject *)type->tp_alloc(type, 0);
    if (!self) {
        if (format) {
            PyMem_Free(format);
        }
        if (shape) {
            PyMem_Free(shape);
        }
        return 0;
    }

    if (!shape) {
        shape = self->imem;
        strides = shape + ndim;
    }
    if (!format) {
        format = self->cmem;
    }
    self->dict = 0;
    self->weakrefs = 0;
    memcpy(&(self->view), view, sizeof(Py_buffer));
    self->view.format = format;
    if (view->format) {
        strcpy(format, view->format);
    }
    else {
        format[0] = 'B';
        format[1] = '\0';
    }
    if (view->shape) {
        self->view.shape = shape;
        memcpy(shape, view->shape, sizeof(Py_ssize_t) * ndim);
    }
    if (view->strides) {
        self->view.strides = strides;
        memcpy(strides, view->strides, sizeof(Py_ssize_t) * ndim);
    }

    self->flags = flags;
    self->before = proxy_null_before;
    if (pybefore) {
        Py_INCREF(pybefore);
        self->before = proxy_python_before;
    }
    else if (before) {
        self->before = before;
    }
    self->pybefore = pybefore;
    self->after = proxy_null_after;
    if (pyafter) {
        Py_INCREF(pyafter);
        self->after = proxy_python_after;
    }
    else if (after) {
        self->after = after;
    }
    self->pyafter = pyafter;
    self->global_release = 0;
    return (PyObject *)self;
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
    Py_XDECREF(view->obj);
}

/**
 * Return a new PgBufproxyObject (Python level constructor).
 */
static PyObject *
proxy_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Py_buffer view;
    PyObject *pybefore = 0;
    PyObject *pyafter = 0;
    PyObject *self = 0;
    int i;
    /* The argument evaluation order is important: strides must follow shape. */
    char *keywords[] = {"shape", "typestr", "data", "strides", "parent",
                        "before", "after", 0};

    view.obj = 0;
    view.len = 0;
    view.readonly = 1;
    view.ndim = 0;
    view.shape = 0;
    view.strides = 0;
    view.suboffsets = 0;
    view.itemsize = 0;
    view.internal = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&|O&O&OO:Bufproxy",
                                     keywords,
                                     _shape_arg_convert, &view,
                                     _typestr_arg_convert, &view,
                                     _data_arg_convert, &view,
                                     _strides_arg_convert, &view,
                                     _parent_arg_convert, &view,
                                     &pybefore, &pyafter)) {
        _free_view(&view);
        return 0;
    }
    if (pybefore == Py_None) {
        pybefore = 0;
    }
    if (pyafter == Py_None) {
        pyafter = 0;
    }
    Py_XINCREF((PyObject *)view.obj);
    view.len = view.itemsize;
    for (i = 0; i < view.ndim; ++i) {
        view.len *= view.shape[i];
    }
    self = proxy_new_from_type(type, &view, 0,
                               0, 0, pybefore, pyafter);
    _free_view(&view);
    return self;
}

/**
 * Deallocates the PgBufproxyObject and its members.
 */
static void
proxy_dealloc(PgBufproxyObject *self)
{
    /* Guard against recursion */
    if (!self->before) {
        return;
    }
    self->before = 0;

    if (self->global_release) {
        self->after((PyObject *)self);
    }
    Py_XDECREF((PyObject *)self->view.obj);
    Py_XDECREF(self->pybefore);
    Py_XDECREF(self->pyafter);
    Py_XDECREF(self->dict);
    if (self->weakrefs) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }
    if (self->view.shape && self->view.shape != self->imem) {
        PyMem_Free(self->view.shape);
    }
    if (self->view.format && self->view.format != self->cmem) {
        PyMem_Free(self->view.format);
    }
    Py_TYPE(self)->tp_free(self);
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
    PyObject* capsule =  ViewAndFlagsAsArrayStruct (&self->view, self->flags);

    if (capsule && !self->global_release) {
        if (self->before ((PyObject *)self)) {
            Py_DECREF (capsule);
            capsule = 0;
        }
        else {
            self->global_release = 1;
        }
    }
    return capsule;
}

static PyObject *
proxy_get_arrayinterface(PgBufproxyObject *self, PyObject *closure)
{
    PyObject *dict = ViewAsDict(&self->view);

    if (dict && !self->global_release) {
        if (self->before((PyObject *)self)) {
            Py_DECREF(dict);
            dict = 0;
        }
        else {
            self->global_release = 1;
        }
    }
    return dict;
}

static PyObject *
proxy_get_parent(PgBufproxyObject *self, PyObject *closure)
{
    PyObject *parent = (PyObject *)self->view.obj;

    if (!parent) {
        Py_RETURN_NONE;
    }
    Py_INCREF(parent);
    return parent;
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
    if (!(self->flags & VIEW_CONTIGUOUS)) {
        PyErr_SetString(PyExc_ValueError, "the bytes are not contiguous");
        return 0;
    }

    return Bytes_FromStringAndSize((char *)self->view.buf,
                                   self->view.len);
}

static PyObject *
proxy_get_length(PgBufproxyObject *self, PyObject *closure)
{
    return PyInt_FromSsize_t(self->view.len);
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
    char *buf;
    Py_ssize_t length;
    Py_ssize_t offset = 0;

    char *keywords[] = {"bytes", "length", "offset", 0};

#if PY_VERSION_HEX < 0x02050000
#define FORMAT_STRING "s#|i"
#else
#define FORMAT_STRING "s#|n"
#endif

    if (!PyArg_ParseTupleAndKeywords(args, kwds, FORMAT_STRING, keywords,
                                     &buf, &length, &offset)) {
        return 0;
    }

    if (buffer->view.readonly) {
        return RAISE(PyExc_ValueError, "cannot overwrite readonly data");
    }

    if (offset + length > buffer->view.len) {
        return RAISE(PyExc_IndexError, "bytes to write exceed buffer size");
    }

    Py_RETURN_NONE;

#undef FORMAT_STRING
}

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
proxy_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    PgBufproxyObject *v = (PgBufproxyObject *)obj;

    if (flags == PyBUF_SIMPLE && !(v->flags & VIEW_CONTIGUOUS)) {
        PyErr_SetString(PyExc_BufferError, "buffer not contiguous");
        return -1;
    }
    if (flags & PyBUF_WRITABLE && v->view.readonly) {
        PyErr_SetString(PyExc_BufferError, "buffer is readonly");
        return -1;
    }
    if (flags & PyBUF_ND && !v->view.shape) {
        PyErr_SetString(PyExc_BufferError, "buffer shape unavailable");
        return -1;
    }
    if (flags & PyBUF_STRIDES && !v->view.strides) {
        PyErr_SetString(PyExc_BufferError, "buffer strides unavailable");
        return -1;
    }
    else if (flags & PyBUF_ND &&
             !(v->flags & (VIEW_CONTIGUOUS | VIEW_C_ORDER))) {
        PyErr_SetString(PyExc_BufferError, "buffer not C contiguous");
        return -1;
    }
    if (flags & PyBUF_ANY_CONTIGUOUS &&
        !(v->flags & (VIEW_CONTIGUOUS |
                      VIEW_C_ORDER |
                      VIEW_F_ORDER))) {
        PyErr_SetString(PyExc_BufferError, "buffer not contiguous");
        return -1;
    }
    if (flags & PyBUF_C_CONTIGUOUS &&
        !(v->flags & (VIEW_CONTIGUOUS | VIEW_C_ORDER))) {
        PyErr_SetString(PyExc_BufferError, "buffer not C contiguous");
        return -1;
    }
    if (flags & PyBUF_F_CONTIGUOUS &&
        !(v->flags & (VIEW_CONTIGUOUS | VIEW_F_ORDER))) {
        PyErr_SetString(PyExc_BufferError, "buffer not F contiguous");
        return -1;
    }
    if (v->before(obj)) {
        return -1;
    }
    view->obj = obj;
    Py_INCREF(obj);
    view->buf = v->view.buf;
    view->len = v->view.len;
    view->readonly = v->view.readonly;
    view->format = v->view.format;
    view->ndim = v->view.ndim;
    view->shape = v->view.shape;
    view->strides = v->view.strides;
    view->suboffsets = v->view.suboffsets;
    view->itemsize = v->view.itemsize;
    view->internal = 0;
    return 0;
}

static void
proxy_releasebuffer(PyObject *obj, Py_buffer *view)
{
    ((PgBufproxyObject *)obj)->after(obj);
}

static PyBufferProcs proxy_bufferprocs = {
    proxy_getbuffer,
    proxy_releasebuffer
};

#else

static Py_ssize_t
proxy_getreadbuf(PgBufproxyObject *buffer, Py_ssize_t _index, const void **ptr)
{
    if (_index != 0) {
        PyErr_SetString(PyExc_TypeError,
                        "Accessing non-existent buffer segment");
        return -1;
    }

    if (PgBufproxy_Trip((PyObject *)buffer)) {
        return -1;
    }
    *ptr = buffer->view.buf;
    return buffer->view.len;
}

static Py_ssize_t
proxy_getwritebuf(PgBufproxyObject *buffer, Py_ssize_t _index, const void **ptr)
{
    if (buffer->view.readonly) {
        PyErr_SetString(PyExc_TypeError,
                        "Attempting to get write access to a readonly buffer");
        return -1;
    }

    if (_index != 0) {
        PyErr_SetString(PyExc_TypeError, 
                        "Accessing non-existent array segment");
        return -1;
    }

    if (PgBufproxy_Trip((PyObject *)buffer)) {
        return -1;
    }
    *ptr = buffer->view.buf;
    return buffer->view.len;
}

static Py_ssize_t
proxy_getsegcount(PgBufproxyObject *buffer, Py_ssize_t *lenp)
{
    if (lenp) {
        *lenp = buffer->view.len;
    }
    return 1;
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

static PyTypeObject PgBufproxy_Type =
{
    TYPE_HEAD(NULL, 0)
    "pygame._view.Bufproxy",    /* tp_name */
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags */
    "Object bufproxy as an array struct\n",
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    offsetof(PgBufproxyObject, weakrefs),    /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    0,                          /* tp_methods */
    0,                          /* tp_members */
    proxy_getsets,              /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    offsetof(PgBufproxyObject, dict), /* tp_dictoffset */
    0,                          /* tp_init */
    0,                          /* tp_alloc */
    proxy_new,                  /* tp_new */
    0,                          /* tp_free */
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

static PyMethodDef proxy_methods[] = {
    {0, 0, 0, 0}
};

/**** Public C api ***/

static PyObject *
PgBufproxy_New(Py_buffer *view,
           int flags,
           PgBufproxy_CallbackBefore before,
           PgBufproxy_CallbackAfter after)
{
    return proxy_new_from_type(&PgBufproxy_Type,
                               view,
                               flags,
                               before,
                               after,
                               0,
                               0);
}

static PyObject *
PgBufproxy_GetParent(PyObject *bufproxy)
{
    PyObject *parent =
        (PyObject *)((PgBufproxyObject *) bufproxy)->view.obj;

    if (!parent) {
        parent = Py_None;
    }
    Py_INCREF(parent);
    return parent;
}

static int
PgBufproxy_Trip(PyObject *obj)
{
    PgBufproxyObject *proxy = (PgBufproxyObject *)obj;

    if (!PyObject_IsInstance(obj, (PyObject *)&PgBufproxy_Type)) {
        PyErr_Format(PyExc_TypeError,
                     "Expected a BufferProxy instance: got type %s",
                     Py_TYPE(obj)->tp_name);
        return -1;
    }
    if (!proxy->global_release) {
        if (proxy->before(obj)) {
            return -1;
        }
        proxy->global_release = 1;
    }
    return 0;
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
        "_view",
        bufferproxy_doc,
        -1,
        proxy_methods,
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
    module = Py_InitModule3(MODPREFIX "bufferproxy", proxy_methods,
                            bufferproxy_doc);
#endif

    Py_INCREF(&PgBufproxy_Type);
    if (PyModule_AddObject(module,
                           "BufferProxy",
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
    apiobj = encapsulate_api(c_api, "bufferproxy");
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
