/*
  pygame - Python Game Library
  Module newbuffer.c, Copyright (C) 2013  Lenard Lindstrom

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

/* Exports type BufferMixin, a mixin which adds a new buffer interface to a
   Python new-style class.
 */
#include <Python.h>
#include "pgcompat.h"

static PyObject *BufferSubtype_New(PyTypeObject *, Py_buffer *);
static PyObject *Py_buffer_New(Py_buffer *);

typedef struct buffer_object_t {
    PyObject_HEAD
    Py_buffer *view_p;
} BufferObject;

static int
check_view(BufferObject *op, const char *name)
{
    if (!op->view_p) {
        PyErr_Format(PyExc_AttributeError,
                     "property %400s is undefined for a NULL view reference",
                     name);
        return -1;
    }
    return 0;
}

static int
check_value(PyObject *o, const char *name)
{
    if (!o) {
        PyErr_Format(PyExc_AttributeError,
                     "property %400s cannot be deleted", name);
        return -1;
    }
    return 0;
}

#if PY_MAJOR_VERSION < 3
#define INT_CHECK(o) (PyInt_Check(o) || PyLong_Check(o))
#define INT_AS_PY_SSIZE_T(o) (PyInt_AsSsize_t(o))
#else
#define INT_CHECK(o) (PyLong_Check(o))
#define INT_AS_PY_SSIZE_T(o) (PyLong_AsSsize_t(o))
#endif

static int
set_void_ptr(void **vpp, PyObject *o, const char *name)
{
    void *vp = 0;

    if (check_value(o, name)) {
        return -1;
    }
    if (INT_CHECK(o)) {
        vp = PyLong_AsVoidPtr(o);
        if (PyErr_Occurred()) {
            return -1;
        }
    }
    else if (o == Py_None) {
        vp = 0;
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "property %400s must be a Python integer, not '%400s'",
                     name, Py_TYPE(o)->tp_name);
        return -1;
    }
    *vpp = vp;
    return 0;
}

static int
set_py_ssize_t(Py_ssize_t *ip, PyObject *o, const char *name)
{
    Py_ssize_t i;

    if (check_value(o, name)) {
        return -1;
    }
    if (!INT_CHECK(o)) {
        PyErr_Format(PyExc_TypeError,
                     "property %100s must be a Python integer, not '%400s'",
                     name, Py_TYPE(o)->tp_name);
        return -1;
    }
    i = INT_AS_PY_SSIZE_T(o);
    if (PyErr_Occurred()) {
        return -1;
    }
    *ip = i;
    return 0;
}

static PyObject *
buffer_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    PyObject *arg = 0;
    Py_buffer *view_p = 0;
    char *keywords[] = {"buffer_address", 0};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O:Py_buffer", keywords, &arg)) {
        return 0;
    }
    if (!arg) {
        arg = Py_None;
    }
    if (INT_CHECK(arg)) {
        view_p = (Py_buffer *)PyLong_AsVoidPtr(arg);
        if (PyErr_Occurred()) {
            return 0;
        }
    }
    else if (arg != Py_None) {
        PyErr_Format(PyExc_TypeError,
                     "%400s() argument must be a Python integer, not '%400s'",
                     Py_TYPE(subtype)->tp_name, Py_TYPE(arg)->tp_name);
        return 0;
    }
    return BufferSubtype_New(subtype, view_p);
}

static PyObject *
buffer_get_obj(BufferObject *self, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return 0;
    }
    if (!self->view_p->obj) {
        Py_RETURN_NONE;
    }
    Py_INCREF(self->view_p->obj);
    return self->view_p->obj;
}

static int
buffer_set_obj(BufferObject *self, PyObject *value, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return -1;
    }
    if (check_value(value, (const char *)closure)) {
        return -1;
    }
    Py_XDECREF(self->view_p->obj);
    if (value != Py_None) {
        Py_INCREF(value);
        self->view_p->obj = value;
    }
    else {
        self->view_p->obj = 0;
    }
    return 0;
}

static PyObject *
buffer_get_buf(BufferObject *self, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return 0;
    }
    if (!self->view_p->buf) {
        Py_RETURN_NONE;
    }
    return PyLong_FromVoidPtr(self->view_p->buf);
}

static int
buffer_set_buf(BufferObject *self, PyObject *value, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return -1;
    }
    return set_void_ptr(&self->view_p->buf, value, (const char *)closure);
}

static PyObject *
buffer_get_len(BufferObject *self, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return 0;
    }
    return PyLong_FromSsize_t(self->view_p->len);
}

static int
buffer_set_len(BufferObject *self, PyObject *value, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return -1;
    }
    return set_py_ssize_t(&self->view_p->len, value, (const char *)closure);
}

static PyObject *
buffer_get_readonly(BufferObject *self, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return 0;
    }
    return PyBool_FromLong((long)self->view_p->readonly);
}

static int
buffer_set_readonly(BufferObject *self, PyObject *value, void *closure)
{
    int readonly = 1;

    if (check_view(self, (const char *)closure)) {
        return -1;
    }
    if (check_value(value, (const char *)closure)) {
        return -1;
    }
    readonly = PyObject_IsTrue(value);
    if (readonly == -1) {
        return -1;
    }
    self->view_p->readonly = readonly;
    return 0;
}

static PyObject *
buffer_get_format(BufferObject *self, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return 0;
    }
    if (!self->view_p->format) {
        Py_RETURN_NONE;
    }
    return PyLong_FromVoidPtr(self->view_p->format);
}

static int
buffer_set_format(BufferObject *self, PyObject *value, void *closure)
{
    void *vp = 0;

    if (check_view(self, (const char *)closure)) {
        return -1;
    }
    if (set_void_ptr(&vp, value, (const char *)closure)) {
        return -1;
    }
    self->view_p->format = (char *)vp;
    return 0;
}

static PyObject *
buffer_get_ndim(BufferObject *self, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return 0;
    }
    return PyLong_FromLong(self->view_p->ndim);
}

static int
buffer_set_ndim(BufferObject *self, PyObject *value, void *closure)
{
    Py_ssize_t ndim = 0;

    if (check_view(self, (const char *)closure)) {
        return -1;
    }
    if (set_py_ssize_t(&ndim, value, (const char *)closure)) {
        return -1;
    }
    self->view_p->ndim = (int)ndim;
    return 0;
}

static PyObject *
buffer_get_shape(BufferObject *self, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return 0;
    }
    if (!self->view_p->shape) {
        Py_RETURN_NONE;
    }
    return PyLong_FromVoidPtr(self->view_p->shape);
}

static int
buffer_set_shape(BufferObject *self, PyObject *value, void *closure)
{
    void *vp;

    if (check_view(self, (const char *)closure)) {
        return -1;
    }
    if (set_void_ptr(&vp, value, (const char *)closure)) {
        return -1;
    }
    self->view_p->shape = (Py_ssize_t *)vp;
    return 0;
}

static PyObject *
buffer_get_strides(BufferObject *self, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return 0;
    }
    if (!self->view_p->strides) {
        Py_RETURN_NONE;
    }
    return PyLong_FromVoidPtr(self->view_p->strides);
}

static int
buffer_set_strides(BufferObject *self, PyObject *value, void *closure)
{
    void *vp;

    if (check_view(self, (const char *)closure)) {
        return -1;
    }
    if (set_void_ptr(&vp, value, (const char *)closure)) {
        return -1;
    }
    self->view_p->strides = (Py_ssize_t *)vp;
    return 0;
}

static PyObject *
buffer_get_suboffsets(BufferObject *self, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return 0;
    }
    if (!self->view_p->suboffsets) {
        Py_RETURN_NONE;
    }
    return PyLong_FromVoidPtr(self->view_p->suboffsets);
}

static int
buffer_set_suboffsets(BufferObject *self, PyObject *value, void *closure)
{
    void *vp;

    if (check_view(self, (const char *)closure)) {
        return -1;
    }
    if (set_void_ptr(&vp, value, (const char *)closure)) {
        return -1;
    }
    self->view_p->suboffsets = (Py_ssize_t *)vp;
    return 0;
}

static PyObject *
buffer_get_itemsize(BufferObject *self, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return 0;
    }
    return PyLong_FromSsize_t(self->view_p->itemsize);
}

static int
buffer_set_itemsize(BufferObject *self, PyObject *value, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return -1;
    }
    return set_py_ssize_t(&self->view_p->itemsize, value, (const char *)closure);
}

static PyObject *
buffer_get_internal(BufferObject *self, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return 0;
    }
    if (!self->view_p->internal) {
        Py_RETURN_NONE;
    }
    return PyLong_FromVoidPtr(self->view_p->internal);
}

static int
buffer_set_internal(BufferObject *self, PyObject *value, void *closure)
{
    if (check_view(self, (const char *)closure)) {
        return -1;
    }
    return set_void_ptr(&self->view_p->internal, value, (const char *)closure);
}

static PyGetSetDef buffer_getset[] = {
    {"obj",
     (getter)buffer_get_obj, (setter)buffer_set_obj, 0, "obj"},
    {"buf",
     (getter)buffer_get_buf, (setter)buffer_set_buf, 0, "buf"},
    {"len",
     (getter)buffer_get_len, (setter)buffer_set_len, 0, "len"},
    {"readonly",
     (getter)buffer_get_readonly, (setter)buffer_set_readonly, 0, "readonly"},
    {"format",
     (getter)buffer_get_format, (setter)buffer_set_format, 0, "format"},
    {"ndim",
     (getter)buffer_get_ndim, (setter)buffer_set_ndim, 0, "ndim"},
    {"shape",
     (getter)buffer_get_shape, (setter)buffer_set_shape, 0, "shape"},
    {"strides",
     (getter)buffer_get_strides, (setter)buffer_set_strides, 0, "shape"},
    {"suboffsets",
     (getter)buffer_get_suboffsets, (setter)buffer_set_suboffsets, 0,
     "suboffsets"},
    {"itemsize",
     (getter)buffer_get_itemsize, (setter)buffer_set_itemsize, 0, "itemsize"},
    {"internal",
     (getter)buffer_get_internal, (setter)buffer_set_internal, 0, "internal"},
    {0, 0, 0, 0, 0}
};

static int
buffer_bool(BufferObject *self)
{
    return self->view_p != NULL;
}

static PyNumberMethods buffer_as_number = {
    (binaryfunc)0,                              /* nb_add */
    (binaryfunc)0,                              /* nb_subtract */
    (binaryfunc)0,                              /* nb_multiply */
#if PY_MAJOR_VERSION < 3
    (binaryfunc)0,                              /* nb_divide */
#endif
    (binaryfunc)0,                              /* nb_remainder */
    (binaryfunc)0,                              /* nb_divmod */
    (ternaryfunc)0,                             /* nb_power */
    (unaryfunc)0,                               /* nb_negative */
    (unaryfunc)0,                               /* nb_positive */
    (unaryfunc)0,                               /* nb_absolute */
    (inquiry)buffer_bool,                       /* nb_nonzero / nb_bool */
    (unaryfunc)0,                               /* nb_invert */
    (binaryfunc)0,                              /* nb_lshift */
    (binaryfunc)0,                              /* nb_rshift */
    (binaryfunc)0,                              /* nb_and */
    (binaryfunc)0,                              /* nb_xor */
    (binaryfunc)0,                              /* nb_or */
#if PY_MAJOR_VERSION < 3
    (coercion)0,                                /* nb_coerce */
#endif
    (unaryfunc)0,                               /* nb_int */
#if PY_MAJOR_VERSION < 3
    (unaryfunc)0,                               /* nb_long */
#else
    0,                                          /* nb_reserved */
#endif
    (unaryfunc)0,                               /* nb_float */
#if PY_MAJOR_VERSION < 3
    (unaryfunc)0,                               /* nb_oct */
    (unaryfunc)0,                               /* nb_hex */
#endif
    (binaryfunc)0,                              /* nb_inplace_add */
    (binaryfunc)0,                              /* nb_inplace_subtract */
    (binaryfunc)0,                              /* nb_inplace_multiply */
#if PY_MAJOR_VERSION < 3
    (binaryfunc)0,                              /* nb_inplace_divide */
#endif
    (binaryfunc)0,                              /* nb_inplace_remainder */
    (ternaryfunc)0,                             /* nb_inplace_power */
    (binaryfunc)0,                              /* nb_inplace_lshift */
    (binaryfunc)0,                              /* nb_inplace_rshift */
    (binaryfunc)0,                              /* nb_inplace_and */
    (binaryfunc)0,                              /* nb_inplace_xor */
    (binaryfunc)0,                              /* nb_inplace_or */
    (binaryfunc)0,                              /* nb_floor_divide */
    (binaryfunc)0,                              /* nb_true_divide */
    (binaryfunc)0,                              /* nb_inplace_floor_divide */
    (binaryfunc)0,                              /* nb_inplace_true_divide */
#if PY_VERSION_HEX >= 0x02050000
    (unaryfunc)0,                               /* nb_index */
#endif
};

#define BUFFER_TYPE_FULLNAME "newbuffer.Py_buffer"
#define BUFFER_TPFLAGS (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE)

static PyTypeObject Py_buffer_Type =
{
    TYPE_HEAD(NULL, 0)
    BUFFER_TYPE_FULLNAME,       /* tp_name */
    sizeof (BufferObject),      /* tp_basicsize */
    0,                          /* tp_itemsize */
    0,                          /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    0,                          /* tp_repr */
    &buffer_as_number,          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    BUFFER_TPFLAGS,             /* tp_flags */
    "Python level Py_buffer struct wrapper\n",
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    0,                          /* tp_methods */
    0,                          /* tp_members */
    buffer_getset,              /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    0,                          /* tp_init */
    PyType_GenericAlloc,        /* tp_alloc */
    buffer_new,                 /* tp_new */
    PyObject_Del,               /* tp_free */
};

static PyObject *
BufferSubtype_New(PyTypeObject *subtype, Py_buffer *view_p)
{
    PyObject *op = Py_buffer_Type.tp_alloc(subtype, 0);
    if (!op) {
        return 0;
    }
    ((BufferObject *)op)->view_p = view_p;
    return op;
}

static PyObject *
Py_buffer_New(Py_buffer *view_p)
{
    return BufferSubtype_New(&Py_buffer_Type, view_p);
}

static PyObject *
mixin__get_buffer(PyObject *self, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError, "abstract method");
    return 0;
}

static PyObject *
mixin__release_buffer(PyObject *self, PyObject *arg)
{
    Py_RETURN_NONE;
}

static PyMethodDef mixin_methods[] = {
    {"_get_buffer", (PyCFunction)mixin__get_buffer, METH_VARARGS,
     "new buffer protocol default bf_getbuffer handler"},
    {"_release_buffer", (PyCFunction)mixin__release_buffer, METH_O,
     "new buffer protocol default bf_releasebuffer handler"},
    {0, 0, 0, 0}
};

static int
mixin_getbuffer(PyObject *self, Py_buffer *view_p, int flags)
{
    BufferObject *py_view = (BufferObject *)Py_buffer_New(view_p);
    PyObject *py_rval = 0;
    int rval = -1;

    if (py_view) {
        view_p->obj = 0;
        py_rval = PyObject_CallMethod(self, "_get_buffer", "Oi",
                                      (PyObject *)py_view, flags);
        py_view->view_p = 0;
        Py_DECREF(py_view);
        if (py_rval == Py_None) {
            rval = 0;
            Py_DECREF(py_rval);
        }
        else if (py_rval) {
            PyErr_SetString(PyExc_ValueError,
                            "_get_buffer method return value was not None");
            Py_DECREF(py_rval);
        }
    }
    return rval;
}

static void
mixin_releasebuffer(PyObject *self, Py_buffer *view_p)
{
    BufferObject *py_view = (BufferObject *)Py_buffer_New(view_p);
    PyObject *py_rval = 0;

    if (py_view) {
        py_rval = PyObject_CallMethod(self, "_release_buffer", "O",
                                      (PyObject *)py_view);
        if (py_rval) {
            Py_DECREF(py_rval);
        }
        else {
            PyErr_Clear();
        }
        py_view->view_p = 0;
        Py_DECREF(py_view);
    }
    else {
        PyErr_Clear();
    }
}

static PyBufferProcs mixin_bufferprocs = {
#if PY_VERSION_HEX < 0x03000000
    0,
    0,
    0,
    0,
#endif
    (getbufferproc)mixin_getbuffer,
    (releasebufferproc)mixin_releasebuffer
};

#if PY_VERSION_HEX < 0x03000000
#define MIXIN_TPFLAGS \
    (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |  Py_TPFLAGS_HAVE_NEWBUFFER)
#else
#define MIXIN_TPFLAGS \
    (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE)
#endif

#define BUFFER_MIXIN_TYPE_FULLNAME "newbuffer.BufferMixin"

static PyTypeObject BufferMixin_Type =
{
    TYPE_HEAD(NULL, 0)
    BUFFER_MIXIN_TYPE_FULLNAME, /* tp_name */
    sizeof (PyObject),          /* tp_basicsize */
    0,                          /* tp_itemsize */
    0,                          /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    0,                          /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    &mixin_bufferprocs,         /* tp_as_buffer */
    MIXIN_TPFLAGS,              /* tp_flags */
    "Python level new buffer protocol exporter\n",
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    mixin_methods,              /* tp_methods */
    0,                          /* tp_members */
    0,                          /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    0,                          /* tp_init */
    PyType_GenericAlloc,        /* tp_alloc */
    PyType_GenericNew,          /* tp_new */
    PyObject_Del,               /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0                           /* tp_del */
};

/*DOC*/ static char newbuffer_doc[] =
/*DOC*/    "exports BufferMixin, add a new buffer interface to a class";

MODINIT_DEFINE(newbuffer)
{
    PyObject *module;
    PyObject *obj;

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "newbuffer",
        newbuffer_doc,
        -1,
        NULL, NULL, NULL, NULL, NULL
    };
#endif

    /* prepare exported types */
    if (PyType_Ready(&Py_buffer_Type) < 0) {
        MODINIT_ERROR;
    }
    if (PyType_Ready(&BufferMixin_Type) < 0) {
        MODINIT_ERROR;
    }

#define bufferproxy_docs ""

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3("newbuffer", 0, newbuffer_doc);
#endif

    Py_INCREF(&BufferMixin_Type);
    if (PyModule_AddObject(module,
                           "BufferMixin",
                           (PyObject *)&BufferMixin_Type)) {
        Py_DECREF(&BufferMixin_Type);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    Py_INCREF(&Py_buffer_Type);
    if (PyModule_AddObject(module, "Py_buffer", (PyObject *)&Py_buffer_Type)) {
        Py_DECREF(&Py_buffer_Type);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    obj = PyLong_FromSsize_t((Py_ssize_t)sizeof (Py_buffer));
    if (!obj) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    if (PyModule_AddObject(module, "Py_buffer_SIZEOF", obj)) {
        Py_DECREF(&obj);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
