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

#include <Python.h>
#include "pgcompat.h"

/* Buffer_TYPE states:
 *
 * Buffer_Type properties are read-only when the BUFOBJ_FILLED flag is set.
 *
 * The BUFOBJ_MEMFREE flag is set when the BufferObject object allocates
 * the memory for the Py_buffer. It is now responsible for freeing the
 * memory.
 *
 * The BUFOBJ_MUTABLE flag can only be changed by a tp_init slot function
 * call.
 *
 * A Buffer_Type object will only release a Py_buffer if both the
 * BUFOBJ_MUTABLE and BUFOBJ_FILLED flags are set. The PyBuffer_Release
 * function is only called by a call of the Buffer_Type release_buffer
 * method, a call to tp_init, or during garbage collection.
 *
 * If only the BUFOBJ_FILLED flag is set, then field values cannot be changed.
 *
 * If the view_p BufferObject field is NULL, then the BUFOBJ_MUTABLE flag
 * must be set. Also, the BUFOBJ_MEMFREE must not be set.
 *
 * The view_p->obj field should always be valid. It either points to
 * an object or is NULL.
 */
#define BUFOBJ_FILLED 0x0001
#define BUFOBJ_MEMFREE 0x0002
#define BUFOBJ_MUTABLE 0x0004

static PyObject *BufferSubtype_New(PyTypeObject *, Py_buffer *, int, int);
static PyObject *Buffer_New(Py_buffer *, int, int);

typedef struct buffer_object_t {
    PyObject_HEAD
    Py_buffer *view_p;
    int flags;
} BufferObject;

static int
Module_AddSsize_tConstant(PyObject *module, const char *name, Py_ssize_t value)
{
    PyObject *py_value = PyLong_FromSsize_t(value);

    if (!py_value) {
        return -1;
    }
    if (PyModule_AddObject(module, name, py_value)) {
        Py_DECREF(py_value);
        return -1;
    }
    return 0;
}

static int
check_view_get(BufferObject *op, const char *name)
{
    if (!op->view_p) {
        PyErr_Format(PyExc_AttributeError,
                     "property %400s is undefined for an unallocated view",
                     name);
        return -1;
    }
    return 0;
}

static int
check_view_set(BufferObject *op, const char *name)
{
    if (op->view_p) {
        if (op->flags & BUFOBJ_FILLED) {
            PyErr_Format(PyExc_AttributeError,
                         "property %400s is read-only", name);
            return -1;
        }
    }
    else {
        PyErr_Format(PyExc_AttributeError,
                     "property %400s is undefined for an unallocated view",
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

/* Restore a BufferObject instance to the equivalent of initializing
 * with a null Py_buffer address.
 */
static void
Buffer_Reset(BufferObject *bp) {
    Py_buffer *view_p;
    int flags;

    if (!bp) {
        return;
    }
    view_p = bp->view_p;
    flags = bp->flags;
    bp->view_p = 0;
    bp->flags = BUFOBJ_MUTABLE;
    if (flags & BUFOBJ_MUTABLE) {
        if (flags & BUFOBJ_FILLED) {
            PyBuffer_Release(view_p);
        }
        else if (view_p && view_p->obj) /* Conditional && */ {
            Py_DECREF(view_p->obj);
        }
        if (flags & BUFOBJ_MEMFREE) {
            PyMem_Free(view_p);
        }
    }
}

static PyObject *
buffer_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    BufferObject *bp = (BufferObject *)subtype->tp_alloc(subtype, 0);

    if (bp) {
        bp->view_p = 0;
        bp->flags = BUFOBJ_MUTABLE;
    }
    return (PyObject *)bp;
}

static int
buffer_init(BufferObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *py_address = 0;
    int filled = 0;
    int preserve = 0;
    Py_buffer *view_p = 0;
    char *keywords[] = {"buffer_address", "filled", "preserve", 0};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Oii:Py_buffer", keywords,
                                     &py_address, &filled, &preserve)) {
        return -1;
    }
    if (py_address == Py_None) {
        py_address = 0;
    }
    if (py_address) {
        if (INT_CHECK(py_address)) {
            view_p = (Py_buffer *)PyLong_AsVoidPtr(py_address);
            if (PyErr_Occurred()) {
                return -1;
            }
        }
        else {
            PyErr_Format(PyExc_TypeError,
                         "argument %400s must be an integer, not '%400s'",
                         keywords[0], Py_TYPE(py_address)->tp_name);
            return -1;
        }
    }
    if (!view_p) {
        if (filled) {
            PyErr_Format(PyExc_ValueError,
                         "argument %400s cannot be True for a NULL %400s",
                         keywords[1], keywords[0]);
            return -1;
        }
        else if (preserve) {
            PyErr_Format(PyExc_ValueError,
                         "argument %400s cannot be True for a NULL %400s",
                         keywords[2], keywords[0]);
            return -1;
        }
    }
    Buffer_Reset(self);
    self->view_p = view_p;
    if (preserve) {
        /* remove mutable flag */
        self->flags &= ~BUFOBJ_MUTABLE;
    }
    if (filled) {
        /* add filled flag */
        self->flags |= BUFOBJ_FILLED;
    }
    else if (view_p) {
        view_p->obj = 0;
        view_p->buf = 0;
        view_p->len = 0;
        view_p->itemsize = 0;
        view_p->readonly = 1;
        view_p->format = 0;
        view_p->ndim = 0;
        view_p->shape = 0;
        view_p->strides = 0;
        view_p->suboffsets = 0;
        view_p->internal = 0;
    }
    return 0;
}

static void
buffer_dealloc(BufferObject *self)
{
    PyObject_GC_UnTrack(self);
    Buffer_Reset(self);
    Py_TYPE(self)->tp_free(self);
}

static int
buffer_traverse(BufferObject *self, visitproc visit, void *arg)
{
    if (self->view_p && self->view_p->obj) /* Conditional && */ {
        Py_VISIT(self->view_p->obj);
    }
    return 0;
}

static PyObject *
buffer_get_buffer(BufferObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    int flags = PyBUF_SIMPLE;
    int bufobj_flags = self->flags;
    char *keywords[] = {"obj", "flags", 0};

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                                     "O|i", keywords, &obj, &flags)) {
        return 0;
    }
    if (bufobj_flags & BUFOBJ_FILLED) {
        PyErr_SetString(PyExc_ValueError,
                        "The Py_buffer struct is already filled in");
        return 0;
    }
    self->flags = BUFOBJ_MUTABLE & bufobj_flags;
    if (!self->view_p) {
        self->view_p = PyMem_New(Py_buffer, 1);
        if (!self->view_p) {
            return PyErr_NoMemory();
        }
        bufobj_flags |= BUFOBJ_MEMFREE;
    }
    if (PyObject_GetBuffer(obj, self->view_p, flags)) {
        if (bufobj_flags & BUFOBJ_MEMFREE) {
            PyMem_Free(self->view_p);
            self->view_p = 0;
        }
        return 0;
    }
    self->flags |= (bufobj_flags & BUFOBJ_MEMFREE) | BUFOBJ_FILLED;
    Py_RETURN_NONE;
}

static PyObject *
buffer_release_buffer(BufferObject *self)
{
    int flags = self->flags;
    Py_buffer *view_p = self->view_p;

    if ((flags & BUFOBJ_FILLED) && (flags & BUFOBJ_MUTABLE)) {
        self->flags = BUFOBJ_FILLED;  /* Guard against reentrant calls */
        PyBuffer_Release(view_p);
        self->flags = BUFOBJ_MUTABLE;
        if (flags & BUFOBJ_MEMFREE) {
            self->view_p = 0;
            PyMem_Free(view_p);
        }
        else {
            view_p->obj = 0;
            view_p->buf = 0;
            view_p->len = 0;
            view_p->itemsize = 0;
            view_p->readonly = 1;
            view_p->format = 0;
            view_p->ndim = 0;
            view_p->shape = 0;
            view_p->strides = 0;
            view_p->suboffsets = 0;
            view_p->internal = 0;
        }
    }
    Py_RETURN_NONE;
}

static struct PyMethodDef buffer_methods[] = {
    {"get_buffer", (PyCFunction)buffer_get_buffer, METH_VARARGS | METH_KEYWORDS,
     "fill in Py_buffer fields with a PyObject_GetBuffer call"},
    {"release_buffer", (PyCFunction)buffer_release_buffer, METH_NOARGS,
     "release the Py_buffer with a PyBuffer_Release call"},
    {0, 0, 0, 0}
};

static PyObject *
buffer_get_obj(BufferObject *self, void *closure)
{
    if (check_view_get(self, (const char *)closure)) {
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
    PyObject *tmp;

    if (check_view_set(self, (const char *)closure)) {
        return -1;
    }
    if (check_value(value, (const char *)closure)) {
        return -1;
    }
    tmp = self->view_p->obj;
    if (value != Py_None) {
        Py_INCREF(value);
        self->view_p->obj = value;
    }
    else {
        self->view_p->obj = 0;
    }
    if (tmp) {
        Py_DECREF(tmp);
    }
    return 0;
}

static PyObject *
buffer_get_buf(BufferObject *self, void *closure)
{
    if (check_view_get(self, (const char *)closure)) {
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
    if (check_view_set(self, (const char *)closure)) {
        return -1;
    }
    return set_void_ptr(&self->view_p->buf, value, (const char *)closure);
}

static PyObject *
buffer_get_len(BufferObject *self, void *closure)
{
    if (check_view_get(self, (const char *)closure)) {
        return 0;
    }
    return PyLong_FromSsize_t(self->view_p->len);
}

static int
buffer_set_len(BufferObject *self, PyObject *value, void *closure)
{
    if (check_view_set(self, (const char *)closure)) {
        return -1;
    }
    return set_py_ssize_t(&self->view_p->len, value, (const char *)closure);
}

static PyObject *
buffer_get_readonly(BufferObject *self, void *closure)
{
    if (check_view_get(self, (const char *)closure)) {
        return 0;
    }
    return PyBool_FromLong((long)self->view_p->readonly);
}

static int
buffer_set_readonly(BufferObject *self, PyObject *value, void *closure)
{
    int readonly = 1;

    if (check_view_set(self, (const char *)closure)) {
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
    if (check_view_get(self, (const char *)closure)) {
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

    if (check_view_set(self, (const char *)closure)) {
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
    if (check_view_get(self, (const char *)closure)) {
        return 0;
    }
    return PyLong_FromLong(self->view_p->ndim);
}

static int
buffer_set_ndim(BufferObject *self, PyObject *value, void *closure)
{
    Py_ssize_t ndim = 0;

    if (check_view_set(self, (const char *)closure)) {
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
    if (check_view_get(self, (const char *)closure)) {
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

    if (check_view_set(self, (const char *)closure)) {
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
    if (check_view_get(self, (const char *)closure)) {
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

    if (check_view_set(self, (const char *)closure)) {
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
    if (check_view_get(self, (const char *)closure)) {
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

    if (check_view_set(self, (const char *)closure)) {
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
    if (check_view_get(self, (const char *)closure)) {
        return 0;
    }
    return PyLong_FromSsize_t(self->view_p->itemsize);
}

static int
buffer_set_itemsize(BufferObject *self, PyObject *value, void *closure)
{
    if (check_view_set(self, (const char *)closure)) {
        return -1;
    }
    return set_py_ssize_t(&self->view_p->itemsize, value, (const char *)closure);
}

static PyObject *
buffer_get_internal(BufferObject *self, void *closure)
{
    if (check_view_get(self, (const char *)closure)) {
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
    if (check_view_set(self, (const char *)closure)) {
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
#define BUFFER_TPFLAGS (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | \
                        Py_TPFLAGS_HAVE_GC)

static PyTypeObject Py_buffer_Type =
{
    TYPE_HEAD(NULL, 0)
    BUFFER_TYPE_FULLNAME,       /* tp_name */
    sizeof (BufferObject),      /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor)buffer_dealloc, /* tp_dealloc */
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
    (traverseproc)buffer_traverse,  /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    buffer_methods,             /* tp_methods */
    0,                          /* tp_members */
    buffer_getset,              /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)buffer_init,      /* tp_init */
    PyType_GenericAlloc,        /* tp_alloc */
    buffer_new,                 /* tp_new */
    PyObject_GC_Del,            /* tp_free */
};

static PyObject *
BufferSubtype_New(PyTypeObject *subtype,
                  Py_buffer *view_p,
                  int filled,
                  int preserve)
{
    BufferObject *bp = (BufferObject *)Py_buffer_Type.tp_alloc(subtype, 0);
    if (!bp) {
        return 0;
    }
    bp->view_p = view_p;
    bp->flags = 0;
    if (bp->view_p) {
        if (filled) {
            bp->flags |= BUFOBJ_FILLED;
        }
        else {
            bp->view_p->obj = 0;
        }
        if (!preserve) {
            bp->flags |= BUFOBJ_MUTABLE;
        }
    }
    else {
        bp->flags = BUFOBJ_MUTABLE;
    }
    return (PyObject *)bp;
}

static PyObject *
Buffer_New(Py_buffer *view_p, int filled, int preserve)
{
    return BufferSubtype_New(&Py_buffer_Type, view_p, filled, preserve);
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
    PyObject *py_view = Buffer_New(view_p, 0, 1);
    PyObject *py_rval = 0;
    int rval = -1;

    if (py_view) {
        view_p->obj = 0;
        py_rval = PyObject_CallMethod(self, "_get_buffer", "(Oi)",
                                      py_view, flags);
        Buffer_Reset((BufferObject *)py_view);
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
    PyObject *py_view = Buffer_New(view_p, 1, 1);
    PyObject *py_rval = 0;

    if (py_view) {
        py_rval = PyObject_CallMethod(self, "_release_buffer", "(O)", py_view);
        if (py_rval) {
            Py_DECREF(py_rval);
        }
        else {
            PyErr_Clear();
        }
        Buffer_Reset((BufferObject *)py_view);
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
    if (Module_AddSsize_tConstant(module,
                                  "PyBUFFER_SIZEOF",
                                  sizeof (Py_buffer)) ||
        PyModule_AddIntMacro(module, PyBUF_SIMPLE) ||
        PyModule_AddIntMacro(module, PyBUF_WRITABLE) ||
        PyModule_AddIntMacro(module, PyBUF_STRIDES) ||
        PyModule_AddIntMacro(module, PyBUF_ND) ||
        PyModule_AddIntMacro(module, PyBUF_C_CONTIGUOUS) ||
        PyModule_AddIntMacro(module, PyBUF_F_CONTIGUOUS) ||
        PyModule_AddIntMacro(module, PyBUF_ANY_CONTIGUOUS) ||
        PyModule_AddIntMacro(module, PyBUF_INDIRECT) ||
        PyModule_AddIntMacro(module, PyBUF_FORMAT) ||
        PyModule_AddIntMacro(module, PyBUF_STRIDED) ||
        PyModule_AddIntMacro(module, PyBUF_STRIDED_RO) ||
        PyModule_AddIntMacro(module, PyBUF_RECORDS) ||
        PyModule_AddIntMacro(module, PyBUF_RECORDS_RO) ||
        PyModule_AddIntMacro(module, PyBUF_FULL) ||
        PyModule_AddIntMacro(module, PyBUF_FULL_RO) ||
        PyModule_AddIntMacro(module, PyBUF_CONTIG) ||
        PyModule_AddIntMacro(module, PyBUF_CONTIG_RO)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
