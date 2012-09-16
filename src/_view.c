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
  This module exports an object which provides an array interface to
  another object's buffer. Both the C level array structure -
  __array_struct__ - interface and Python level - __array_interface__ -
  are exposed.
 */

#define NO_PYGAME_C_API
#include "pygame.h"
#include "pgcompat.h"
#include "pgview.h"

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define VIEW_MY_ENDIAN '<'
#define VIEW_OTHER_ENDIAN '>'
#else
#define VIEW_MY_ENDIAN '>'
#define VIEW_OTHER_ENDIAN '<'
#endif

typedef struct PgViewObject_s {
    PyObject_HEAD
    Py_buffer bufview;
    Py_ssize_t imem[6];                   /* shape/stride alloc for ndim <= 3 */
    char cmem[3];                         /* format alloc for simple types    */
    int flags;                            /* contiguity and array shape order */
    PgView_PreludeCallback prelude;       /* Lock callback                    */
    PgView_PostscriptCallback postscript; /* Release callback                 */
    PyObject *pyprelude;                  /* Python lock callable             */
    PyObject *pypostscript;               /* Python release callback          */
    int global_release;                   /* dealloc callback flag            */
    PyObject *weakrefs;                   /* There can be reference cycles    */
} PgViewObject;

typedef struct capsule_interface_s {
    PyArrayInterface inter;
    PyObject *parent;
    Py_intptr_t imem[1];
} CapsuleInterface;

static int Pg_GetArrayInterface(PyObject *, PyObject **, PyArrayInterface **);
static PyObject *Pg_ArrayStructAsDict(PyArrayInterface *);
static PyObject *Pg_BufferViewAsDict(Py_buffer *);

/**
 * Helper functions.
 */
static int
_view_null_prelude(PyObject *view) {
    return 0;
}

static void
_view_null_postscript(PyObject *view) {
    return;
}

static int
_view_python_prelude(PyObject *view)
{
    PgViewObject *v = (PgViewObject *)view;
    PyObject *rvalue;
    PyObject *parent;
    int failed = 0;
    
    parent = (PyObject *)v->bufview.obj;
    if (!parent) {
        parent = Py_None;
    }
    Py_INCREF(parent);
    rvalue = PyObject_CallFunctionObjArgs(v->pyprelude, parent, 0);
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
_view_python_postscript(PyObject *view)
{
    PgViewObject *v = (PgViewObject *)view;
    PyObject *rvalue;
    PyObject *parent;

    parent = (PyObject *)v->bufview.obj;
    if (!parent) {
        parent = Py_None;
    }
    Py_INCREF(parent);
    rvalue = PyObject_CallFunctionObjArgs(v->pypostscript, parent, 0);
    PyErr_Clear();
    Py_XDECREF(rvalue);
    Py_DECREF(parent);
}

static PyObject *
_view_new_from_type(PyTypeObject *type,
                    Py_buffer *bufview,
                    int flags,
                    PgView_PreludeCallback prelude,
                    PgView_PostscriptCallback postscript,
                    PyObject *pyprelude,
                    PyObject *pypostscript)
{
    int ndim = bufview->ndim;
    PgViewObject *self;
    Py_ssize_t *shape = 0;
    Py_ssize_t *strides = 0;
    Py_ssize_t format_len = 0;
    char *format = 0;

    if (bufview->suboffsets) {
        PyErr_SetString(PyExc_BufferError, "unable to handle suboffsets");
        return 0;
    }
    if (bufview->format) {
        format_len = strlen(bufview->format);
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
    
    self = (PgViewObject *)type->tp_alloc(type, 0);
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
    self->weakrefs = 0;
    memcpy(&(self->bufview), bufview, sizeof(Py_buffer));
    self->bufview.format = format;
    if (bufview->format) {
        strcpy(format, bufview->format);
    }
    else {
        format[0] = 'B';
        format[1] = '\0';
    }
    if (bufview->shape) {
        self->bufview.shape = shape;
        memcpy(shape, bufview->shape, sizeof(Py_ssize_t) * ndim);
    }
    if (bufview->strides) {
        self->bufview.strides = strides;
        memcpy(strides, bufview->strides, sizeof(Py_ssize_t) * ndim);
    }
    
    self->flags = flags;
    self->prelude = _view_null_prelude;
    if (pyprelude) {
        Py_INCREF(pyprelude);
        self->prelude = _view_python_prelude;
    }
    else if (prelude) {
        self->prelude = prelude;
    }
    self->pyprelude = pyprelude;
    self->postscript = _view_null_postscript;
    if (pypostscript) {
        Py_INCREF(pypostscript);
        self->postscript = _view_python_postscript;
    }
    else if (postscript) {
        self->postscript = postscript;
    }
    self->pypostscript = pypostscript;
    self->global_release = 0;
    return (PyObject *)self;
}

static char
_as_arrayinter_typekind(const Py_buffer *view)
{
    char type = view->format[0];
    char typekind;
    
    switch (type) {
    
    case '<':
    case '>':
    case '=':
    case '@':
    case '!':
        type = view->format[1];
    }
    switch (type) {
        
    case 'c':
    case 'h':
    case 'i':
    case 'l':
    case 'q':
        typekind = 'i';
        break;
    case 'b':
    case 'B':
    case 'H':
    case 'I':
    case 'L':
    case 'Q':
    case 's':
        typekind = 'u';
        break;
    default:
        /* Unknown type */
        typekind = 's';
    }
    return typekind;
}

static char
_as_arrayinter_byteorder(const Py_buffer *view)
{
    char format_0 = view->format[0];
    char byteorder;
    
    switch (format_0) {
        
    case '<':
    case '>':
        byteorder = format_0;
        break;
    case '!':
        byteorder = '>';
        break;
    case 'c':
    case 's':
    case 'p':
    case 'b':
    case 'B':
        byteorder = '|';
        break;
    default:
        byteorder = VIEW_MY_ENDIAN;
    }
    return byteorder;
}

static int
_as_arrayinter_flags(const Py_buffer *view, int flags)
{
    int inter_flags = PAI_ALIGNED; /* atomic int types always aligned */
    
    if (!view->readonly) {
        inter_flags |= PAI_WRITEABLE;
    }
    switch (view->format[0]) {
        
    case '<':
        inter_flags |= SDL_BYTEORDER == SDL_LIL_ENDIAN ? PAI_NOTSWAPPED : 0;
        break;
    case '>':
    case '!':
        inter_flags |= SDL_BYTEORDER == SDL_BIG_ENDIAN ? PAI_NOTSWAPPED : 0;
        break;
    default:
        inter_flags |= PAI_NOTSWAPPED;
    }
    if (flags & VIEW_CONTIGUOUS) {
        inter_flags |= PAI_CONTIGUOUS;
    }
    if (flags & VIEW_F_ORDER) {
        inter_flags |= PAI_FORTRAN;
    }
    return inter_flags;
}

static CapsuleInterface *
_new_capsuleinterface(const Py_buffer *view, int flags)
{
    int ndim = view->ndim;
    Py_ssize_t cinter_size;
    CapsuleInterface *cinter_p;
    int i;
    
    cinter_size = (sizeof(CapsuleInterface) +
                   sizeof(Py_intptr_t) * (2 * ndim - 1));
    cinter_p = (CapsuleInterface *)PyMem_Malloc(cinter_size);
    if (!cinter_p) {
        PyErr_NoMemory();
        return 0;
    }
    cinter_p->inter.two = 2;
    cinter_p->inter.nd = ndim;
    cinter_p->inter.typekind = _as_arrayinter_typekind(view);
    cinter_p->inter.itemsize = view->itemsize;
    cinter_p->inter.flags = _as_arrayinter_flags(view, flags);
    if (view->shape) {
        cinter_p->inter.shape = cinter_p->imem;
        for (i = 0; i < ndim; ++i) {
            cinter_p->inter.shape[i] = (Py_intptr_t)view->shape[i];
        }
    }
    if (view->strides) {
        cinter_p->inter.strides = cinter_p->imem + ndim;
        for (i = 0; i < ndim; ++i) {
            cinter_p->inter.strides[i] = (Py_intptr_t)view->strides[i];
        }
    }
    cinter_p->inter.data = view->buf;
    cinter_p->inter.descr = 0;
    cinter_p->parent = (PyObject *)view->obj;
    Py_XINCREF(cinter_p->parent);
    return cinter_p;
}

static void
_free_capsuleinterface(void *p)
{
    CapsuleInterface *cinter_p = (CapsuleInterface *)p;
    
    Py_XDECREF(cinter_p->parent);
    PyMem_Free(p);
}

#if PY3
static void
_capsule_free_capsuleinterface(PyObject *capsule)
{
    _free_capsuleinterface(PyCapsule_GetPointer(capsule, 0));
}
#endif

static PyObject *
_view_get_typestr_obj(Py_buffer *view)
{
    return Text_FromFormat("%c%c%i",
                           _as_arrayinter_byteorder(view),
                           _as_arrayinter_typekind(view),
                           (int)view->itemsize);
}

static PyObject *
_view_get_shape_obj(Py_buffer *view)
{
    PyObject *shapeobj = PyTuple_New(view->ndim);
    PyObject *lengthobj;
    Py_ssize_t i;

    if (!shapeobj) {
        return 0;
    }
    for (i = 0; i < view->ndim; ++i) {
        lengthobj = PyInt_FromLong((long)view->shape[i]);
        if (!lengthobj) {
            Py_DECREF(shapeobj);
            return 0;
        }
        PyTuple_SET_ITEM(shapeobj, i, lengthobj);
    }
    return shapeobj;
}

static PyObject *
_view_get_strides_obj(Py_buffer *view)
{
    PyObject *shapeobj = PyTuple_New(view->ndim);
    PyObject *lengthobj;
    Py_ssize_t i;

    if (!shapeobj) {
        return 0;
    }
    for (i = 0; i < view->ndim; ++i) {
        lengthobj = PyInt_FromLong((long)view->strides[i]);
        if (!lengthobj) {
            Py_DECREF(shapeobj);
            return 0;
        }
        PyTuple_SET_ITEM(shapeobj, i, lengthobj);
    }
    return shapeobj;
}

static PyObject *
_view_get_data_obj(Py_buffer *view)
{
    return Py_BuildValue("NN",
                         PyLong_FromVoidPtr(view->buf),
                         PyBool_FromLong((long)view->readonly));
}

static PyObject *
_shape_as_tuple(PyArrayInterface *inter_p)
{
    PyObject *shapeobj = PyTuple_New((Py_ssize_t)inter_p->nd);
    PyObject *lengthobj;
    Py_ssize_t i;

    if (!shapeobj) {
        return 0;
    }
    for (i = 0; i < inter_p->nd; ++i) {
        lengthobj = PyInt_FromLong((long)inter_p->shape[i]);
        if (!lengthobj) {
            Py_DECREF(shapeobj);
            return 0;
        }
        PyTuple_SET_ITEM(shapeobj, i, lengthobj);
    }
    return shapeobj;
}

static PyObject *
_typekind_as_str(PyArrayInterface *inter_p)
{
    return Text_FromFormat("%c%c%i", 
                           inter_p->itemsize > 1 ?
                               (inter_p->flags & PAI_NOTSWAPPED ?
                                    VIEW_MY_ENDIAN :
                                    VIEW_OTHER_ENDIAN) :
                               '|',
                           inter_p->typekind, inter_p->itemsize);
}

static PyObject *
_strides_as_tuple(PyArrayInterface *inter_p)
{
    PyObject *stridesobj = PyTuple_New((Py_ssize_t)inter_p->nd);
    PyObject *lengthobj;
    Py_ssize_t i;

    if (!stridesobj) {
        return 0;
    }
    for (i = 0; i < inter_p->nd; ++i) {
        lengthobj = PyInt_FromLong((long)inter_p->strides[i]);
        if (!lengthobj) {
            Py_DECREF(stridesobj);
            return 0;
        }
        PyTuple_SET_ITEM(stridesobj, i, lengthobj);
    }
    return stridesobj;
}

static PyObject *
_data_as_tuple(PyArrayInterface *inter_p)
{
    long readonly = (inter_p->flags & PAI_WRITEABLE) == 0;

    return Py_BuildValue("NN",
                         PyLong_FromVoidPtr(inter_p->data),
                         PyBool_FromLong(readonly));
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

    case VIEW_MY_ENDIAN:
        is_swapped = 0;
        break;
    case VIEW_OTHER_ENDIAN:
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
        byteorder = is_swapped ? VIEW_OTHER_ENDIAN : '=';
        type = is_signed ? 'h' : 'H';
        itemsize = 2;
        break;
    case '3':
        type = 's';
        itemsize = 3;
        break;
    case '4':
        byteorder = is_swapped ? VIEW_OTHER_ENDIAN : '=';
        type = is_signed ? 'i' : 'I';
        itemsize = 4;
        break;
    case '6':
        type = 's';
        itemsize = 6;
        break;
    case '8':
        byteorder = is_swapped ? VIEW_OTHER_ENDIAN : '=';
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
_free_bufview(Py_buffer *view)
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
 * Return a new PgViewObject (Python level constructor).
 */
static PyObject *
_view_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Py_buffer bufview;
    PyObject *pyprelude = 0;
    PyObject *pypostscript = 0;
    PyObject *self = 0;
    /* The argument evaluation order is important: strides must follow shape. */
    char *keywords[] = {"shape", "typestr", "data", "strides", "parent",
                        "prelude", "postscript", 0};
               
    bufview.obj = 0;
    bufview.len = 0;
    bufview.readonly = 1;
    bufview.ndim = 0;
    bufview.shape = 0;
    bufview.strides = 0;
    bufview.suboffsets = 0;
    bufview.itemsize = 0;
    bufview.internal = 0;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&|O&O&OO:View", keywords,
                                     _shape_arg_convert, &bufview,
                                     _typestr_arg_convert, &bufview,
                                     _data_arg_convert, &bufview,
                                     _strides_arg_convert, &bufview,
                                     _parent_arg_convert, &bufview,
                                     &pyprelude, &pypostscript)) {
        _free_bufview(&bufview);
        return 0;
    }
    if (pyprelude == Py_None) {
        pyprelude = 0;
    }
    if (pypostscript == Py_None) {
        pypostscript = 0;
    }
    Py_XINCREF((PyObject *)bufview.obj);
    self = _view_new_from_type(type, &bufview, 0,
                               0, 0, pyprelude, pypostscript);
    _free_bufview(&bufview);
    return self;
}

/**
 * Deallocates the PgViewObject and its members.
 */
static void
_view_dealloc(PgViewObject *self)
{
    /* Guard against recursion */
    if (!self->prelude) {
        return;
    }
    self->prelude = 0;
    
    if (self->global_release) {
        self->postscript((PyObject *)self);
    }
    Py_XDECREF((PyObject *)self->bufview.obj);
    Py_XDECREF(self->pyprelude);
    Py_XDECREF(self->pypostscript);
    if (self->weakrefs) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }
    if (self->bufview.shape && self->bufview.shape != self->imem) {
        PyMem_Free(self->bufview.shape);
    }
    if (self->bufview.format && self->bufview.format != self->cmem) {
        PyMem_Free(self->bufview.format);
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
_view_get_arraystruct(PgViewObject *self, PyObject *closure)
{
    void *cinter_p;
    PyObject *capsule;
    
    cinter_p = _new_capsuleinterface(&self->bufview, self->flags);
    if (!cinter_p) {
        return 0;
    }
#if PY3
    capsule = PyCapsule_New(cinter_p, 0, _capsule_free_capsuleinterface);
#else
    capsule = PyCObject_FromVoidPtr(cinter_p, _free_capsuleinterface);
#endif
    if (!capsule) {
        _free_capsuleinterface((void *)cinter_p);
        return 0;
    }
    if (!self->global_release) {
        if (self->prelude((PyObject *)self)) {
            Py_DECREF(capsule);
            capsule = 0;
        }
        else {
            self->global_release = 1;
        }
    }
    return capsule;
}

#if PY3
#else
#endif

static PyObject *
_view_get_arrayinterface(PgViewObject *self, PyObject *closure)
{
    PyObject *dict = Pg_BufferViewAsDict(&self->bufview);
    
    if (dict && !self->global_release) {
        if (self->prelude((PyObject *)self)) {
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
_view_get_parent(PgViewObject *self, PyObject *closure)
{
    PyObject *parent = (PyObject *)self->bufview.obj;
    
    if (!parent) {
        Py_RETURN_NONE;
    }
    Py_INCREF(parent);
    return parent;
}

/**** Methods ****/

/**
 * Representation method.
 */
static PyObject *
_view_repr (PgViewObject *self)
{
    return Text_FromFormat("<%s(%p)>", Py_TYPE(self)->tp_name, self);
}

/**
 * Getters and setters for the PgViewObject.
 */
static PyGetSetDef _view_getsets[] =
{
    {"__array_struct__", (getter)_view_get_arraystruct, 0, 0, 0},
    {"__array_interface__", (getter)_view_get_arrayinterface, 0, 0, 0},
    {"parent", (getter)_view_get_parent, 0, 0, 0},
    {0, 0, 0, 0, 0}
};

#if PY3
static int
_view_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    PgViewObject *v = (PgViewObject *)obj;
    
    if (flags == PyBUF_SIMPLE && !(v->flags & VIEW_CONTIGUOUS)) {
        PyErr_SetString(PyExc_BufferError, "buffer not contiguous");
        return -1;
    }
    if (flags & PyBUF_WRITABLE && v->bufview.readonly) {
        PyErr_SetString(PyExc_BufferError, "buffer is readonly");
        return -1;
    }
    if (flags & PyBUF_ND && !v->bufview.shape) {
        PyErr_SetString(PyExc_BufferError, "buffer shape unavailable");
        return -1;
    }
    if (flags & PyBUF_STRIDES && !v->bufview.strides) {
        PyErr_SetString(PyExc_BufferError, "buffer strides unavailable");
        return -1;
    }
    else if (flags & PyBUF_ND &&
             !(v->flags & (VIEW_CONTIGUOUS | VIEW_C_ORDER))) {
        PyErr_SetString(PyExc_BufferError, "buffer not C contiguous");
        return -1;
    }            
    if (flags & PyBUF_ANY_CONTIGUOUS &&
        !(v->flags & (VIEW_CONTIGUOUS | VIEW_C_ORDER | VIEW_F_ORDER))) {
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
    if (v->prelude(obj)) {
        return -1;
    }
    view->obj = obj;
    Py_INCREF(obj);
    view->buf = v->bufview.buf;
    view->len = v->bufview.len;
    view->readonly = v->bufview.readonly;
    view->format = v->bufview.format;
    view->ndim = v->bufview.ndim;
    view->shape = v->bufview.shape;
    view->strides = v->bufview.strides;
    view->suboffsets = v->bufview.suboffsets;
    view->itemsize = v->bufview.itemsize;
    view->internal = 0;
    return 0;
}

static void
_view_releasebuffer(PyObject *obj, Py_buffer *view)
{
    ((PgViewObject *)obj)->postscript(obj);
}

static PyBufferProcs _view_bufferprocs = 
    {_view_getbuffer, _view_releasebuffer};
#endif

static PyTypeObject PgView_Type =
{
    TYPE_HEAD (NULL, 0)
    "pygame._view.View",        /* tp_name */
    sizeof (PgViewObject),      /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor)_view_dealloc,  /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc)_view_repr,       /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
#if PY3
    &_view_bufferprocs,         /* tp_as_buffer */
#else
    0,                          /* tp_as_buffer */
#endif
    Py_TPFLAGS_DEFAULT,         /* tp_flags */
    "Object view as an array struct\n",
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    offsetof(PgViewObject, weakrefs),    /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    0,                          /* tp_methods */
    0,                          /* tp_members */
    _view_getsets,              /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    0,                          /* tp_init */
    0,                          /* tp_alloc */
    _view_new,                  /* tp_new */
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

static PyObject *
get_array_interface(PyObject *self, PyObject *arg)
{
    PyObject *cobj;
    PyArrayInterface *inter_p;
    PyObject *dictobj;

    if (Pg_GetArrayInterface(arg, &cobj, &inter_p)) {
        return 0;
    }
    dictobj = Pg_ArrayStructAsDict(inter_p);
    Py_DECREF(cobj);
    return dictobj;
}

static PyMethodDef _view_methods[] = {
    { "get_array_interface", get_array_interface, METH_O,
      "return an array struct interface as an interface dictionary" },
    {0, 0, 0, 0}
};

/**** Public C api ***/

static PyObject *
PgView_New(Py_buffer *bufview,
           int flags,
           PgView_PreludeCallback prelude,
           PgView_PostscriptCallback postscript)
{
    return _view_new_from_type(&PgView_Type,
                               bufview,
                               flags,
                               prelude,
                               postscript,
                               0,
                               0);
}

static PyObject *
PgView_GetParent(PyObject *view)
{
    PyObject *parent = (PyObject *)((PgViewObject *) view)->bufview.obj;
    
    if (!parent) {
        parent = Py_None;
    }
    Py_INCREF(parent);
    return parent;
}

static int
Pg_GetArrayInterface(PyObject *obj,
                     PyObject **cobj_p,
                     PyArrayInterface **inter_p)
{
    PyObject *cobj = PyObject_GetAttrString(obj, "__array_struct__");
    PyArrayInterface *inter = NULL;

    if (cobj == NULL) {
        if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
                PyErr_Clear();
                PyErr_SetString(PyExc_ValueError,
                                "no C-struct array interface");
        }
        return -1;
    }

#if PG_HAVE_COBJECT
    if (PyCObject_Check(cobj)) {
        inter = (PyArrayInterface *)PyCObject_AsVoidPtr(cobj);
    }
#endif
#if PG_HAVE_CAPSULE
    if (PyCapsule_IsValid(cobj, NULL)) {
        inter = (PyArrayInterface *)PyCapsule_GetPointer(cobj, NULL);
    }
#endif
    if (inter == NULL || inter->two != 2 /* conditional or */) {
        Py_DECREF(cobj);
        PyErr_SetString(PyExc_ValueError, "invalid array interface");
        return -1;
    }

    *cobj_p = cobj;
    *inter_p = inter;
    return 0;
}

static PyObject *
Pg_ArrayStructAsDict(PyArrayInterface *inter_p)
{
    PyObject *dictobj = Py_BuildValue("{sisNsNsNsN}",
                                      "version", (int)3,
                                      "typestr", _typekind_as_str(inter_p),
                                      "shape", _shape_as_tuple(inter_p),
                                      "strides", _strides_as_tuple(inter_p),
                                      "data", _data_as_tuple(inter_p));

    if (!dictobj) {
        return 0;
    }
    if (inter_p->flags & PAI_ARR_HAS_DESCR) {
        if (!inter_p->descr) {
            Py_DECREF(dictobj);
            PyErr_SetString(PyExc_ValueError,
                            "Array struct has descr flag set"
                            " but no descriptor");
            return 0;
        }
        if (PyDict_SetItemString(dictobj, "descr", inter_p->descr)) {
            Py_DECREF(dictobj);
            return 0;
        }
    }
    return dictobj;
}

static PyObject *
Pg_BufferViewAsDict(Py_buffer *bufview)
{
    PyObject *dictobj = Py_BuildValue("{sisNsNsNsN}",
                                      "version", (int)3,
                                      "typestr", _view_get_typestr_obj(bufview),
                                      "shape", _view_get_shape_obj(bufview),
                                      "strides", _view_get_strides_obj(bufview),
                                      "data", _view_get_data_obj(bufview));
    PyObject *obj = (PyObject *)bufview->obj;

    if (!dictobj) {
        return 0;
    }
    if (obj) {
        if (PyDict_SetItemString(dictobj, "__obj", obj)) {
            Py_DECREF(dictobj);
            return 0;
        }
    }
    return dictobj;
}

/*DOC*/ static char _view_doc[] =
/*DOC*/    "exports View, a generic wrapper object for an array "
/*DOC*/    "struct capsule";

MODINIT_DEFINE(_view)
{
    PyObject *module;
    PyObject *apiobj;
    static void* c_api[PYGAMEAPI_VIEW_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "_view",
        _view_doc,
        -1,
        _view_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    if (PyType_Ready(&PgView_Type) < 0) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "_view", _view_methods, _view_doc);
#endif

    Py_INCREF(&PgView_Type);
    if (PyModule_AddObject(module, "View", (PyObject *)&PgView_Type)) {
        Py_DECREF(&PgView_Type);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
#if PYGAMEAPI_VIEW_NUMSLOTS != 5
#error export slot count mismatch
#endif
    c_api[0] = &PgView_Type;
    c_api[1] = PgView_New;
    c_api[2] = PgView_GetParent;
    c_api[3] = Pg_GetArrayInterface;
    c_api[4] = Pg_ArrayStructAsDict;
    apiobj = encapsulate_api(c_api, "_view");
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
