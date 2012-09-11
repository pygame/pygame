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

static int Pg_GetArrayInterface(PyObject *, PyObject **, PyArrayInterface **);
static PyObject *Pg_ArrayStructAsDict(PyArrayInterface *);

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
    int failed = 0;
    
    rvalue = PyObject_CallFunctionObjArgs(v->pyprelude, v->parent, 0);
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

    rvalue = PyObject_CallFunctionObjArgs(v->pypostscript, v->parent, 0);
    PyErr_Clear();
    Py_XDECREF(rvalue);
}

static PyObject *
_view_new_from_type(PyTypeObject *type,
                    PyArrayInterface *inter_p,
                    PyObject *parent,
                    PgView_PreludeCallback prelude,
                    PgView_PostscriptCallback postscript,
                    PyObject *pyprelude,
                    PyObject *pypostscript)
{
    int nd = inter_p->nd;
    PgViewObject *self;
    Py_intptr_t *intmem;
    int i;

    if (inter_p->two != 2) {
        PyErr_SetString(PyExc_SystemError,
                        "pygame: _view_new_from_type:"
                        " array interface two.");
        return 0;
    }
    if ((inter_p->flags & PAI_ARR_HAS_DESCR) && !inter_p->descr) {
        PyErr_SetString(PyExc_SystemError,
                        "pygame: _view_new_from_type:"
                        " array interface descr");
        return 0;
    }
    
    intmem = PyMem_New(Py_intptr_t, (inter_p->strides ? 2 : 1) * nd);
    if (!intmem) {
        return PyErr_NoMemory();
    }
    
    self = (PgViewObject *)type->tp_alloc(type, 0);
    if (!self) {
        PyMem_Free(intmem);
        return 0;
    }
    
    self->weakrefs = 0;
    self->inter.two = 2;
    self->inter.nd = nd;
    self->inter.typekind = inter_p->typekind;
    self->inter.itemsize = inter_p->itemsize;
    self->inter.flags = inter_p->flags;
    self->inter.shape = intmem;
    for (i = 0; i < nd; ++i) {
        intmem[i] = inter_p->shape[i];
    }
    if (inter_p->strides) {
        intmem += nd;
        self->inter.strides = intmem;
        for (i = 0; i < nd; ++i) {
            intmem[i] = inter_p->strides[i];
        }
    }
    else {
        inter_p->strides = 0;
    }
    self->inter.data = inter_p->data;
    if (inter_p->flags & PAI_ARR_HAS_DESCR) {
        Py_INCREF(inter_p->descr);
        self->inter.descr = inter_p->descr;
    }
    else {
        self->inter.descr = 0;
    }
    if (!parent) {
        parent = Py_None;
    }
    Py_INCREF(parent);
    self->parent = parent;
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
    PyArrayInterface *inter_p = (PyArrayInterface *)a;
    
    if (!_tuple_as_ints(o, "shape", &(inter_p->shape), &(inter_p->nd))) {
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
    PyArrayInterface *inter_p = (PyArrayInterface *)a;
    int flags = inter_p->flags;
    char typekind;
    int itemsize;
    PyObject *s;
    const char *typestr;
    
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
        flags |= PAI_NOTSWAPPED;
        break;
    case VIEW_OTHER_ENDIAN:
        break;
    case '|':
        break;
    default:
        PyErr_Format(PyExc_ValueError,
                     "unknown byteorder character %c in typestr",
                     typestr[0]);
        Py_DECREF(s);
        return 0;
    }
    typekind = typestr[1];
    switch (typekind) {

    case 'i':
        break;
    case 'u':
        break;
    default:
        PyErr_Format(PyExc_ValueError,
                     "unsupported typekind %c in typestr",
                     typekind);
        Py_DECREF(s);
        return 0;
    }
    switch (typestr[2]) {

    case '1':
        itemsize = 1;
        break;
    case '2':
        itemsize = 2;
        break;
    case '3':
        itemsize = 3;
        break;
    case '4':
        itemsize = 4;
        break;
    case '6':
        itemsize = 6;
        break;
    case '8':
        itemsize = 8;
        break;
    default:
        PyErr_Format(PyExc_ValueError,
                     "unsupported size %c in typestr",
                     typestr[2]);
        Py_DECREF(s);
        return 0;
    }
    inter_p->typekind = typekind;
    inter_p->itemsize = itemsize;
    inter_p->flags = flags;
    return 1;
}

static int
_strides_arg_convert(PyObject *o, void *a)
{
    /* Must be called after the array interface nd field has been filled in.
     */
    PyArrayInterface *inter_p = (PyArrayInterface *)a;
    int n = 0;

    if (o == Py_None) {
        return 1; /* no strides (optional) given */
    }
    if (!_tuple_as_ints(o, "strides", &(inter_p->strides), &n)) {
        return 0;
    }
    if (n != inter_p->nd) {
        PyErr_SetString(PyExc_TypeError,
                        "strides and shape tuple lengths differ");
        return 0;
    }
    return 1;
}

static int
_data_arg_convert(PyObject *o, void *a)
{
    PyArrayInterface *inter_p = (PyArrayInterface *)a;
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
    inter_p->data = (void *)address;
    inter_p->flags |= readonly ? 0 : PAI_WRITEABLE;
    return 1;
}

static void
_free_inter(PyArrayInterface *inter_p)
{
    if (inter_p->shape) {
        PyMem_Free(inter_p->shape);
    }
    if (inter_p->strides) {
        PyMem_Free(inter_p->strides);
    }
}

/**
 * Return a new PgViewObject (Python level constructor).
 */
static PyObject *
_view_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyArrayInterface inter = {0, 0, '\0', 0, 0, 0, 0, 0, 0};
    PyObject *parent = 0;
    PyObject *pyprelude = 0;
    PyObject *pypostscript = 0;
    void *inter_vp = (void *)&inter;
    PyObject *self = 0;
    /* The argument evaluation order is important: strides must follow shape. */
    char *keywords[] = {"shape", "typestr", "data", "strides", "parent",
                        "prelude", "postscript", 0};
                        
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&|O&OOO:View", keywords,
                                     _shape_arg_convert, inter_vp,
                                     _typestr_arg_convert, inter_vp,
                                     _data_arg_convert, inter_vp,
                                     _strides_arg_convert, inter_vp,
                                     &parent, &pyprelude, &pypostscript)) {
        _free_inter(&inter);
        return 0;
    }
    if (parent == Py_None) {
        parent = 0;
    }
    if (pyprelude == Py_None) {
        pyprelude = 0;
    }
    if (pypostscript == Py_None) {
        pypostscript = 0;
    }
    inter.two = 2;
    self = _view_new_from_type(type,
                               &inter,
                               parent,
                               0,
                               0,
                               pyprelude,
                               pypostscript);
    _free_inter(&inter);
    return self;
}

/**
 * Deallocates the PgViewObject and its members.
 */
static void
_view_dealloc(PgViewObject *self)
{
    /* Guard against recursion */
    if (self->inter.two == 0) {
        return;
    }
    self->inter.two = 0;
    
    if (self->global_release) {
        self->postscript((PyObject *)self);
    }
    Py_DECREF(self->parent);
    Py_XDECREF(self->pyprelude);
    Py_XDECREF(self->pypostscript);
    if (self->weakrefs) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }
    PyMem_Free(self->inter.shape);
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
    PyObject *capsule = Capsule_New(&self->inter);

    if (capsule && !self->global_release /* conditional && */ ) {
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

static PyObject *
_view_get_arrayinterface(PgViewObject *self, PyObject *closure)
{
    PyObject *dict = Pg_ArrayStructAsDict(&self->inter);
    
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
    if (!self->parent) {
        Py_RETURN_NONE;
    }
    Py_INCREF(self->parent);
    return self->parent;
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
    0,                          /* tp_as_buffer */
    (Py_TPFLAGS_HAVE_WEAKREFS | Py_TPFLAGS_BASETYPE |
     Py_TPFLAGS_HAVE_CLASS),
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
PgView_New(PyArrayInterface *inter_p,
           PyObject *parent,
           PgView_PreludeCallback prelude,
           PgView_PostscriptCallback postscript)
{
    return _view_new_from_type(&PgView_Type,
                               inter_p,
                               parent,
                               prelude,
                               postscript,
                               0,
                               0);
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
    if (inter == NULL ||   /* conditional or */
        inter->two != 2  ) {
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
#if PYGAMEAPI_VIEW_NUMSLOTS != 4
#error export slot count mismatch
#endif
    c_api[0] = &PgView_Type;
    c_api[1] = PgView_New;
    c_api[2] = Pg_GetArrayInterface;
    c_api[3] = Pg_ArrayStructAsDict;
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
