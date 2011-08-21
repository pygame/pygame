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
  This module exports an object which provides a C level interface to
  another object's buffer. For now only an array structure -
  __array_struct__ - interface is exposed. When memoryview is ready it
  will replace the View object.
 */

#define NO_PYGAME_C_API
#include "pygame.h"
#include "pgcompat.h"
#include "pgview.h"

typedef struct {
    PyObject_HEAD
    PyObject *capsule;            /* Wrapped array struct            */
    PyObject *parent;             /* Object responsible for the view */
    PgView_Destructor destructor; /* Optional release callback       */
    PyObject *pydestructor;       /* Python callable destructor      */
    PyObject *weakrefs;           /* There can be reference cycles   */
} PgViewObject;

static int Pg_GetArrayInterface(PyObject *, PyObject **, PyArrayInterface **);
static PyObject *Pg_ArrayStructAsDict(PyArrayInterface *);

/**
 * Helper functions.
 */
static void
_view_default_destructor(PyObject *view) {
    PgViewObject *v = (PgViewObject *)view;
    PyObject *rvalue;

    if (v->pydestructor) {
        rvalue = PyObject_CallFunctionObjArgs(v->pydestructor,
                                              v->capsule,
                                              v->parent,
                                              0);
        PyErr_Clear();
        Py_XDECREF(rvalue);
    }
}

static PyObject *
_view_new_from_type(PyTypeObject *type,
                    PyObject *capsule,
                    PyObject *parent,
                    PgView_Destructor destructor,
                    PyObject *pydestructor)
{
    PgViewObject *self = (PgViewObject *)type->tp_alloc(type, 0);

    if (!self) {
        return 0;
    }
    self->weakrefs = 0;
    Py_INCREF(capsule);
    self->capsule = capsule;
    if (!parent) {
        parent = Py_None;
    }
    Py_INCREF(parent);
    self->parent = parent;
    self->destructor = destructor ? destructor : _view_default_destructor;
    Py_XINCREF(pydestructor);
    self->pydestructor = pydestructor;
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

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define MY_ENDIAN '<'
#define OTHER_ENDIAN '>'
#else
#define MY_ENDIAN '>'
#define OTHER_ENDIAN '<'
#endif

static PyObject *
_typekind_as_str(PyArrayInterface *inter_p)
{
    return Text_FromFormat("%c%c%i", 
                           inter_p->flags & PAI_NOTSWAPPED ?
                           MY_ENDIAN : OTHER_ENDIAN,
                           inter_p->typekind, inter_p->itemsize);
}

#undef MY_ENDIAN
#undef OTHER_ENDIAN

static PyObject *_strides_as_tuple(PyArrayInterface *inter_p)
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

static PyObject *_data_as_tuple(PyArrayInterface *inter_p)
{
    long readonly = (inter_p->flags & PAI_WRITEABLE) == 0;

    return Py_BuildValue("NN",
                         PyLong_FromVoidPtr(inter_p->data),
                         PyBool_FromLong(readonly));
}

/**
 * Creates a new PgViewObject.
 */
static PyObject *
_view_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *capsule;
    PyObject *parent = 0;
    PyObject *pydestructor = 0;
    char *keywords[] = {"capsule", "parent", "destructor", 0};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO:View", keywords,
                                     &capsule, &parent, &pydestructor)) {
        return 0;
    }
    if (pydestructor == Py_None) {
        pydestructor = 0;
    }
    return _view_new_from_type(type,
                               capsule,
                               parent,
                               0,
                               pydestructor);
}

/**
 * Deallocates the PgViewObject and its members.
 */
static void
_view_dealloc(PgViewObject *self)
{
    PgView_Destructor destructor = self->destructor;

    /* Guard against recursion */
    self->destructor = 0;
    if (!destructor) {
        return;
    }

    destructor((PyObject *)self);
    Py_DECREF(self->capsule);
    Py_DECREF(self->parent);
    Py_XDECREF(self->pydestructor);
    if (self->weakrefs) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }
    Py_TYPE(self)->tp_free(self);
}

/**** Getter and setter access ****/

static PyObject *
_view_get_arraystruct(PgViewObject *self, PyObject *closure)
{
    Py_INCREF(self->capsule);
    return self->capsule;
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
    return Text_FromFormat("<%s(%p)>", Py_TYPE(self)->tp_name, self->capsule);
}

/**
 * Getters and setters for the PgViewObject.
 */
static PyGetSetDef _view_getsets[] =
{
    {"__array_struct__", (getter)_view_get_arraystruct, 0, 0, 0},
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
PgView_New(PyObject *capsule,
           PyObject *parent,
           PgView_Destructor destructor)
{
    if (!capsule) {
        PyErr_SetString(PyExc_TypeError, "the capsule argument is required");
        return 0;
    }
    return _view_new_from_type(&PgView_Type, capsule, parent, destructor, 0);
}

static PyObject *
PgView_GetCapsule(PyObject *view)
{
    PyObject *capsule = ((PgViewObject *)view)->capsule;

    Py_INCREF(capsule);
    return capsule;
}

static PyObject *
PgView_GetParent(PyObject *view)
{
    PyObject *parent = ((PgViewObject *)view)->parent;

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
#if PYGAMEAPI_VIEW_NUMSLOTS != 6
#error export slot count mismatch
#endif
    c_api[0] = &PgView_Type;
    c_api[1] = PgView_New;
    c_api[2] = PgView_GetCapsule;
    c_api[3] = PgView_GetParent;
    c_api[4] = Pg_GetArrayInterface;
    c_api[5] = Pg_ArrayStructAsDict;
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
