/*
  pygame - Python Game Library

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

#define PYGAMEAPI_MATH_INTERNAL
#include "pygame.h"
#include "pygamedocs.h"
#include "structmember.h"
#include "pgcompat.h"
#include <float.h>
#include <math.h>


static PyTypeObject PyVector2_Type;
#define PyVector2_Check(x) ((x)->ob_type == &PyVector2_Type)
#define PyVector_Check(x) (PyVector2_Check(x))



/********************************
 * Global helper functions
 ********************************/

static int
checkRealNumber(PyObject *obj)
{
    if (PyNumber_Check(obj)) {
        if (!PyComplex_Check(obj)) {
            return 1;
        }
    }
    return 0;
}

static double
PySequence_GetItem_AsDouble(PyObject *seq, Py_ssize_t index)
{
    PyObject *item;
    double value;
    assert(PySequence_Check(seq));
    item = PySequence_GetItem(seq, index);
    value = PyFloat_AsDouble(item);
    Py_XDECREF(item);
    return value;
}

static int 
checkPyVectorCompatible(PyObject *obj, int dim)
{
    int i;
    PyObject *tmp;

    switch(dim) {
    case 2:
        if (PyVector2_Check(obj)) {
            return 1;
        }
        break;
/*
    case 3:
        if (PyVector3d_Check(obj)) {
            return 1;
        }
        break;
    case 4:
        if (PyVector4d_Check(obj)) {
            return 1;
        }
        break;
*/
    default:
        PyErr_SetString(PyExc_SystemError, 
                        "Wrong internal call to checkPyVectorCompatible.");
        return 0;
    }

    if (!PySequence_Check(obj) || (PySequence_Length(obj) != dim)) {
        return 0;
    }

    for (i = 0; i < dim; ++i) {
        tmp = PySequence_GetItem(obj, i);
        if (!checkRealNumber(tmp)) {
            Py_DECREF(tmp);
            return 0;
        }
        Py_DECREF(tmp);
    }
    return 1;
}


static PyMemberDef vector_members[] = {
    {"epsilon", T_DOUBLE, offsetof(PyVector, epsilon), 0,
     "small value used in comparisons"},
    {NULL}  /* Sentinel */
};


static PyObject*
PyVector_NEW(int dim)
{
    PyVector *vec;
    switch (dim) {
    case 2:
        vec = PyObject_New(PyVector, &PyVector2_Type);
        break;
/*
    case 3:
        vec = PyObject_New(PyVector, &PyVector3_Type);
        break;
    case 4:
        vec = PyObject_New(PyVector, &PyVector4_Type);
        break;
*/
    default:
        PyErr_SetString(PyExc_SystemError, 
                        "Wrong internal call to PyVector_NEW.\n");
        return NULL;
    }

    if (vec != NULL) {
        vec->dim = dim;
        vec->epsilon = FLT_EPSILON;
        vec->coords = PyMem_New(double, dim);
        if (vec->coords == NULL) {
            Py_DECREF(vec);
            return PyErr_NoMemory();
        }
    }

    return (PyObject *)vec;
}

static void
vector_dealloc(PyVector* self)
{
    PyMem_Del(self->coords);
    self->ob_type->tp_free((PyObject*)self);
}






/**********************************************
 * Generic vector PyNumber emulation routines
 **********************************************/


static PyObject *
vector_add(PyObject *o1, PyObject *o2)
{
    int i;
    if (PyVector_Check(o1)) {
        int dim = ((PyVector*)o1)->dim;
        if (checkPyVectorCompatible(o2, dim)) {
            PyVector *ret = (PyVector*)PyVector_NEW(dim);
            for (i = 0; i < dim; i++) {
                ret->coords[i] = ((PyVector*)o1)->coords[i] + PySequence_GetItem_AsDouble(o2, i);
            }
            return (PyObject*)ret;
        }
    }
    else {
        int dim = ((PyVector*)o2)->dim;
        if (checkPyVectorCompatible(o1, dim)) {
            PyVector *ret = (PyVector*)PyVector_NEW(dim);
            for (i = 0; i < dim; i++) {
                ret->coords[i] = PySequence_GetItem_AsDouble(o1, i) + ((PyVector*)o2)->coords[i];
            }
            return (PyObject*)ret;
        }
    }
     
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
vector_inplace_add(PyVector *self, PyObject *other)
{
    int i;
    if (checkPyVectorCompatible(other, self->dim)) {
        for (i = 0; i < self->dim; i++) {
            self->coords[i] += PySequence_GetItem_AsDouble(other, i);
        }
        Py_INCREF(self);
        return (PyObject*)self;
    }
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}


static PyObject *
vector_sub(PyObject *o1, PyObject *o2)
{
    int i;
    if (PyVector_Check(o1)) {
        int dim = ((PyVector*)o1)->dim;
        if (checkPyVectorCompatible(o2, dim)) {
            PyVector *ret = (PyVector*)PyVector_NEW(dim);
            for (i = 0; i < dim; i++) {
                ret->coords[i] = ((PyVector*)o1)->coords[i] - PySequence_GetItem_AsDouble(o2, i);
            }
            return (PyObject*)ret;
        }
    }
    else {
        int dim = ((PyVector*)o2)->dim;
        if (checkPyVectorCompatible(o1, dim)) {
            PyVector *ret = (PyVector*)PyVector_NEW(dim);
            for (i = 0; i < dim; i++) {
                ret->coords[i] = PySequence_GetItem_AsDouble(o1, i) - ((PyVector*)o2)->coords[i];
            }
            return (PyObject*)ret;
        }
    }
        
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
vector_inplace_sub(PyVector *self, PyObject *other)
{
    int i;
    if (checkPyVectorCompatible(other, self->dim)) {
        for (i = 0; i < self->dim; i++) {
            self->coords[i] += PySequence_GetItem_AsDouble(other, i);
        }
        Py_INCREF(self);
        return (PyObject*)self;
    }
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
vector_mul(PyObject *o1, PyObject *o2)
{
    int i, row, col, dim;
    if (PyVector_Check(o1)) {
        dim = ((PyVector*)o1)->dim;
        /* vector * vector ? */
        if (checkPyVectorCompatible(o2, dim)) {
            double ret = 0.;
            for (i = 0; i < dim; i++) {
                ret += ((PyVector*)o1)->coords[i] * PySequence_GetItem_AsDouble(o2, i);
            }
            return PyFloat_FromDouble(ret);
        }
        /* vector * scalar ? */
        else if (checkRealNumber(o2)) {
            PyVector *ret = (PyVector*)PyVector_NEW(dim);
            double scalar = PyFloat_AsDouble(o2);
            for (i = 0; i < dim; i++) {
                ret->coords[i] = ((PyVector*)o1)->coords[i] * scalar;
            }
            return (PyObject*)ret;
        }
    }
    else {
        dim = ((PyVector*)o2)->dim;
        /* vector * vector ? */
        if (checkPyVectorCompatible(o1, dim)) {
            double ret = 0.;
            for (i = 0; i < dim; i++) {
                ret += PySequence_GetItem_AsDouble(o1, i) * ((PyVector*)o2)->coords[i];
            }
            return PyFloat_FromDouble(ret);
        }
        /* scalar * vector ? */
        else if (checkRealNumber(o1)) {
            PyVector *ret = (PyVector*)PyVector_NEW(dim);
            double scalar = PyFloat_AsDouble(o1);
            for (i = 0; i < dim; i++) {
                ret->coords[i] = scalar * ((PyVector*)o2)->coords[i];
            }
            return (PyObject*)ret;
        }
    }

    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
vector_inplace_mul(PyVector *self, PyObject *other)
{
    int i, row, col, dim;
    dim = self->dim;
    /* vector * scalar ? */
    if (checkRealNumber(other)) {
        double scalar = PyFloat_AsDouble(other);
        for (i = 0; i < dim; i++) {
            self->coords[i] *= scalar;
        }
        Py_INCREF(self);
        return (PyObject*)self;
    }

    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
vector_div(PyVector *self, PyObject *other)
{
    int i;
    if (checkRealNumber(other)) {
        double tmp = 1. / PyFloat_AsDouble(other);
        PyVector *ret = (PyVector*)PyVector_NEW(self->dim);
        for (i = 0; i < self->dim; i++) {
            ret->coords[i] = self->coords[i] * tmp;
        }
        return (PyObject*)ret;
    }
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
vector_inplace_div(PyVector *self, PyObject *other)
{
    int i;
    if (checkRealNumber(other)) {
        double tmp = 1. / PyFloat_AsDouble(other);
        for (i = 0; i < self->dim; i++) {
            self->coords[i] *= tmp;
        }
        Py_INCREF(self);
        return (PyObject*)self;
    }
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
vector_floor_div(PyVector *self, PyObject *other)
{
    int i;
    if (checkRealNumber(other)) {
        double tmp = 1. / PyFloat_AsDouble(other);
        PyVector *ret = (PyVector*)PyVector_NEW(self->dim);
        for (i = 0; i < self->dim; i++) {
            ret->coords[i] = floor(self->coords[i] * tmp);
        }
        return (PyObject*)ret;
    }
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
vector_inplace_floor_div(PyVector *self, PyObject *other)
{
    int i;
    if (checkRealNumber(other)) {
        double tmp = 1. / PyFloat_AsDouble(other);
        for (i = 0; i < self->dim; i++) {
            self->coords[i] = floor(self->coords[i] * tmp);
        }
        Py_INCREF(self);
        return (PyObject*)self;
    }
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
vector_neg(PyVector *self)
{
    int i;
    PyVector *ret = (PyVector*)PyVector_NEW(self->dim);
    for (i = 0; i < self->dim; i++) {
        ret->coords[i] = -self->coords[i];
    }
    return (PyObject*)ret;
}

static PyObject *
vector_pos(PyVector *self)
{
    int i;
    PyVector *ret = (PyVector*)PyVector_NEW(self->dim);
    memcpy(ret->coords, self->coords, sizeof(ret->coords[0]) * ret->dim);
    return (PyObject*)ret;
}

static int
vector_nonzero(PyVector *self)
{
    int i;
    for (i = 0; i < self->dim; i++) {
        if (fabs(self->coords[i]) > self->epsilon) {
            return 1;
        }
    }
    return 0;
}

PyNumberMethods vector_as_number = {
    (binaryfunc)vector_add,         /* nb_add;       __add__ */
    (binaryfunc)vector_sub,         /* nb_subtract;  __sub__ */
    (binaryfunc)vector_mul,         /* nb_multiply;  __mul__ */
    (binaryfunc)vector_div,         /* nb_divide;    __div__ */
    (binaryfunc)0,                  /* nb_remainder; __mod__ */
    (binaryfunc)0,                  /* nb_divmod;    __divmod__ */
    (ternaryfunc)0,                 /* nb_power;     __pow__ */
    (unaryfunc)vector_neg,          /* nb_negative;  __neg__ */
    (unaryfunc)vector_pos,          /* nb_positive;  __pos__ */
    (unaryfunc)0,                   /* nb_absolute;  __abs__ */
    (inquiry)vector_nonzero,        /* nb_nonzero;   __nonzero__ */
    (unaryfunc)0,                   /* nb_invert;    __invert__ */
    (binaryfunc)0,                  /* nb_lshift;    __lshift__ */
    (binaryfunc)0,                  /* nb_rshift;    __rshift__ */
    (binaryfunc)0,                  /* nb_and;       __and__ */
    (binaryfunc)0,                  /* nb_xor;       __xor__ */
    (binaryfunc)0,                  /* nb_or;        __or__ */
    (coercion)0,                    /* nb_coerce;    __coerce__ */
    (unaryfunc)0,                   /* nb_int;       __int__ */
    (unaryfunc)0,                   /* nb_long;      __long__ */
    (unaryfunc)0,                   /* nb_float;     __float__ */
    (unaryfunc)0,                   /* nb_oct;       __oct__ */
    (unaryfunc)0,                   /* nb_hex;       __hex__ */

    /* Added in release 2.0 */
    (binaryfunc)vector_inplace_add, /* nb_inplace_add;       __iadd__ */
    (binaryfunc)vector_inplace_sub, /* nb_inplace_subtract;  __isub__ */
    (binaryfunc)vector_inplace_mul, /* nb_inplace_multiply;  __imul__ */
    (binaryfunc)vector_inplace_div, /* nb_inplace_divide;    __idiv__ */
    (binaryfunc)0,                  /* nb_inplace_remainder; __imod__ */
    (ternaryfunc)0,                 /* nb_inplace_power;     __pow__ */
    (binaryfunc)0,                  /* nb_inplace_lshift;    __ilshift__ */
    (binaryfunc)0,                  /* nb_inplace_rshift;    __irshift__ */
    (binaryfunc)0,                  /* nb_inplace_and;       __iand__ */
    (binaryfunc)0,                  /* nb_inplace_xor;       __ixor__ */
    (binaryfunc)0,                  /* nb_inplace_or;        __ior__ */

    /* Added in release 2.2 */
    (binaryfunc)vector_floor_div,   /* nb_floor_divide;         __floor__ */
    (binaryfunc)vector_div,         /* nb_true_divide;          __truediv__ */
    (binaryfunc)vector_inplace_floor_div, /* nb_inplace_floor_divide; __ifloor__ */
    (binaryfunc)vector_inplace_div, /* nb_inplace_true_divide;  __itruediv__ */

    /* Added in release 2.5 */
    (unaryfunc)0,                   /* nb_index;  __index__ */
};



/*************************************************
 * Generic vector PySequence emulation routines
 *************************************************/

static Py_ssize_t
vector_len(PyVector *self)
{
    return (Py_ssize_t)self->dim;
}

static PyObject *
vector_GetItem(PyVector *self, Py_ssize_t index)
{
    if (index < 0 || index >= self->dim) {
        PyErr_SetString(PyExc_ValueError, "subscript out of range.");
        Py_RETURN_NONE;
    }
    return PyFloat_FromDouble(self->coords[index]);
}

static int
vector_SetItem(PyVector *self, Py_ssize_t index, PyObject *value)
{
/*
    if (!checkRealNumber(value)) {
        PyErr_SetString(PyExc_TypeError, "cannot assign a non float.");
        return -1;
    }
*/
    if (index < 0 || index >= self->dim) {
        PyErr_SetString(PyExc_TypeError, "subscript out of range.");
        return -1;
    }
    self->coords[index] = PyFloat_AsDouble(value);
    return 0;
}

PySequenceMethods vector_as_sequence = {
    (lenfunc)vector_len,             /* sq_length;    __len__ */
    (binaryfunc)0,                   /* sq_concat;    __add__ */
    (ssizeargfunc)0,                 /* sq_repeat;    __mul__ */
    (ssizeargfunc)vector_GetItem,    /* sq_item;      __getitem__ */
    (ssizessizeargfunc)0,            /* sq_slice;     __getslice__ */
    (ssizeobjargproc)vector_SetItem, /* sq_ass_item;  __setitem__ */
    (ssizessizeobjargproc)0,         /* sq_ass_slice; __setslice__ */
};

static PyObject*
vector_getx (PyVector *self, void *closure)
{
    return PyFloat_FromDouble(self->coords[0]);
}

static int
vector_setx (PyVector *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the x attribute");
        return -1;
    }

    self->coords[0] = PyFloat_AsDouble(value);
    return 0;
}

static PyObject*
vector_gety (PyVector *self, void *closure)
{
    return PyFloat_FromDouble(self->coords[1]);
}

static int
vector_sety (PyVector *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the y attribute");
        return -1;
    }

    self->coords[1] = PyFloat_AsDouble(value);
    return 0;
}



static PyObject *
vector_richcompare(PyVector *self, PyObject *other, int op)
{
    int i;
    if (!checkPyVectorCompatible(other, self->dim)) {
        if (op == Py_EQ)
            Py_RETURN_FALSE;
        else if (op == Py_NE)
            Py_RETURN_TRUE;
        else {
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
    }

    switch(op) {
    case Py_EQ:
        for (i = 0; i < self->dim; i++) {
            if (fabs(self->coords[i] - PySequence_GetItem_AsDouble(other, i)) >= self->epsilon) {
                Py_RETURN_FALSE;
            }
        }
        Py_RETURN_TRUE;
        break;
    case Py_NE:
        for (i = 0; i < self->dim; i++) {
            if (fabs(self->coords[i] - PySequence_GetItem_AsDouble(other, i)) >= self->epsilon) {
                Py_RETURN_TRUE;
            }
        }
        Py_RETURN_FALSE;
        break;
    default:
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
        break;
    }
}







static PyObject *
vector2_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyVector *vec = (PyVector *)type->tp_alloc(type, 0);

    if (vec != NULL) {
        vec->dim = 2;
        vec->epsilon = FLT_EPSILON;
        vec->coords = PyMem_New(double, vec->dim);
        if (vec->coords == NULL) {
            vec->ob_type->tp_free((PyObject*)vec);
            return NULL;
        }
    }

    return (PyObject *)vec;
}

static int
vector2_init(PyVector *self, PyObject *args, PyObject *kwds)
{
    PyObject *xOrSequence=NULL, *y=NULL, *z=NULL;
    static char *kwlist[] = {"x", "y", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist,
                                      &xOrSequence, &y))
        return -1;

    if (xOrSequence) {
        if (checkRealNumber(xOrSequence)) {
            self->coords[0] = PyFloat_AsDouble(xOrSequence);
        } 
        else if (checkPyVectorCompatible(xOrSequence, self->dim)) {
            self->coords[0] = PySequence_GetItem_AsDouble(xOrSequence, 0);
            self->coords[1] = PySequence_GetItem_AsDouble(xOrSequence, 1);
            return 0;
        } 
        else {
            PyErr_SetString(PyExc_ValueError,
                            "Vector2d must be initialized with 2 real numbers or a sequence of 2 real numbers");
            return -1;
        }
    } 
    else {
        self->coords[0] = 0.;
    }

    if (y) {
        if (checkRealNumber(y)) {
            self->coords[1] = PyFloat_AsDouble(y);
        } 
        else {
            PyErr_SetString(PyExc_ValueError,
                            "Vector2d must be initialized with 2 real numbers or a sequence of 2 real numbers");
            return -1;
        }
    } 
    else {
        self->coords[1] = 0.;
    }

    return 0;
}


static PyMethodDef vector2_methods[] = {
/*
    {"rotate", (PyCFunction)PyVector2d_rotate, METH_VARARGS,
     "rotates the vector around the given axis by the amount given by angle."
    },
    {"rotated", (PyCFunction)PyVector2d_rotated, METH_VARARGS,
     "returns a vector rotated around the given axis by the amount given by angle."
    },
    {"normalized", (PyCFunction)PyVectorNd_normalized, METH_NOARGS,
     "returns a vector that has length == 1 and the same direction as self."
    },
    {"normalize", (PyCFunction)PyVectorNd_normalize, METH_NOARGS,
     "Normalizes the vector so that it has length == 1."
    },
    {"isNormalized", (PyCFunction)PyVectorNd_isNormalized, METH_NOARGS,
     "returns True if the vector has length == 1. otherwise it returns False."
    },
    {"cross", (PyCFunction)PyVector2d_cross, METH_O,
     "calculates the cross product."
    },
    {"dot", (PyCFunction)PyVectorNd_dot, METH_O,
     "calculates the dot product."
    },
    {"angleTo", (PyCFunction)PyVector2d_angleTo, METH_O,
     "returns the angle between self and the given vector."
    },
*/
    {NULL}  /* Sentinel */
};

static PyObject *
vector2_repr(PyVector *self)
{
    return PyString_FromFormat("<Vector2d(%g, %g)>", 
                               self->coords[0], self->coords[1]);
}

static PyObject *
vector2_str(PyVector *self)
{
    return PyString_FromFormat("(%g, %g)", 
                               self->coords[0], self->coords[1]);
}


static PyGetSetDef vector2_getsets[] = {
    { "x", (getter)vector_getx, (setter)vector_setx, NULL, NULL },
    { "y", (getter)vector_gety, (setter)vector_sety, NULL, NULL },
    { NULL, 0, NULL, NULL, NULL }  /* Sentinel */
};


/********************************
 * PyVector2d type definition
 ********************************/

static PyTypeObject PyVector2_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "pygame.math.Vector2",         /*tp_name*/
    sizeof(PyVector),          /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    /* Methods to implement standard operations */
    (destructor)vector_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)vector2_repr,   /*tp_repr*/
    /* Method suites for standard classes */
    &vector_as_number,       /*tp_as_number*/
    &vector_as_sequence,     /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    /* More standard operations (here for binary compatibility) */
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    (reprfunc)vector2_str,    /*tp_str*/
    PyObject_GenericGetAttr,   /*tp_getattro*/
    0,                         /*tp_setattro*/
    /* Functions to access object as input/output buffer */
    0,                         /*tp_as_buffer*/
    /* Flags to define presence of optional/expanded features */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES | Py_TPFLAGS_HAVE_INPLACEOPS, /*tp_flags*/
    /* Documentation string */
    DOC_PYGAMEVECTOR2,               /* tp_doc */

    /* Assigned meaning in release 2.0 */
    /* call function for all accessible objects */
    0,                         /* tp_traverse */
    /* delete references to contained objects */
    0,                         /* tp_clear */

    /* Assigned meaning in release 2.1 */
    /* rich comparisons */
    (richcmpfunc)vector_richcompare, /* tp_richcompare */
    /* weak reference enabler */
    0,                         /* tp_weaklistoffset */

    /* Added in release 2.2 */
    /* Iterators */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    /* Attribute descriptor and subclassing stuff */
    vector2_methods,          /* tp_methods */
    vector_members,          /* tp_members */
    vector2_getsets,        /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)vector2_init,   /* tp_init */
    0,                         /* tp_alloc */
    (newfunc)vector2_new,     /* tp_new */
    0,                         /* tp_free */
    0,                         /* tp_is_gc */
    0,                         /* tp_bases */
    0,                         /* tp_mro */
    0,                         /* tp_cache */
    0,                         /* tp_subclasses */
    0,                         /* tp_weaklist */
};


static PyMethodDef _math_methods[] =
{
    {NULL, NULL, 0, NULL}
};

/* DOC */ static char _math_doc[] = 
/* DOC */    "Module for various math related classes and functions\n";

/****************************
 * Module init function
 ****************************/

MODINIT_DEFINE (math)
{
    PyObject *module, *apiobj;
    static void *c_api[PYGAMEAPI_MATH_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "math",
        _math_doc,
        -1,
        _math_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    /* initialize the extension types */
    if ((PyType_Ready(&PyVector2_Type) < 0) /*|| 
        (PyType_Ready(&PyVector3_Type) < 0) ||
        (PyType_Ready(&PyVector4_Type) < 0)*/) {
        MODINIT_ERROR;
    }

    /* initialize the module */
#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 (MODPREFIX "math", _math_methods, _math_doc);
#endif

    if (module == NULL) {
        MODINIT_ERROR;
    }

    /* add extension types to module */
    Py_INCREF(&PyVector2_Type);
//    Py_INCREF(&PyVector3_Type);
//    Py_INCREF(&PyVector4_Type);
    if ((PyModule_AddObject(module, "Vector2", (PyObject *)&PyVector2_Type) != 0) /*||
        (PyModule_AddObject(module, "Vector3", (PyObject *)&PyVector3_Type) != 0) ||
        (PyModule_AddObject(module, "Vector4", (PyObject *)&PyVector4_Type) != 0)*/) {
        Py_DECREF(&PyVector2_Type);
//        Py_DECREF(&PyVector3_Type);
//        Py_DECREF(&PyVector4_Type);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    /* export the C api */
    c_api[0] = &PyVector2_Type;
//    c_api[1] = &PyVector3_Type;
//    c_api[2] = &PyVector4_Type;
//    c_api[3] = PyVector_NEW;
//    c_api[4] = checkPyVectorCompatible;
    apiobj = PyCObject_FromVoidPtr(c_api, NULL);
    if (apiobj == NULL) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj) != 0) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    MODINIT_RETURN (module);
}
