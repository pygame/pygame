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
#define PYGAME_MATHVECTOR_INTERNAL

#include "pgbase.h"
#include "mathmod.h"
#include "pgmath.h"
#include "mathbase_doc.h"

static int _vector_init (PyObject *self, PyObject *args, PyObject *kwds);
static void _vector_dealloc (PyVector *self);
static PyObject* _vector_repr (PyObject *self);

static PyObject* _vector_get_dimension (PyObject *self, void *closure);
static PyObject* _vector_get_epsilon (PyObject *self, void *closure);
static int _vector_set_epsilon (PyObject *self, PyObject *value, void *closure);
static PyObject* _vector_get_elements (PyObject *self, void *closure);
static int _vector_set_elements (PyObject *self, PyObject *value, void *closure);
static PyObject* _vector_get_length (PyVector *self, void *closure);
static PyObject* _vector_get_length_squared (PyVector *self, void *closure);
static PyObject* _vector_normalize (PyVector *self);
static PyObject* _vector_normalize_ip (PyVector *self);

/* Generic math operations for vectors. */
static PyObject* _vector_generic_math (PyObject *o1, PyObject *o2, int op);
/* Number protocol methods */
static PyObject* _vector_add (PyObject *o1, PyObject *o2);
static PyObject* _vector_sub (PyObject *o1, PyObject *o2);
static PyObject* _vector_mul (PyObject *o1, PyObject *o2);
static PyObject* _vector_div (PyVector *self, PyObject *other);
static PyObject* _vector_floor_div (PyVector *self, PyObject *other);
static PyObject* _vector_inplace_add (PyVector *self, PyObject *other);
static PyObject* _vector_inplace_sub (PyVector *self, PyObject *other);
static PyObject* _vector_inplace_mul (PyVector *self, PyObject *other);
static PyObject* _vector_inplace_div (PyVector *self, PyObject *other);
static PyObject* _vector_inplace_floor_div (PyVector *self, PyObject *other);
static PyObject* _vector_neg (PyVector *self);
static PyObject* _vector_pos (PyVector *self);
static int _vector_nonzero (PyVector *self);

static PyObject* _vector_richcompare (PyObject *o1, PyObject *o2, int op);

/* Sequence protocol methods */
static Py_ssize_t _vector_len (PyVector *self);
static PyObject* _vector_item (PyVector *self, Py_ssize_t _index);
static int _vector_ass_item (PyVector *self, Py_ssize_t _index, PyObject *value);
static PyObject* _vector_slice (PyVector *self, Py_ssize_t ilow,
    Py_ssize_t ihigh);
static int _vector_ass_slice (PyVector *self, Py_ssize_t ilow, Py_ssize_t ihigh,
    PyObject *v);

/* Subscript protocol methods */
static PyObject* _vector_subscript (PyVector *self, PyObject *op);
static int _vector_ass_subscript (PyVector *self, PyObject *op,
    PyObject *value);

/**
 * Methods for the PyVector.
 */
static PyMethodDef _vector_methods[] = {
    { "normalize", (PyCFunction) _vector_normalize, METH_NOARGS,
      DOC_BASE_VECTOR_NORMALIZE },
    { "normalize_ip", (PyCFunction) _vector_normalize_ip, METH_NOARGS,
      DOC_BASE_VECTOR_NORMALIZE_IP },
    { NULL, NULL, 0, NULL },
};

/**
 * Getters and setters for the PyVector.
 */
static PyGetSetDef _vector_getsets[] =
{
    { "dimension", _vector_get_dimension, NULL, DOC_BASE_VECTOR_DIMENSION,
      NULL },
    { "epsilon", _vector_get_epsilon, _vector_set_epsilon,
      DOC_BASE_VECTOR_EPSILON, NULL },
    { "elements", _vector_get_elements, _vector_set_elements,
      DOC_BASE_VECTOR_ELEMENTS, NULL },
    { "length", (getter)_vector_get_length, NULL, DOC_BASE_VECTOR_LENGTH,
      NULL },
    { "length_squared", (getter) _vector_get_length_squared, NULL,
      DOC_BASE_VECTOR_LENGTH, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

static PyNumberMethods _vector_as_number =
{
    (binaryfunc) _vector_add,         /* nb_add;       __add__ */
    (binaryfunc) _vector_sub,         /* nb_subtract;  __sub__ */
    (binaryfunc) _vector_mul,         /* nb_multiply;  __mul__ */
#ifndef IS_PYTHON_3
    (binaryfunc) _vector_div,         /* nb_divide;    __div__ */
#endif
    (binaryfunc)0,                    /* nb_remainder; __mod__ */
    (binaryfunc)0,                    /* nb_divmod;    __divmod__ */
    (ternaryfunc)0,                   /* nb_power;     __pow__ */
    (unaryfunc) _vector_neg,          /* nb_negative;  __neg__ */
    (unaryfunc) _vector_pos,          /* nb_positive;  __pos__ */
    (unaryfunc)0,                     /* nb_absolute;  __abs__ */
    (inquiry) _vector_nonzero,        /* nb_nonzero;   __nonzero__ */
    (unaryfunc) _vector_neg,          /* nb_invert;    __invert__ */
    (binaryfunc)0,                    /* nb_lshift;    __lshift__ */
    (binaryfunc)0,                    /* nb_rshift;    __rshift__ */
    (binaryfunc)0,                    /* nb_and;       __and__ */
    (binaryfunc)0,                    /* nb_xor;       __xor__ */
    (binaryfunc)0,                    /* nb_or;        __or__ */
#ifndef IS_PYTHON_3
    (coercion)0,                      /* nb_coerce;    __coerce__ */
#endif
    (unaryfunc)0,                     /* nb_int;       __int__ */
    (unaryfunc)0,                     /* nb_long;      __long__ */
    (unaryfunc)0,                     /* nb_float;     __float__ */
#ifndef IS_PYTHON_3
    (unaryfunc)0,                     /* nb_oct;       __oct__ */
    (unaryfunc)0,                     /* nb_hex;       __hex__ */
#endif
    (binaryfunc) _vector_inplace_add, /* nb_inplace_add;       __iadd__ */
    (binaryfunc) _vector_inplace_sub, /* nb_inplace_subtract;  __isub__ */
    (binaryfunc) _vector_inplace_mul, /* nb_inplace_multiply;  __imul__ */
#ifndef IS_PYTHON_3
    (binaryfunc) _vector_inplace_div, /* nb_inplace_divide;    __idiv__ */
#endif
    (binaryfunc)0,                    /* nb_inplace_remainder; __imod__ */
    (ternaryfunc)0,                   /* nb_inplace_power;     __pow__ */
    (binaryfunc)0,                    /* nb_inplace_lshift;    __ilshift__ */
    (binaryfunc)0,                    /* nb_inplace_rshift;    __irshift__ */
    (binaryfunc)0,                    /* nb_inplace_and;       __iand__ */
    (binaryfunc)0,                    /* nb_inplace_xor;       __ixor__ */
    (binaryfunc)0,                    /* nb_inplace_or;        __ior__ */
    (binaryfunc) _vector_floor_div,   /* nb_floor_divide;         __floor__ */
    (binaryfunc) _vector_div,         /* nb_true_divide;          __truediv__ */
    (binaryfunc) _vector_inplace_floor_div, /* nb_inplace_floor_divide; __ifloor__ */
    (binaryfunc) _vector_inplace_div, /* nb_inplace_true_divide;  __itruediv__ */
#if PY_VERSION_HEX >= 0x02050000
    (unaryfunc)0,                     /* nb_index */
#endif
};

static PySequenceMethods _vector_as_sequence =
{
    (lenfunc) _vector_len,                    /* sq_length;    __len__ */
    (binaryfunc)0,                            /* sq_concat;    __add__ */
    (ssizeargfunc)0,                          /* sq_repeat;    __mul__ */
    (ssizeargfunc) _vector_item,              /* sq_item;      __getitem__ */
    (ssizessizeargfunc) _vector_slice,        /* sq_slice;     __getslice__ */
    (ssizeobjargproc) _vector_ass_item,       /* sq_ass_item;  __setitem__ */
    (ssizessizeobjargproc) _vector_ass_slice, /* sq_ass_slice; __setslice__ */
    NULL,                                     /* sq_contains */
    NULL,                                     /* sq_inplace_concat */
    NULL,                                     /* sq_inplace_repeat */
};

static PyMappingMethods _vector_as_mapping =
{
    (lenfunc) _vector_len,                 /* mp_length */
    (binaryfunc) _vector_subscript,        /* mp_subscript */
    (objobjargproc) _vector_ass_subscript  /* mp_ass_subscript */
};

PyTypeObject PyVector_Type =
{
    TYPE_HEAD(NULL,0)
    "base.Vector",              /* tp_name */
    sizeof (PyVector),          /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _vector_dealloc,/* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc) _vector_repr,    /* tp_repr */
    &_vector_as_number,         /* tp_as_number */
    &_vector_as_sequence,       /* tp_as_sequence */
    &_vector_as_mapping,        /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    DOC_BASE_VECTOR,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    _vector_richcompare,        /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _vector_methods,            /* tp_methods */
    0,                          /* tp_members */
    _vector_getsets,            /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)_vector_init,     /* tp_init */
    0,                          /* tp_alloc */
    0,                          /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0,                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0                           /* tp_version_tag */
#endif
};

static int
_vector_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *elems = NULL;
    int isseq = 0;
    int dim = 0;
    PyVector *vector = (PyVector *) self;
    
    if (!PyArg_ParseTuple (args, "O", &elems))
        return -1;
    if (PySequence_Check (elems))
    {
        /* The passed argument is a list of vector elements */
        isseq = 1;
        dim = PySequence_Size (elems);
        vector->dim = dim;
    }
    else if (IntFromObj (elems, &dim))
    {
        if (dim < 2)
        {
            PyErr_SetString (PyExc_ValueError,
                "dimension must be greater than 1");
            return -1;
        }
        vector->dim = (Py_ssize_t) dim;
    }
    else
    {
        PyErr_SetString (PyExc_TypeError,
            "argument must be a sequence of vector elements or integer");
        return -1;
    }

    /* Allocate enough space. */
    vector->coords = PyMem_New (double, vector->dim);
    if (!vector->coords)
        return -1;
    if (!isseq)
    {
        Py_ssize_t i;
        for (i = 0; i < vector->dim; i++)
            vector->coords[i] = 0.f;
    }
    else
    {
        double tmp;
        Py_ssize_t i;
        for (i = 0; i < vector->dim; i++)
        {
            if (!DoubleFromSeqIndex (elems, i, &tmp))
            {
                PyMem_Free (vector->coords);
                return -1;
            }
            vector->coords[i] = tmp;
        }
    }
    vector->epsilon = DBL_EPSILON;
    return 0;
}

static void
_vector_dealloc (PyVector *self)
{
    if (self->coords)
    {
        PyMem_Free (self->coords);
        self->coords = NULL;
    }
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static PyObject*
_vector_repr (PyObject *self)
{
    PyVector *v = (PyVector*) self;
    /* TODO */
    return Text_FromFormat ("Vector%dd()", v->dim);
}

/* Vector getters/setters */

/**
 * x = Vector.dimension
 */
static PyObject*
_vector_get_dimension (PyObject *self, void *closure)
{
    return PyInt_FromSsize_t (((PyVector*)self)->dim);
}

/**
 * x = Vector.epsilon
 */
static PyObject*
_vector_get_epsilon (PyObject *self, void *closure)
{
    return PyFloat_FromDouble (((PyVector *)self)->epsilon);
}

/**
 * Vector.epsilon = x
 */
static int
_vector_set_epsilon (PyObject *self, PyObject *value, void *closure)
{
    double eps;
    if (!DoubleFromObj (value, &eps))
        return -1;
    ((PyVector *) self)->epsilon = eps;
    return 0;
}

/**
 * elems = Vector.elements
 */
static PyObject*
_vector_get_elements (PyObject *self, void *closure)
{
    PyVector *v = (PyVector *) self;
    Py_ssize_t i;
    
    PyObject *tuple = PyTuple_New (v->dim);
    if (!tuple)
        return NULL;
    for (i = 0; i < v->dim; i++)
    {
        PyTuple_SET_ITEM (tuple, i, PyFloat_FromDouble (v->coords[i]));
    }
    return tuple;
}

/**
 * Vector.elements = [x, y, z, ...]
 */
static int
_vector_set_elements (PyObject *self, PyObject *value, void *closure)
{
    PyVector *v = (PyVector *) self;
    Py_ssize_t count, i;
    double *tmpcoords;
    double tmp;
    
    if (!PySequence_Check (value))
    {
        PyErr_SetString (PyExc_TypeError, "value must be a sequence");
        return -1;
    }
    tmpcoords = PyMem_New (double, v->dim);
    if (!tmpcoords)
        return -1;
    memcpy (tmpcoords, v->coords, sizeof (double) * v->dim);
    
    count = PySequence_Size (value);
    for (i = 0; i < count && i < v->dim; i++)
    {
        if (!DoubleFromSeqIndex (value, i, &tmp))
        {
            PyMem_Free (tmpcoords);
            return -1;
        }
        tmpcoords[i] = tmp;
    }
    
    PyMem_Free (v->coords);
    v->coords = tmpcoords;
    return 0;
}

/**
 * vector.length
 */
static PyObject*
_vector_get_length (PyVector *self, void *closure)
{
    double length_squared = _ScalarProduct(self->coords, self->coords,
        self->dim);
    return PyFloat_FromDouble (sqrt (length_squared));
}

/**
 * vector.length_squared
 */
static PyObject*
_vector_get_length_squared (PyVector *self, void *closure)
{
    double length_squared = _ScalarProduct(self->coords, self->coords,
        self->dim);
    return PyFloat_FromDouble (length_squared);
}

/* Methods */
static PyObject*
_vector_normalize (PyVector *self)
{
    Py_ssize_t i;
    double length;
    PyVector *ret;
    
    length = sqrt (_ScalarProduct(self->coords, self->coords, self->dim));
    if (length == 0)
    {
        PyErr_SetString (PyExc_ZeroDivisionError,
            "can not normalize vector of length 0");
        return NULL;
    }

    ret = (PyVector *) PyVector_NewSpecialized (self->dim);
    for (i = 0; i < self->dim; ++i)
        ret->coords[i] = self->coords[i] / length;
    return (PyObject *) ret;
}

static PyObject*
_vector_normalize_ip(PyVector *self)
{
    Py_ssize_t i;
    double length;
    PyVector *ret;
    
    length = sqrt (_ScalarProduct(self->coords, self->coords, self->dim));
    if (length == 0)
    {
        PyErr_SetString (PyExc_ZeroDivisionError,
            "can not normalize vector of length 0");
        return NULL;
    }

    for (i = 0; i < self->dim; ++i)
        self->coords[i] /= length;
    Py_RETURN_NONE;
}

static PyObject*
_vector_generic_math (PyObject *o1, PyObject *o2, int op)
{
    PyVector *v, *retval;
    PyObject *other;
    Py_ssize_t dim, otherdim = 0, i;
    
    if (!o1 || !o2)
    {
        PyErr_SetString (PyExc_ValueError, "arguments must not be NULL");
        return NULL;
    }
    
    if (PyVector_Check (o1))
    {
        v = (PyVector *) o1;
        other = o2;
    }
    else if (PyVector_Check (o2))
    {
        v = (PyVector *) o2;
        other = o1;
        op |= OP_ARG_REVERSE;
    }
    else
    {
        PyErr_SetString (PyExc_TypeError, "None of the operands is a Vector");
        return NULL;
    }
    dim = v->dim;
    
    if (PyVector_Check (other))
    {
        op |= OP_ARG_VECTOR;
        otherdim = ((PyVector*) other)->dim;
        /* right-hand vector must have less dimensions than left-hand. */
        if (otherdim > dim)
        {
            PyErr_SetString (PyExc_TypeError,
                "right-hand argument must not have more dimensions than left-hand argument");
            return NULL;
        }
    }
    else if (IsSimpleNumber (other))
        op |= OP_ARG_NUMBER;
    else
        op |= OP_ARG_UNKNOWN;

    switch (op)
    {
    case OP_ADD | OP_ARG_VECTOR:
    case OP_ADD | OP_ARG_VECTOR | OP_ARG_REVERSE:
    {
        retval = (PyVector *) PyVector_NewSpecialized (dim);
        if (!retval)
            return NULL;
        memcpy (retval->coords, v->coords, sizeof (double) * dim);
        for (i = 0; i < otherdim; i++)
            retval->coords[i] += ((PyVector*)other)->coords[i];
        return (PyObject*) retval;
    }
    case OP_IADD | OP_ARG_VECTOR:
    {
        for (i = 0; i < otherdim; i++)
            v->coords[i] += ((PyVector*)other)->coords[i];
        return (PyObject*) v;
    }
    case OP_SUB | OP_ARG_VECTOR:
    case OP_SUB | OP_ARG_VECTOR | OP_ARG_REVERSE:
    {
        retval = (PyVector *) PyVector_NewSpecialized (dim);
        if (!retval)
            return NULL;
        memcpy (retval->coords, v->coords, sizeof (double) * dim);
        for (i = 0; i < otherdim; i++)
            retval->coords[i] -= ((PyVector*)other)->coords[i];
        return (PyObject*) retval;
    }
    case OP_ISUB | OP_ARG_VECTOR:
    {
        for (i = 0; i < otherdim; i++)
            v->coords[i] -= ((PyVector*)other)->coords[i];
        return (PyObject*) v;
    }
    case OP_MUL | OP_ARG_VECTOR:
    case OP_MUL | OP_ARG_VECTOR | OP_ARG_REVERSE:
    {
        double tmp = 0.f;
        for (i = 0; i < otherdim; i++)
            tmp += v->coords[i] * ((PyVector*)other)->coords[i];
        return PyFloat_FromDouble (tmp);
    }
    case OP_MUL | OP_ARG_NUMBER:
    case OP_MUL | OP_ARG_NUMBER | OP_ARG_REVERSE:
    {
        double tmp;
        if (!DoubleFromObj (other, &tmp))
            return NULL;
        
        retval = (PyVector *) PyVector_NewSpecialized (dim);
        if (!retval)
            return NULL;
        for (i = 0; i < dim; i++)
            retval->coords[i] = v->coords[i] * tmp;
        return (PyObject *) retval;
    }
    case OP_IMUL | OP_ARG_NUMBER:
    {
        double tmp;
        if (!DoubleFromObj (other, &tmp))
            return NULL;
        for (i = 0; i < dim; i++)
            v->coords[i] *= tmp;
        return (PyObject *) v;
    }
    case OP_DIV | OP_ARG_NUMBER:
    {
        double tmp;
        if (!DoubleFromObj (other, &tmp))
            return NULL;
        tmp = 1.f / tmp;
        retval = (PyVector *) PyVector_NewSpecialized (dim);
        if (!retval)
            return NULL;
        for (i = 0; i < dim; i++)
            retval->coords[i] = v->coords[i] * tmp;
        return (PyObject *) retval;
    }
    case OP_IDIV | OP_ARG_NUMBER:
    {
        double tmp;
        if (!DoubleFromObj (other, &tmp))
            return NULL;
        tmp = 1. / tmp;
        for (i = 0; i < dim; i++)
            v->coords[i] *= tmp;
        return (PyObject *) v;
    }
    case OP_FLOOR_DIV | OP_ARG_NUMBER:
    {
        double tmp;
        if (!DoubleFromObj (other, &tmp))
            return NULL;
        tmp = 1. / tmp;
        retval = (PyVector *) PyVector_NewSpecialized(dim);
        if (!retval)
            return NULL;
        for (i = 0; i < dim; i++)
            retval->coords[i] = floor (v->coords[i] * tmp);
        return (PyObject *) retval;
    }
    case OP_IFLOOR_DIV | OP_ARG_NUMBER:
    {
        double tmp;
        if (!DoubleFromObj (other, &tmp))
            return NULL;
        tmp = 1. / tmp;
        for (i = 0; i < dim; i++)
            v->coords[i] = floor (v->coords[i] * tmp);
        return (PyObject *) v;
    }
    default:
        Py_INCREF (Py_NotImplemented);
        return Py_NotImplemented;
    }
}

/**
 * vector1 + vector2
 */
static PyObject*
_vector_add (PyObject *o1, PyObject *o2)
{
    return _vector_generic_math (o1, o2, OP_ADD);
}

/**
 * vector1 += vector2
 */
static PyObject*
_vector_inplace_add (PyVector *o1, PyObject *o2)
{
    return _vector_generic_math ((PyObject*)o1, o2, OP_IADD);
}

/**
 * vector1 - vector2
 */
static PyObject*
_vector_sub (PyObject *o1, PyObject *o2)
{
    return _vector_generic_math (o1, o2, OP_SUB);
}

/**
 * vector1 -= vector2
 */
static PyObject*
_vector_inplace_sub (PyVector *o1, PyObject *o2)
{
    return _vector_generic_math ((PyObject*)o1, o2, OP_ISUB);
}

/**
 * vector1 * x
 */
static PyObject*
_vector_mul (PyObject *o1, PyObject *o2)
{
    return _vector_generic_math (o1, o2, OP_MUL);
}

/**
 * vector1 *= x
 */
static PyObject*
_vector_inplace_mul (PyVector *o1, PyObject *o2)
{
    return _vector_generic_math ((PyObject*)o1, o2, OP_IMUL);
}

/**
 * vector1 / x
 */
static PyObject*
_vector_div (PyVector *o1, PyObject *o2)
{
    return _vector_generic_math ((PyObject*)o1, o2, OP_DIV);
}

/**
 * vector1 /= x
 */
static PyObject*
_vector_inplace_div (PyVector *o1, PyObject *o2)
{
    return _vector_generic_math ((PyObject*)o1, o2, OP_IDIV);
}

/**
 * vector1 // x
 */
static PyObject*
_vector_floor_div (PyVector *o1, PyObject *o2)
{
    return _vector_generic_math ((PyObject*)o1, o2, OP_FLOOR_DIV);
}

/**
 * vector1 //= x
 */
static PyObject*
_vector_inplace_floor_div (PyVector *o1, PyObject *o2)
{
    return _vector_generic_math ((PyObject*)o1, o2, OP_IFLOOR_DIV);
}

/**
 * -vector1, ~vector1
 */
static PyObject*
_vector_neg (PyVector *self)
{
    Py_ssize_t i;
    PyVector *ret = (PyVector *) PyVector_NewSpecialized (self->dim);
    if (!ret)
        return NULL;
    for (i = 0; i < self->dim; i++)
        ret->coords[i] = -self->coords[i];
    return (PyObject*) ret;
}

static PyObject *
_vector_pos(PyVector *self)
{
    PyVector *ret = (PyVector*) PyVector_NewSpecialized (self->dim);
    if (!ret)
        return NULL;
    memcpy (ret->coords, self->coords, sizeof (double) * ret->dim);
    return (PyObject *) ret;
}

static int
_vector_nonzero (PyVector *self)
{
    Py_ssize_t i;
    for (i = 0; i < self->dim; i++)
    {
        if (fabs (self->coords[i]) > self->epsilon)
            return 1;
    }
    return 0;
}

/**
 * len (vector)
 */
static Py_ssize_t
_vector_len (PyVector *self)
{
    return (Py_ssize_t) self->dim;
}

/**
 * vector[x]
 */
static PyObject*
_vector_item (PyVector *self, Py_ssize_t _index)
{
    if (_index < 0 || _index >= self->dim)
    {
        PyErr_SetString (PyExc_IndexError, "invalid index");
        return NULL;
    }
    return PyFloat_FromDouble (self->coords[_index]);
}

/**
 * vector[x] = y
 */
static int
_vector_ass_item (PyVector *self, Py_ssize_t _index, PyObject *value)
{
    double tmp;
    if (_index < 0 || _index >= self->dim)
    {
        PyErr_SetString (PyExc_IndexError, "invalid index");
        return -1;
    }
    if (!DoubleFromObj (value, &tmp))
        return -1;
    self->coords[_index] = tmp;
    return 0;
}

/**
 * vector[x,y]
 */
static PyObject*
_vector_slice (PyVector *self, Py_ssize_t ilow, Py_ssize_t ihigh)
{
    /* some code was taken from the CPython source listobject.c */
    PyVector *retval;
    Py_ssize_t len;
    double *src;

    /* make sure boundaries are sane */
    if (ilow < 0)
        ilow = 0;
    else if (ilow > self->dim)
        ilow = self->dim;
    if (ihigh < ilow)
        ihigh = ilow;
    else if (ihigh > self->dim)
        ihigh = self->dim;
    
    len = ihigh - ilow;
    if (len == 0)
    {
        /* Return single number */
        return PyFloat_FromDouble (self->coords[ihigh]);
    }
    
    retval = (PyVector *) PyVector_NewSpecialized (len);
    if (!retval)
        return NULL;
    src = self->coords + ilow;
    memcpy (retval->coords, src, sizeof (double) * len);
    return (PyObject *) retval;
}

static int
_vector_ass_slice(PyVector *self, Py_ssize_t ilow, Py_ssize_t ihigh,
    PyObject *v)
{
    Py_ssize_t i, len;

    if (ilow < 0)
        ilow = 0;
    else if (ilow > self->dim)
        ilow = self->dim;
    if (ihigh < ilow)
        ihigh = ilow;
    else if (ihigh > self->dim)
        ihigh = self->dim;
    
    len = ihigh - ilow;

    if (PyVector_Check (v))
    {
        if (len != ((PyVector *) v)->dim)
        {
            PyErr_SetString (PyExc_ValueError,
                "cannot assign slice of different length");
            return -1;
        }
        memcpy (self->coords + ilow, ((PyVector *)v)->coords,
            sizeof (double) * len);
    }
    else if (PySequence_Check (v))
    {
        double tmp, *tmpcoords;
        
        if (len != PySequence_Length (v))
        {
            PyErr_SetString (PyExc_ValueError,
                "cannot assign slice of different length");
            return -1;
        }
        tmpcoords = PyMem_New (double, self->dim);
        if (!tmpcoords)
            return -1;
        memcpy (tmpcoords, self->coords, sizeof (double) * self->dim);
        
        for (i = 0; i < len; ++i)
        {
            if (!DoubleFromSeqIndex (v, i, &tmp))
            {
                PyMem_Free (tmpcoords);
                return -1;
            }
            tmpcoords[i + ilow] = tmp;
        }
        PyMem_Free (self->coords);
        self->coords = tmpcoords;
    }
    return 0;
}

static PyObject*
_vector_subscript (PyVector *self, PyObject *op)
{
    if (PyIndex_Check (op) || PyInt_Check (op) || PyLong_Check (op))
    {
        Py_ssize_t i;
#if PY_VERSION_HEX >= 0x02050000
        PyObject *val = PyNumber_Index (op);
        if (!val)
            return NULL;
        /* A simple index. */
        i = PyNumber_AsSsize_t (val, PyExc_IndexError);
        Py_DECREF (val);
        if (i == -1 && PyErr_Occurred ())
            return NULL;
#else
        if (!IntFromObj (op, &i))
            return NULL;
#endif
        return _vector_item (self, i);
    }
    /* TODO */
    return NULL;
}

static int
_vector_ass_subscript (PyVector *self, PyObject *op, PyObject *value)
{
    if (PyIndex_Check (op) || PyInt_Check (op) || PyLong_Check (op))
    {
        Py_ssize_t i;
#if PY_VERSION_HEX >= 0x02050000
        PyObject *val = PyNumber_Index (op);
        if (!val)
            return -1;
        /* A simple index. */
        i = PyNumber_AsSsize_t (val, PyExc_IndexError);
        Py_DECREF (val);
        if (i == -1 && PyErr_Occurred ())
            return -1;
#else
        if (!IntFromObj (op, &i))
            return -1;
#endif
        return _vector_ass_item (self, i, value);
    }
    /* TODO */
    return 0;
}

static PyObject*
_vector_richcompare (PyObject *o1, PyObject *o2, int op)
{
    Py_ssize_t i, otherdim;
    double diff;
    PyVector *v = NULL, *v2 = NULL;
    PyObject *other = NULL;
    int swap = 0, retval = 0;
    
    if (PyVector_Check (o1))
    {
        v = (PyVector *) o1;
        if (PyVector_Check (o2))
        {
            v2 = (PyVector *) o2;
            otherdim = v2->dim;
        }
        else if (PySequence_Check (o2))
        {
            other = o2;
            otherdim = PySequence_Size (other);
        }
        else
        {
            Py_INCREF (Py_NotImplemented);
            return Py_NotImplemented;
        }
    }
    else if (PyVector_Check (o2))
    {
        swap = 1;
        v = (PyVector *) o2;
        if (PyVector_Check (o1))
        {
            v2 = (PyVector *) o1;
            otherdim = v2->dim;
        }
        else if (PySequence_Check (o1))
        {
            other = o1;
            otherdim = PySequence_Size (other);
        }
        else
        {
            Py_INCREF (Py_NotImplemented);
            return Py_NotImplemented;
        }
    }
    else
    {
        Py_INCREF (Py_NotImplemented);
        return Py_NotImplemented;
    }

    if (v->dim != otherdim)
        Py_RETURN_FALSE;

    for (i = 0; i < v->dim; i++)
    {
        if (v2)
            diff = v->coords[i] - v2->coords[i];
        else
        {
            double tmp;
            if (!DoubleFromSeqIndex (other, i, &tmp))
                return NULL;
            diff = v->coords[i] - tmp;
        }
        
        if (isnan (diff) || fabs (diff) >= v->epsilon)
        {
            retval = (swap == 1) ? 1 : 0;
            break;
        }
    }
    retval = (swap == 1) ? 0 : 1;
    
    switch (op)
    {
    case Py_EQ:
        return PyBool_FromLong (retval);
    case Py_NE:
        return PyBool_FromLong (!retval);
    default:
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
}

/* C API */
PyObject*
PyVector_New (Py_ssize_t dim)
{
    PyVector *v;
    Py_ssize_t i;
    
    if (dim < 2)
    {
        PyErr_SetString (PyExc_ValueError, "dimension must be greater than 1");
        return NULL;
    }
    
    v = (PyVector*) PyVector_Type.tp_new (&PyVector_Type, NULL, NULL);
    if (!v)
        return NULL;
    v->dim = dim;
    v->epsilon = DBL_EPSILON;
    v->coords = PyMem_New (double, dim);
    if (!v->coords)
    {
        Py_DECREF (v);
        return NULL;
    }
    for (i = 0; i < dim; i++)
        v->coords[i] = 0.f;
    return (PyObject *) v;
}

PyObject*
PyVector_NewFromSeq (PyObject *seq)
{
    Py_ssize_t dim, i;
    PyVector *v;
    double tmp;
    
    if (!seq)
    {
        PyErr_SetString (PyExc_ValueError, "seq must not be NULL");
        return NULL;
    }
    
    if (!PySequence_Check (seq))
    {
        PyErr_SetString (PyExc_TypeError, "seq must be a sequence");
        return NULL;
    }
    
    dim = PySequence_Size (seq);
    v = (PyVector *) PyVector_New (dim);
    if (!v)
        return NULL;
    for (i = 0; i < dim; i++)
    {
        if (!DoubleFromSeqIndex (seq, i, &tmp))
        {
            Py_DECREF (v);
            return NULL;
        }
        v->coords[i] = tmp;
    }
    return (PyObject *) v;
}

PyObject*
PyVector_NewSpecialized (Py_ssize_t dim)
{
    switch (dim)
    {
    case 2:
        return PyVector2_New (0.f, 0.f);
    case 3:
        return PyVector3_New (0.f, 0.f, 0.f);
    default:
        return PyVector_New (dim);
    }
}

void
vector_export_capi (void **capi)
{
    capi[PYGAME_MATHVECTOR_FIRSTSLOT] = &PyVector_Type;
    capi[PYGAME_MATHVECTOR_FIRSTSLOT+1] = &PyVector_New;
    capi[PYGAME_MATHVECTOR_FIRSTSLOT+2] = &PyVector_NewFromSeq;
    capi[PYGAME_MATHVECTOR_FIRSTSLOT+3] = &PyVector_NewSpecialized;
}
