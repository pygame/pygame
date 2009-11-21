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
#include "doc/math_doc.h"
#include "pygame.h"
#include "structmember.h"
#include "pgcompat.h"
#include <float.h>
#include <math.h>
#include <stddef.h>

/* on some windows platforms math.h doesn't define M_PI */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define VECTOR_MAX_SIZE (4)
#define STRING_BUF_SIZE (100)
#define SWIZZLE_ERR_NO_ERR         0
#define SWIZZLE_ERR_DOUBLE_IDX     1
#define SWIZZLE_ERR_EXTRACTION_ERR 2
#define RETURN_ERROR 0
#define RETURN_NO_ERROR 1

#define OP_ADD          1
#define OP_IADD         2
#define OP_SUB          3
#define OP_ISUB         4
#define OP_MUL          5
#define OP_IMUL         6
#define OP_DIV          7
#define OP_IDIV         8
#define OP_FLOOR_DIV    9
#define OP_IFLOOR_DIV  10
#define OP_MOD         11
#define OP_ARG_REVERSE 32
#define OP_ARG_UNKNOWN 64
#define OP_ARG_VECTOR 128
#define OP_ARG_NUMBER 256

static PyTypeObject PyVector2_Type;
static PyTypeObject PyVector3_Type;
static PyTypeObject PyVectorElementwiseProxy_Type;
static PyTypeObject PyVectorIter_Type;
static PyTypeObject PyVector_SlerpIter_Type;

#define PyVector2_Check(x) (Py_TYPE(x) == &PyVector2_Type)
#define PyVector3_Check(x) (Py_TYPE(x) == &PyVector3_Type)
#define PyVector_Check(x) (PyVector2_Check(x) || PyVector3_Check(x))
#define vector_elementwiseproxy_Check(x) \
    (Py_TYPE(x) == &PyVectorElementwiseProxy_Type)

#define DEG2RAD(angle) ((angle) * M_PI / 180.)
#define RAD2DEG(angle) ((angle) * 180. / M_PI)

typedef struct
{
    PyObject_HEAD
    double *coords;     /* Coordinates */
    unsigned int dim;   /* Dimension of the vector */
    double epsilon;     /* Small value for comparisons */
} PyVector;

typedef struct {
    PyObject_HEAD
    long it_index;
    PyVector *vec;
} vectoriter;

typedef struct {
    PyObject_HEAD
    long it_index;
    long steps;
    long dim;
    double coords[VECTOR_MAX_SIZE];
    double matrix[VECTOR_MAX_SIZE][VECTOR_MAX_SIZE];
    double radial_factor;
} vector_slerpiter;

typedef struct {
    PyObject_HEAD
    long it_index;
    long steps;
    long dim;
    double coords[VECTOR_MAX_SIZE];
    double step_vec[VECTOR_MAX_SIZE];
} vector_lerpiter;

typedef struct {
    PyObject_HEAD
    PyVector *vec;
} vector_elementwiseproxy;


/* further forward declerations */
/* generic helper functions */
static int RealNumber_Check(PyObject *obj);
static double PySequence_GetItem_AsDouble(PyObject *seq, Py_ssize_t index);
static int PySequence_AsVectorCoords(PyObject *seq, double *coords, const size_t size);
static int PyVectorCompatible_Check(PyObject *obj, int dim);
static double _scalar_product(const double *coords1, const double *coords2, int size);
static void _make_vector2_slerp_matrix(vector_slerpiter *it, 
                                       const double *vec1_coords,
                                       const double *vec2_coords,
                                       double angle);
static void _make_vector3_slerp_matrix(vector_slerpiter *it, 
                                       const double *vec1_coords,
                                       const double *vec2_coords,
                                       double angle);

/* generic vector functions */
static PyObject *PyVector_NEW(int dim);
static void vector_dealloc(PyVector* self);
static PyObject *vector_generic_math(PyObject *o1, PyObject *o2, int op);
static PyObject *vector_add(PyObject *o1, PyObject *o2);
static PyObject *vector_sub(PyObject *o1, PyObject *o2);
static PyObject *vector_mul(PyObject *o1, PyObject *o2);
static PyObject *vector_div(PyVector *self, PyObject *other);
static PyObject *vector_floor_div(PyVector *self, PyObject *other);
static PyObject *vector_inplace_add(PyVector *self, PyObject *other);
static PyObject *vector_inplace_sub(PyVector *self, PyObject *other);
static PyObject *vector_inplace_mul(PyVector *self, PyObject *other);
static PyObject *vector_inplace_div(PyVector *self, PyObject *other);
static PyObject *vector_inplace_floor_div(PyVector *self, PyObject *other);
static PyObject *vector_neg(PyVector *self);
static PyObject *vector_pos(PyVector *self);
static int vector_nonzero(PyVector *self);
static Py_ssize_t vector_len(PyVector *self);
static PyObject *vector_GetItem(PyVector *self, Py_ssize_t index);
static int vector_SetItem(PyVector *self, Py_ssize_t index, PyObject *value);
static PyObject *vector_GetSlice(PyVector *self, Py_ssize_t ilow, Py_ssize_t ihigh);
static int vector_SetSlice(PyVector *self, Py_ssize_t ilow, Py_ssize_t ihigh, PyObject *v);
static PyObject *vector_getx (PyVector *self, void *closure);
static PyObject *vector_gety (PyVector *self, void *closure);
static PyObject *vector_getz (PyVector *self, void *closure);
static PyObject *vector_getw (PyVector *self, void *closure);
static int vector_setx (PyVector *self, PyObject *value, void *closure);
static int vector_sety (PyVector *self, PyObject *value, void *closure);
static int vector_setz (PyVector *self, PyObject *value, void *closure);
static int vector_setw (PyVector *self, PyObject *value, void *closure);
static PyObject *vector_richcompare(PyVector *self, PyObject *other, int op);
static PyObject *vector_length(PyVector *self);
static PyObject *vector_length_squared(PyVector *self);
static PyObject *vector_normalize(PyVector *self);
static PyObject *vector_normalize_ip(PyVector *self);
static PyObject *vector_dot(PyVector *self, PyObject *other);
static PyObject *vector_scale_to_length(PyVector *self, PyObject *length);
static int _vector_reflect_helper(double *dst_coords, const double *src_coords, 
                                  PyObject *normal, int dim, double epsilon);
static PyObject *vector_reflect(PyVector *self, PyObject *normal);
static PyObject *vector_reflect_ip(PyVector *self, PyObject *normal);
static PyObject *vector_distance_to(PyVector *self, PyObject *other);
static PyObject *vector_distance_squared_to(PyVector *self, PyObject *other);
static PyObject *vector_getAttr_swizzle(PyVector *self, PyObject *attr_name);
static int vector_setAttr_swizzle(PyVector *self, PyObject *attr_name, PyObject *val);
static PyObject *vector_elementwise(PyVector *self);
static PyObject *vector_slerp(PyVector *self, PyObject *args);
static PyObject *vector_lerp(PyVector *self, PyObject *args);
static PyObject *vector_repr(PyVector *self);
static PyObject *vector_str(PyVector *self);
/*
static Py_ssize_t vector_readbuffer(PyVector *self, Py_ssize_t segment, void **ptrptr);
static Py_ssize_t vector_writebuffer(PyVector *self, Py_ssize_t segment, void **ptrptr);
static Py_ssize_t vector_segcount(PyVector *self, Py_ssize_t *lenp);
*/

/* vector2 specific functions */
static PyObject *vector2_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int vector2_init(PyVector *self, PyObject *args, PyObject *kwds);
static void _vector2_do_rotate(double *dst_coords, const double *src_coords,
                               double angle, double epsilon);
static PyObject *vector2_rotate(PyVector *self, PyObject *args);
static PyObject *vector2_rotate_ip(PyVector *self, PyObject *args);
static PyObject *vector2_cross(PyVector *self, PyObject *other);
static PyObject *vector2_angle_to(PyVector *self, PyObject *other);
static PyObject *vector2_as_polar(PyVector *self);
static PyObject *vector2_from_polar(PyVector *self, PyObject *args);

/* vector3 specific functions */
static PyObject *vector3_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int vector3_init(PyVector *self, PyObject *args, PyObject *kwds);
static void _vector3_do_rotate(double *dst_coords, const double *src_coords,
                               const double *axis_coords, 
                               double angle, double epsilon);
static PyObject *vector3_rotate(PyVector *self, PyObject *args);
static PyObject *vector3_rotate_ip(PyVector *self, PyObject *args);
static PyObject *vector3_cross(PyVector *self, PyObject *other);
static PyObject *vector3_angle_to(PyVector *self, PyObject *other);
static PyObject *vector3_as_spherical(PyVector *self);
static PyObject *vector3_from_spherical(PyVector *self, PyObject *args);

/* vector iterator functions */
static void vectoriter_dealloc(vectoriter *it);
static PyObject *vectoriter_next(vectoriter *it);
static PyObject *vectoriter_len(vectoriter *it);
static PyObject *vector_iter(PyObject *vec);

/* elementwiseproxy */
static void vector_elementwiseproxy_dealloc(vector_elementwiseproxy *it);
static PyObject *vector_elementwiseproxy_richcompare(PyObject *o1, PyObject *o2, int op);
static PyObject *vector_elementwiseproxy_generic_math(PyObject *o1,
                                                      PyObject *o2, int op);
static PyObject *vector_elementwiseproxy_add(PyObject *o1, PyObject *o2);
static PyObject *vector_elementwiseproxy_sub(PyObject *o1, PyObject *o2);
static PyObject *vector_elementwiseproxy_mul(PyObject *o1, PyObject *o2);
static PyObject *vector_elementwiseproxy_div(PyObject *o1, PyObject *o2);
static PyObject *vector_elementwiseproxy_floor_div(PyObject *o1, PyObject *o2);
static PyObject *vector_elementwiseproxy_pow(PyObject *baseObj, PyObject *expoObj, PyObject *mod);
static PyObject *vector_elementwiseproxy_mod(PyObject *o1, PyObject *o2);
static PyObject *vector_elementwiseproxy_abs(vector_elementwiseproxy *self);
static PyObject *vector_elementwiseproxy_neg(vector_elementwiseproxy *self);
static PyObject *vector_elementwiseproxy_pos(vector_elementwiseproxy *self);
static int vector_elementwiseproxy_nonzero(vector_elementwiseproxy *self);
static PyObject *vector_elementwise(PyVector *vec);


static int swizzling_enabled = 0;


/********************************
 * Global helper functions
 ********************************/

static int
RealNumber_Check(PyObject *obj)
{
    if (PyNumber_Check(obj) && !PyComplex_Check(obj))
        return 1;
    return 0;
}

static double
PySequence_GetItem_AsDouble(PyObject *seq, Py_ssize_t index)
{
    PyObject *item, *fltobj;
    double value;

    item = PySequence_GetItem(seq, index);
    if (item == NULL) {
        PyErr_SetString(PyExc_TypeError, "a sequence is expected");
        return -1;
    }
    fltobj = PyNumber_Float(item);
    Py_DECREF(item);
    if (!fltobj) {
        PyErr_SetString(PyExc_TypeError, "a float is required");
        return -1;
    }
    value = PyFloat_AsDouble(fltobj);
    Py_DECREF(fltobj);
    return value;
}

static int
PySequence_AsVectorCoords(PyObject *seq, double *coords, const size_t size)
{
    int i;

    if (PyVector_Check(seq)) {
        memcpy(coords, ((PyVector *)seq)->coords, sizeof(double) * size);
        return 1;
    }
    if (!PySequence_Check(seq) || PySequence_Length(seq) != size) {
        coords = NULL;
        return 0;
    }

    for (i = 0; i < size; ++i) {
        coords[i] = PySequence_GetItem_AsDouble(seq, i);
        if (PyErr_Occurred()) {
            coords = NULL;
            return 0;
        }
    }
    return 1;
}

static int 
PyVectorCompatible_Check(PyObject *obj, int dim)
{
    int i;
    PyObject *tmp;

    switch(dim) {
    case 2:
        if (PyVector2_Check(obj)) {
            return 1;
        }
        break;
    case 3:
        if (PyVector3_Check(obj)) {
            return 1;
        }
        break;
/*
    case 4:
        if (PyVector4_Check(obj)) {
            return 1;
        }
        break;
*/
    default:
        PyErr_SetString(PyExc_SystemError, 
                        "Wrong internal call to PyVectorCompatible_Check.");
        return 0;
    }

    if (!PySequence_Check(obj) || (PySequence_Length(obj) != dim)) {
        return 0;
    }

    for (i = 0; i < dim; ++i) {
        tmp = PySequence_GetItem(obj, i);
        if (!RealNumber_Check(tmp)) {
            Py_DECREF(tmp);
            return 0;
        }
        Py_DECREF(tmp);
    }
    return 1;
}

static double
_scalar_product(const double *coords1, const double *coords2, int size)
{
    int i;
    double product = 0;
    for (i = 0; i < size; ++i)
        product += coords1[i] * coords2[i];
    return product;
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
    case 3:
        vec = PyObject_New(PyVector, &PyVector3_Type);
        break;
/*
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
    Py_TYPE(self)->tp_free((PyObject*)self);
}






/**********************************************
 * Generic vector PyNumber emulation routines
 **********************************************/

static PyObject *
vector_generic_math(PyObject *o1, PyObject *o2, int op)
{
    int i, dim;
    double *vec_coords;
    double tmp;
    PyObject *other;
    PyVector *vec, *ret;
    if (PyVector_Check(o1)) {
        vec = (PyVector *)o1;
        other = o2;
    }
    else {
        vec = (PyVector *)o2;
        other = o1;
        op |= OP_ARG_REVERSE;
    }
    dim = vec->dim;
    vec_coords = vec->coords;

    if (PyVectorCompatible_Check(other, dim))
        op |= OP_ARG_VECTOR;
    else if (RealNumber_Check(other))
        op |= OP_ARG_NUMBER;
    else
        op |= OP_ARG_UNKNOWN;

    switch (op) {
    case OP_ADD | OP_ARG_VECTOR:
    case OP_ADD | OP_ARG_VECTOR | OP_ARG_REVERSE:
        ret = (PyVector*)PyVector_NEW(dim);
        for (i = 0; i < dim; i++) {
            ret->coords[i] = (vec_coords[i] + 
                              PySequence_GetItem_AsDouble(other, i));
        }
        return (PyObject*)ret;
    case OP_IADD | OP_ARG_VECTOR:
        for (i = 0; i < dim; i++)
            vec_coords[i] += PySequence_GetItem_AsDouble(other, i);
        return (PyObject*)vec;
    case OP_SUB | OP_ARG_VECTOR:
        ret = (PyVector*)PyVector_NEW(dim);
        for (i = 0; i < dim; i++) {
            ret->coords[i] = (vec_coords[i] -
                              PySequence_GetItem_AsDouble(other, i));
        }
        return (PyObject*)ret;
    case OP_SUB | OP_ARG_VECTOR | OP_ARG_REVERSE:
        ret = (PyVector*)PyVector_NEW(dim);
        for (i = 0; i < dim; i++) {
            ret->coords[i] = (PySequence_GetItem_AsDouble(other, i) -
                              vec_coords[i]);
        }
        return (PyObject*)ret;
    case OP_ISUB | OP_ARG_VECTOR:
        for (i = 0; i < dim; i++)
            vec_coords[i] -= PySequence_GetItem_AsDouble(other, i);
        return (PyObject*)vec;
    case OP_MUL | OP_ARG_VECTOR:
    case OP_MUL | OP_ARG_VECTOR | OP_ARG_REVERSE:
        tmp = 0.;
        for (i = 0; i < dim; i++)
            tmp += vec_coords[i] * PySequence_GetItem_AsDouble(other, i);
        return PyFloat_FromDouble(tmp);
    case OP_MUL | OP_ARG_NUMBER:
    case OP_MUL | OP_ARG_NUMBER | OP_ARG_REVERSE:
        ret = (PyVector*)PyVector_NEW(dim);
        tmp = PyFloat_AsDouble(other);
        for (i = 0; i < dim; i++)
            ret->coords[i] = vec_coords[i] * tmp;
        return (PyObject*)ret;
    case OP_IMUL | OP_ARG_NUMBER:
        tmp = PyFloat_AsDouble(other);
        for (i = 0; i < dim; i++)
            vec_coords[i] *= tmp;
        return (PyObject*)vec;
    case OP_DIV | OP_ARG_NUMBER:
        tmp = 1. / PyFloat_AsDouble(other);
        ret = (PyVector*)PyVector_NEW(dim);
        for (i = 0; i < dim; i++)
            ret->coords[i] = vec_coords[i] * tmp;
        return (PyObject*)ret;
    case OP_IDIV | OP_ARG_NUMBER:
        tmp = 1. / PyFloat_AsDouble(other);
        for (i = 0; i < dim; i++)
            vec_coords[i] *= tmp;
        return (PyObject*)vec;
    case OP_FLOOR_DIV | OP_ARG_NUMBER:
        tmp = 1. / PyFloat_AsDouble(other);
        ret = (PyVector*)PyVector_NEW(dim);
        for (i = 0; i < dim; i++)
            ret->coords[i] = floor(vec_coords[i] * tmp);
        return (PyObject*)ret;
    case OP_IFLOOR_DIV | OP_ARG_NUMBER:
        tmp = 1. / PyFloat_AsDouble(other);
        for (i = 0; i < dim; i++)
            vec->coords[i] = floor(vec_coords[i] * tmp);
        return (PyObject*)vec;
    default:
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
}


static PyObject *
vector_add(PyObject *o1, PyObject *o2)
{
    return vector_generic_math(o1, o2, OP_ADD);
}
static PyObject *
vector_inplace_add(PyVector *o1, PyObject *o2)
{
    return vector_generic_math((PyObject*)o1, o2, OP_IADD);
}
static PyObject *
vector_sub(PyObject *o1, PyObject *o2)
{
    return vector_generic_math(o1, o2, OP_SUB);
}
static PyObject *
vector_inplace_sub(PyVector *o1, PyObject *o2)
{
    return vector_generic_math((PyObject*)o1, o2, OP_ISUB);
}
static PyObject *
vector_mul(PyObject *o1, PyObject *o2)
{
    return vector_generic_math(o1, o2, OP_MUL);
}
static PyObject *
vector_inplace_mul(PyVector *o1, PyObject *o2)
{
    return vector_generic_math((PyObject*)o1, o2, OP_IMUL);
}
static PyObject *
vector_div(PyVector *o1, PyObject *o2)
{
    return vector_generic_math((PyObject*)o1, o2, OP_DIV);
}
static PyObject *
vector_inplace_div(PyVector *o1, PyObject *o2)
{
    return vector_generic_math((PyObject*)o1, o2, OP_IDIV);
}
static PyObject *
vector_floor_div(PyVector *o1, PyObject *o2)
{
    return vector_generic_math((PyObject*)o1, o2, OP_FLOOR_DIV);
}
static PyObject *
vector_inplace_floor_div(PyVector *o1, PyObject *o2)
{
    return vector_generic_math((PyObject*)o1, o2, OP_IFLOOR_DIV);
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

static PyNumberMethods vector_as_number = {
    (binaryfunc)vector_add,         /* nb_add;       __add__ */
    (binaryfunc)vector_sub,         /* nb_subtract;  __sub__ */
    (binaryfunc)vector_mul,         /* nb_multiply;  __mul__ */
#if !PY3
    (binaryfunc)vector_div,         /* nb_divide;    __div__ */
#endif
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
#if !PY3
    (coercion)0,                    /* nb_coerce;    __coerce__ */
#endif
    (unaryfunc)0,                   /* nb_int;       __int__ */
    (unaryfunc)0,                   /* nb_long;      __long__ */
    (unaryfunc)0,                   /* nb_float;     __float__ */
#if !PY3
    (unaryfunc)0,                   /* nb_oct;       __oct__ */
    (unaryfunc)0,                   /* nb_hex;       __hex__ */
#endif
    /* Added in release 2.0 */
    (binaryfunc)vector_inplace_add, /* nb_inplace_add;       __iadd__ */
    (binaryfunc)vector_inplace_sub, /* nb_inplace_subtract;  __isub__ */
    (binaryfunc)vector_inplace_mul, /* nb_inplace_multiply;  __imul__ */
#if !PY3
    (binaryfunc)vector_inplace_div, /* nb_inplace_divide;    __idiv__ */
#endif
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
        PyErr_SetString(PyExc_IndexError, "subscript out of range.");
        return NULL;
    }
    return PyFloat_FromDouble(self->coords[index]);
}

static int
vector_SetItem(PyVector *self, Py_ssize_t index, PyObject *value)
{
    if (index < 0 || index >= self->dim) {
        PyErr_SetString(PyExc_IndexError, "subscript out of range.");
        return -1;
    }
    self->coords[index] = PyFloat_AsDouble(value);
    return 0;
}


static PyObject *
vector_GetSlice(PyVector *self, Py_ssize_t ilow, Py_ssize_t ihigh)
{
    /* some code was taken from the CPython source listobject.c */
    PyListObject *slice;
    Py_ssize_t i, len;
    double *src;
    PyObject **dest;

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
    slice = (PyListObject *) PyList_New(len);
    if (slice == NULL)
        return NULL;
    
    src = self->coords + ilow;
    dest = slice->ob_item;
    for (i = 0; i < len; i++) {
        dest[i] = PyFloat_FromDouble(src[i]);
    }
    return (PyObject *)slice;
}

static int
vector_SetSlice(PyVector *self, Py_ssize_t ilow, Py_ssize_t ihigh, PyObject *v)
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
    if (len != PySequence_Length(v)) {
        PyErr_SetString(PyExc_ValueError, 
                        "Cannot assign slice of different length.");
        return -1;
    }
    
    /* TODO: better error checking. for example if v is not a 
             sequence or doesn't numbers. */
    for (i = 0; i < len; ++i) {
        self->coords[i + ilow] = PySequence_GetItem_AsDouble(v, i);
    }
    return 0;
}


static PySequenceMethods vector_as_sequence = {
    (lenfunc)vector_len,             /* sq_length;    __len__ */
    (binaryfunc)0,                   /* sq_concat;    __add__ */
    (ssizeargfunc)0,                 /* sq_repeat;    __mul__ */
    (ssizeargfunc)vector_GetItem,    /* sq_item;      __getitem__ */
    (ssizessizeargfunc)vector_GetSlice, /* sq_slice;     __getslice__ */
    (ssizeobjargproc)vector_SetItem, /* sq_ass_item;  __setitem__ */
    (ssizessizeobjargproc)vector_SetSlice, /* sq_ass_slice; __setslice__ */
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

static PyObject*
vector_getz (PyVector *self, void *closure)
{
    return PyFloat_FromDouble(self->coords[2]);
}

static int
vector_setz (PyVector *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the z attribute");
        return -1;
    }

    self->coords[2] = PyFloat_AsDouble(value);
    return 0;
}

static PyObject*
vector_getw (PyVector *self, void *closure)
{
    return PyFloat_FromDouble(self->coords[3]);
}

static int
vector_setw (PyVector *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the w attribute");
        return -1;
    }

    self->coords[3] = PyFloat_AsDouble(value);
    return 0;
}



static PyObject *
vector_richcompare(PyVector *self, PyObject *other, int op)
{
    int i;
    double diff;
    if (!PyVectorCompatible_Check(other, self->dim)) {
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
            diff = self->coords[i] - PySequence_GetItem_AsDouble(other, i);
            /* test diff != diff to catch NaN */
            if ((diff != diff) || (fabs(diff) >= self->epsilon)) {
                Py_RETURN_FALSE;
            }
        }
        Py_RETURN_TRUE;
    case Py_NE:
        for (i = 0; i < self->dim; i++) {
            diff = self->coords[i] - PySequence_GetItem_AsDouble(other, i);
            if ((diff != diff) || (fabs(diff) >= self->epsilon)) {
                Py_RETURN_TRUE;
            }
        }
        Py_RETURN_FALSE;
    default:
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
}








static PyObject *
vector_length(PyVector *self)
{
    double length_squared = _scalar_product(self->coords, self->coords, 
                                            self->dim);
    return PyFloat_FromDouble(sqrt(length_squared));
}

static PyObject *
vector_length_squared(PyVector *self)
{
    double length_squared = _scalar_product(self->coords, self->coords, 
                                            self->dim);
    return PyFloat_FromDouble(length_squared);
}

static PyObject *
vector_normalize(PyVector *self)
{
    int i;
    double length;
    PyVector *ret;
    
    length = sqrt(_scalar_product(self->coords, self->coords, self->dim));
    if (length == 0) {
        PyErr_SetString(PyExc_ZeroDivisionError, 
                        "Can't normalize Vector of length Zero");
        return NULL;
    }

    ret = (PyVector*)PyVector_NEW(self->dim);
    for (i = 0; i < self->dim; ++i)
        ret->coords[i] = self->coords[i] / length;

    return (PyObject *)ret;
}

static PyObject *
vector_normalize_ip(PyVector *self)
{
    int i;
    double length;
    
    length = sqrt(_scalar_product(self->coords, self->coords, self->dim));

    if (length == 0) {
        PyErr_SetString(PyExc_ZeroDivisionError, 
                        "Can't normalize Vector of length Zero");
        return NULL;
    }

    for (i = 0; i < self->dim; ++i)
        self->coords[i] /= length;

    Py_RETURN_NONE;
}

static PyObject *
vector_is_normalized(PyVector *self)
{
    double length_squared = _scalar_product(self->coords, self->coords, 
                                            self->dim);
    if (fabs(length_squared - 1) < self->epsilon)
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

static PyObject *
vector_dot(PyVector *self, PyObject *other)
{
    double other_coords[VECTOR_MAX_SIZE];
    if (!PySequence_AsVectorCoords(other, other_coords, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "Cannot perform dot product with this type.");
        return NULL;
    }
    return PyFloat_FromDouble(_scalar_product(self->coords, other_coords, 
                                              self->dim));
}

static PyObject *
vector_scale_to_length(PyVector *self, PyObject *length)
{
    int i;
    double new_length, old_length;
    double fraction;

    if (!RealNumber_Check(length)) {
        PyErr_SetString(PyExc_ValueError, "new length must be a number");
        return NULL;
    }
    new_length = PyFloat_AsDouble(length);

    old_length = 0;
    for (i = 0; i < self->dim; ++i)
        old_length += self->coords[i] * self->coords[i];
    old_length = sqrt(old_length);

    if (old_length < self->epsilon) {
        PyErr_SetString(PyExc_ZeroDivisionError, "Cannot scale a vector with zero length");
        return NULL;
    }

    fraction = new_length / old_length;
    for (i = 0; i < self->dim; ++i)
        self->coords[i] *= fraction;

    Py_RETURN_NONE;
}

static int 
_vector_reflect_helper(double *dst_coords, const double *src_coords, 
                       PyObject *normal, int dim, double epsilon)
{
    int i;
    double dot_product;
    double norm_length;
    /* allocate enough space for 2, 3 and 4 dim vectors */
    double norm_coords[VECTOR_MAX_SIZE];

    if (!PyVectorCompatible_Check(normal, dim)) {
        PyErr_SetString(PyExc_TypeError, "Expected a vector.");
        return RETURN_ERROR;
    }

    /* normalize the normal */
    norm_length = 0;
    for (i = 0; i < dim; ++i) {
        norm_coords[i] = PySequence_GetItem_AsDouble(normal, i);
        norm_length += norm_coords[i] * norm_coords[i];
    }
    if (norm_length < epsilon) {
        PyErr_SetString(PyExc_ZeroDivisionError, "Normal must not be of length zero.");
        return RETURN_ERROR;;
    }
    if (norm_length != 1) {
        norm_length = sqrt(norm_length);
        for (i = 0; i < dim; ++i)
            norm_coords[i] /= norm_length;
    }
    
    /* calculate the dot_product for the projection */
    dot_product = 0;
    for (i = 0; i < dim; ++i)
        dot_product += src_coords[i] * norm_coords[i];

    for (i = 0; i < dim; ++i)
        dst_coords[i] = src_coords[i] - 2 * norm_coords[i] * dot_product;
    return RETURN_NO_ERROR;
}

static PyObject *
vector_reflect(PyVector *self, PyObject *normal)
{
    int error_code;
    PyVector *ret = (PyVector *)PyVector_NEW(self->dim);

    error_code = _vector_reflect_helper(ret->coords, self->coords,
                                        normal, self->dim, self->epsilon);
    if (error_code == RETURN_ERROR)
        return NULL;

    return (PyObject *)ret;
}

static PyObject *
vector_reflect_ip(PyVector *self, PyObject *normal)
{
    int error_code;
    double tmp_coords[VECTOR_MAX_SIZE];

    error_code = _vector_reflect_helper(tmp_coords, self->coords, 
                                        normal, self->dim, self->epsilon);
    if (error_code == RETURN_ERROR)
        return NULL;

    memcpy(self->coords, tmp_coords, self->dim * sizeof(tmp_coords[0]));
    Py_RETURN_NONE;
}

static PyObject *
vector_distance_to(PyVector *self, PyObject *other)
{
    int i;
    double distance_squared;

    distance_squared = 0;
    for (i = 0; i < self->dim; ++i) {
        double tmp = PySequence_GetItem_AsDouble(other, i) - self->coords[i];
        distance_squared += tmp * tmp;
    }
    
    return PyFloat_FromDouble(sqrt(distance_squared));
}

static PyObject *
vector_distance_squared_to(PyVector *self, PyObject *other)
{
    int i;
    double distance_squared;

    distance_squared = 0;
    for (i = 0; i < self->dim; ++i) {
        double tmp = PySequence_GetItem_AsDouble(other, i) - self->coords[i];
        distance_squared += tmp * tmp;
    }
    
    return PyFloat_FromDouble(distance_squared);
}



static PyObject *
vector_repr(PyVector *self)
{
    int i;
    int bufferIdx;
    char buffer[2][STRING_BUF_SIZE];
    
    bufferIdx = 1;
    PyOS_snprintf(buffer[0], STRING_BUF_SIZE, "<Vector%d(", self->dim);
    for (i = 0; i < self->dim - 1; ++i) {
        PyOS_snprintf(buffer[bufferIdx % 2], STRING_BUF_SIZE, "%s%g, ", 
                      buffer[(bufferIdx + 1) % 2], self->coords[i]);
        bufferIdx++;
    }
    PyOS_snprintf(buffer[bufferIdx % 2], STRING_BUF_SIZE, "%s%g)>", 
                  buffer[(bufferIdx + 1) % 2], self->coords[i]);
    return Text_FromUTF8(buffer[bufferIdx % 2]); 
}

static PyObject *
vector_str(PyVector *self)
{
    int i;
    int bufferIdx;
    char buffer[2][STRING_BUF_SIZE];
    
    bufferIdx = 1;
    PyOS_snprintf(buffer[0], STRING_BUF_SIZE, "[");
    for (i = 0; i < self->dim - 1; ++i) {
        PyOS_snprintf(buffer[bufferIdx % 2], STRING_BUF_SIZE, "%s%g, ", 
                      buffer[(bufferIdx + 1) % 2], self->coords[i]);
        bufferIdx++;
    }
    PyOS_snprintf(buffer[bufferIdx % 2], STRING_BUF_SIZE, "%s%g]", 
                  buffer[(bufferIdx + 1) % 2], self->coords[i]);
    return Text_FromUTF8(buffer[bufferIdx % 2]); 
}

static PyObject*
vector_getAttr_swizzle(PyVector *self, PyObject *attr_name)
{
    PyObject *res = PyObject_GenericGetAttr((PyObject*)self, attr_name);
    /* if normal lookup failed try to swizzle */
    if (swizzling_enabled && 
        PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        Py_ssize_t i, len = PySequence_Length(attr_name);
        double *coords = self->coords;
        const char *attr = Bytes_AsString(attr_name);
        if (attr == NULL)
            return NULL;
        res = (PyObject*)PyTuple_New(len);
        for (i = 0; i < len; i++) {
            switch (attr[i]) {
            case 'x':
                PyTuple_SetItem(res, i, PyFloat_FromDouble(coords[0]));
                break;
            case 'y':
                PyTuple_SetItem(res, i, PyFloat_FromDouble(coords[1]));
                break;
            default:
                /* swizzling failed! clean up and return NULL
                 * the exception from PyObject_GenericGetAttr is still set */
                Py_DECREF(res);
                return NULL;
            }
        }
        /* swizzling succeeded! clear the error and return result */
        PyErr_Clear();
    }
    return res;
}

static int
vector_setAttr_swizzle(PyVector *self, PyObject *attr_name, PyObject *val)
{
    const char *attr = Bytes_AsString(attr_name);
    Py_ssize_t len = PySequence_Length(attr_name);
    double entry[VECTOR_MAX_SIZE];
    int entry_was_set[VECTOR_MAX_SIZE];
    int swizzle_err = SWIZZLE_ERR_NO_ERR;
    int i;

    /* if swizzling is disabled always default to generic implementation */
    if (!swizzling_enabled)
        return PyObject_GenericSetAttr((PyObject*)self, attr_name, val);

    if (attr == NULL)
        return -1;

    /* if swizzling is enabled first try swizzle */
    for (i = 0; i < self->dim; ++i)
        entry_was_set[i] = 0;

    for (i = 0; i < len; ++i) {
        int idx;
        switch (attr[i]) {
        case 'x':
        case 'y':
        case 'z':
            idx = attr[i] - 'x';
            break;
        case 'w':
            idx = 3;
            break;
        default:
            /* swizzle failed. attempt generic attribute setting */
            return PyObject_GenericSetAttr((PyObject*)self, attr_name, val);
        }
        if (idx >= self->dim)
            /* swizzle failed. attempt generic attribute setting */
            return PyObject_GenericSetAttr((PyObject*)self, attr_name, val);
        if (entry_was_set[idx]) 
            swizzle_err = SWIZZLE_ERR_DOUBLE_IDX;
        if (swizzle_err == SWIZZLE_ERR_NO_ERR) {
            entry_was_set[idx] = 1;
            entry[idx] = PySequence_GetItem_AsDouble(val, i);
            if (PyErr_Occurred())
                swizzle_err = SWIZZLE_ERR_EXTRACTION_ERR;
        }
    }
    switch (swizzle_err) {
    case SWIZZLE_ERR_NO_ERR:
        /* swizzle successful */
        for (i = 0; i < self->dim; ++i)
            if (entry_was_set[i])
                self->coords[i] = entry[i];
        return 0;
    case SWIZZLE_ERR_DOUBLE_IDX:
        PyErr_SetString(PyExc_AttributeError, 
                        "Attribute assignment conflicts with swizzling.");
        return -1;
    case SWIZZLE_ERR_EXTRACTION_ERR:
        /* exception was set by PySequence_GetItem_AsDouble */
        return -1;
    default:
        /* this should NEVER happen and means a bug in the code */
        PyErr_SetString(PyExc_RuntimeError, "Unhandled error in swizzle code");
        return -1;
    }
} 

/*
static Py_ssize_t
vector_readbuffer(PyVector *self, Py_ssize_t segment, void **ptrptr)
{
    if (segment != 0) {
        PyErr_SetString(PyExc_SystemError, 
                        "accessing non-existent vector segment");
        return -1;
    }
    *ptrptr = self->coords;
    return self->dim;
}

static Py_ssize_t
vector_writebuffer(PyVector *self, Py_ssize_t segment, void **ptrptr)
{
    if (segment != 0) {
        PyErr_SetString(PyExc_SystemError, 
                        "accessing non-existent vector segment");
        return -1;
    }
    *ptrptr = self->coords;
    return self->dim;
}

static Py_ssize_t
vector_segcount(PyVector *self, Py_ssize_t *lenp)
{
    if (lenp) {
        *lenp = self->dim * sizeof(self->coords[0]);
    }
    return 1;
}

static int
vector_getbuffer(PyVector *self, Py_buffer *view, int flags)
{
    int ret;
    void *ptr;
    if (view == NULL) {
        self->ob_exports++;
        return 0;
    }
    ptr = self->coords;
    ret = PyBuffer_FillInfo(view, (PyObject*)self, ptr, Py_SIZE(self), 0, flags);
    if (ret >= 0) {
        obj->ob_exports++;
    }
    return ret;
}

static void
vector_releasebuffer(PyVector *self, Py_buffer *view)
{
    self->ob_exports--;
}


static PyBufferProcs vector_as_buffer = {
    (readbufferproc)vector_readbuffer,
    (writebufferproc)vector_writebuffer,
    (segcountproc)vector_segcount,
    (charbufferproc)0,
    (getbufferproc)vector_getbuffer,
    (releasebufferproc)vector_releasebuffer,
};
*/

/*********************************************************************
 * vector2 specific functions
 *********************************************************************/


static PyObject *
vector2_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyVector *vec = (PyVector *)type->tp_alloc(type, 0);

    if (vec != NULL) {
        vec->dim = 2;
        vec->epsilon = FLT_EPSILON;
        vec->coords = PyMem_New(double, vec->dim);
        if (vec->coords == NULL) {
            Py_TYPE(vec)->tp_free((PyObject*)vec);
            return NULL;
        }
    }

    return (PyObject *)vec;
}

static int
vector2_init(PyVector *self, PyObject *args, PyObject *kwds)
{
    PyObject *xOrSequence=NULL, *y=NULL;
    static char *kwlist[] = {"x", "y", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|OO:Vector2", kwlist,
                                      &xOrSequence, &y))
        return -1;

    if (xOrSequence) {
        if (RealNumber_Check(xOrSequence)) {
            self->coords[0] = PyFloat_AsDouble(xOrSequence);
        } 
        else if (PyVectorCompatible_Check(xOrSequence, self->dim)) {
            self->coords[0] = PySequence_GetItem_AsDouble(xOrSequence, 0);
            self->coords[1] = PySequence_GetItem_AsDouble(xOrSequence, 1);
            /* successful initialization from sequence type */
            return 0;
        } 
        else if (Text_Check(xOrSequence)) {
            /* This should make "Vector2(Vector2().__repr__())" possible */
            char *endptr;
            char *str = Bytes_AsString(xOrSequence);
            if (str == NULL)
                return -1;
            if (strncmp(str, "<Vector2(", strlen("<Vector2(")) != 0)
                goto error;
            str += strlen("<Vector2(");
            self->coords[0] = PyOS_ascii_strtod(str, &endptr);
            if (endptr == str) {
                goto error;
            }
            str = endptr + strlen(", ");
            self->coords[1] = PyOS_ascii_strtod(str, &endptr);
            if (endptr == str)
                goto error;
            /* successful conversion from string */
            return 0;
        }
        else {
            goto error;
        }
    } 
    else {
        self->coords[0] = 0.;
    }

    if (y) {
        if (RealNumber_Check(y)) {
            self->coords[1] = PyFloat_AsDouble(y);
        } 
        else {
            goto error;
        }
    } 
    else {
        self->coords[1] = 0.;
    }
    /* success initialization */
    return 0;
error:
    PyErr_SetString(PyExc_ValueError,
                    "Vector2 must be initialized with 2 real numbers or a sequence of 2 real numbers");
    return -1;
}

static void
_vector2_do_rotate(double *dst_coords, const double *src_coords,
                   double angle, double epsilon)
{
    /* make sure angle is in range [0, 360) */
    angle = fmod(angle, 360.);
    if (angle < 0)
        angle += 360.;

    /* special-case rotation by 0, 90, 180 and 270 degrees */
    if (fmod(angle + epsilon, 90.) < 2 * epsilon) {
        switch ((int)((angle + epsilon) / 90)) {
        case 0: /* 0 degrees */
            dst_coords[0] = src_coords[0];
            dst_coords[1] = src_coords[1];
            break;
        case 1: /* 90 degrees */
            dst_coords[0] = -src_coords[1];
            dst_coords[1] = src_coords[0];
            break;
        case 2: /* 180 degrees */
            dst_coords[0] = -src_coords[0];
            dst_coords[1] = -src_coords[1];
            break;
        case 3: /* 270 degrees */
            dst_coords[0] = src_coords[1];
            dst_coords[1] = -src_coords[0];
            break;
        default:
            /* this should NEVER happen and means a bug in the code */
            PyErr_SetString(PyExc_RuntimeError, "Please report this bug in vector2_do_rotate to the developers");
            break;
        }
    }
    else {
        double sinValue, cosValue;

        angle = DEG2RAD(angle);
        sinValue = sin(angle);
        cosValue = cos(angle);

        dst_coords[0] = cosValue * src_coords[0] - sinValue * src_coords[1];
        dst_coords[1] = sinValue * src_coords[0] + cosValue * src_coords[1];
    }
}

static PyObject *
vector2_rotate(PyVector *self, PyObject *args)
{
    double angle;
    PyVector *ret;

    if (!PyArg_ParseTuple(args, "d:rotate", &angle)) {
        return NULL;
    }

    ret = (PyVector*)PyVector_NEW(self->dim);
    _vector2_do_rotate(ret->coords, self->coords, angle, self->epsilon);
    return (PyObject*)ret;
}

static PyObject *
vector2_rotate_ip(PyVector *self, PyObject *args)
{
    double angle;
    double tmp[2];
    
    if (!PyArg_ParseTuple(args, "d:rotate_ip", &angle)) {
        return NULL;
    }

    tmp[0] = self->coords[0];
    tmp[1] = self->coords[1];
    _vector2_do_rotate(self->coords, tmp, angle, self->epsilon);
    Py_RETURN_NONE;
}

static PyObject *
vector2_cross(PyVector *self, PyObject *other)
{
    if (!PyVectorCompatible_Check(other, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "cannot calculate cross Product");
        return NULL;
    }
    
    if (PyVector_Check(other)) {
        return PyFloat_FromDouble((self->coords[0] * ((PyVector *)other)->coords[1]) -
                                  (self->coords[1] * ((PyVector *)other)->coords[0]));
    }
    else {
        return PyFloat_FromDouble((self->coords[0] * PySequence_GetItem_AsDouble(other, 1)) -
                                  (self->coords[1] * PySequence_GetItem_AsDouble(other, 0)));
    }
}

static PyObject *
vector2_angle_to(PyVector *self, PyObject *other)
{
    double angle;
    if (!PyVectorCompatible_Check(other, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "expected an vector.");
        return NULL;
    }
    
    angle = (atan2(PySequence_GetItem_AsDouble(other, 1),
                   PySequence_GetItem_AsDouble(other, 0)) - 
             atan2(self->coords[1], self->coords[0]));
    return PyFloat_FromDouble(RAD2DEG(angle));
}

static PyObject *
vector2_as_polar(PyVector *self)
{
    double r, phi;
    r = sqrt(_scalar_product(self->coords, self->coords, self->dim));
    phi = atan2(self->coords[1], self->coords[0]);
    return Py_BuildValue("(dd)", r, phi);
}

static PyObject *
vector2_from_polar(PyVector *self, PyObject *args)
{
    PyObject *polar_obj, *r_obj, *phi_obj;
    double r, phi;
    if (!PyArg_ParseTuple(args, "O:Vector2.from_polar", &polar_obj)) {
        return NULL;
    }
    if (!PySequence_Check(polar_obj) || PySequence_Length(polar_obj) != 2) {
        PyErr_SetString(PyExc_TypeError, 
                        "2-tuple containing r and phi is expected.");
        return NULL;
    }
    r_obj = PySequence_GetItem(polar_obj, 0);
    phi_obj = PySequence_GetItem(polar_obj, 1);
    if (!PyNumber_Check(r_obj) || !PyNumber_Check(phi_obj)) {
        PyErr_SetString(PyExc_TypeError, 
                        "expected numerical values");
        Py_DECREF(r_obj);
        Py_DECREF(phi_obj);
        return NULL;
    }
    r = PyFloat_AsDouble(r_obj);
    phi = PyFloat_AsDouble(phi_obj);
    Py_DECREF(r_obj);
    Py_DECREF(phi_obj);
    self->coords[0] = r * cos(phi);
    self->coords[1] = r * sin(phi);
    
    Py_RETURN_NONE;
}



static PyMethodDef vector2_methods[] = {
    {"length", (PyCFunction)vector_length, METH_NOARGS,
     DOC_VECTOR2LENGTH
    },
    {"length_squared", (PyCFunction)vector_length_squared, METH_NOARGS,
     DOC_VECTOR2LENGTHSQUARED
    },
    {"rotate", (PyCFunction)vector2_rotate, METH_VARARGS,
     DOC_VECTOR2ROTATE
    },
    {"rotate_ip", (PyCFunction)vector2_rotate_ip, METH_VARARGS,
     DOC_VECTOR2ROTATEIP
    },
    {"slerp", (PyCFunction)vector_slerp, METH_VARARGS,
     DOC_VECTOR2SLERP
    },
    {"lerp", (PyCFunction)vector_lerp, METH_VARARGS,
     DOC_VECTOR2LERP
    },
    {"normalize", (PyCFunction)vector_normalize, METH_NOARGS,
     DOC_VECTOR2NORMALIZE
    },
    {"normalize_ip", (PyCFunction)vector_normalize_ip, METH_NOARGS,
     DOC_VECTOR2NORMALIZEIP
    },
    {"is_normalized", (PyCFunction)vector_is_normalized, METH_NOARGS,
     DOC_VECTOR2ISNORMALIZED
    },
    {"cross", (PyCFunction)vector2_cross, METH_O,
     DOC_VECTOR2CROSS
    },
    {"dot", (PyCFunction)vector_dot, METH_O,
     DOC_VECTOR2DOT
    },
    {"angle_to", (PyCFunction)vector2_angle_to, METH_O,
     DOC_VECTOR2ANGLETO
    },
    {"scale_to_length", (PyCFunction)vector_scale_to_length, METH_O,
     DOC_VECTOR2SCALETOLENGTH
    },
    {"reflect", (PyCFunction)vector_reflect, METH_O,
     DOC_VECTOR2REFLECT
    },
    {"reflect_ip", (PyCFunction)vector_reflect_ip, METH_O,
     DOC_VECTOR2REFLECTIP
    },
    {"distance_to", (PyCFunction)vector_distance_to, METH_O,
     DOC_VECTOR2DISTANCETO
    },
    {"distance_squared_to", (PyCFunction)vector_distance_squared_to, METH_O,
     DOC_VECTOR2DISTANCESQUAREDTO
    },
    {"elementwise", (PyCFunction)vector_elementwise, METH_NOARGS,
     DOC_VECTOR2ELEMENTWISE
    },
    {"as_polar", (PyCFunction)vector2_as_polar, METH_NOARGS,
     DOC_VECTOR2ASPOLAR
    },
    {"from_polar", (PyCFunction)vector2_from_polar, METH_VARARGS,
     DOC_VECTOR2FROMPOLAR
    },
    
    {NULL}  /* Sentinel */
};


static PyGetSetDef vector2_getsets[] = {
    { "x", (getter)vector_getx, (setter)vector_setx, NULL, NULL },
    { "y", (getter)vector_gety, (setter)vector_sety, NULL, NULL },
    { NULL, 0, NULL, NULL, NULL }  /* Sentinel */
};


/********************************
 * PyVector2 type definition
 ********************************/

static PyTypeObject PyVector2_Type = {
    TYPE_HEAD(NULL, 0)
    "pygame.math.Vector2",     /* tp_name */
    sizeof(PyVector),          /* tp_basicsize */
    0,                         /* tp_itemsize */
    /* Methods to implement standard operations */
    (destructor)vector_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_compare */
    (reprfunc)vector_repr,     /* tp_repr */
    /* Method suites for standard classes */
    &vector_as_number,         /* tp_as_number */
    &vector_as_sequence,       /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    /* More standard operations (here for binary compatibility) */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    (reprfunc)vector_str,      /* tp_str */
    (getattrofunc)vector_getAttr_swizzle, /* tp_getattro */
    (setattrofunc)vector_setAttr_swizzle, /* tp_setattro */
    /* Functions to access object as input/output buffer */
    0,                         /* tp_as_buffer */
    /* Flags to define presence of optional/expanded features */
#if PY3
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
#else
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | 
    Py_TPFLAGS_CHECKTYPES, /* tp_flags */
#endif
    /* Documentation string */
    DOC_PYGAMEMATHVECTOR2,     /* tp_doc */

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
    vector_iter,               /* tp_iter */
    0,                         /* tp_iternext */
    /* Attribute descriptor and subclassing stuff */
    vector2_methods,           /* tp_methods */
    vector_members,            /* tp_members */
    vector2_getsets,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)vector2_init,    /* tp_init */
    0,                         /* tp_alloc */
    (newfunc)vector2_new,      /* tp_new */
    0,                         /* tp_free */
    0,                         /* tp_is_gc */
    0,                         /* tp_bases */
    0,                         /* tp_mro */
    0,                         /* tp_cache */
    0,                         /* tp_subclasses */
    0,                         /* tp_weaklist */
};










/*************************************************************
 *  PyVector3 specific functions
 *************************************************************/



static PyObject *
vector3_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyVector *vec = (PyVector *)type->tp_alloc(type, 0);

    if (vec != NULL) {
        vec->dim = 3;
        vec->epsilon = FLT_EPSILON;
        vec->coords = PyMem_New(double, vec->dim);
        if (vec->coords == NULL) {
            Py_TYPE(vec)->tp_free((PyObject*)vec);
            return NULL;
        }
    }

    return (PyObject *)vec;
}

static int
vector3_init(PyVector *self, PyObject *args, PyObject *kwds)
{
    PyObject *xOrSequence=NULL, *y=NULL, *z=NULL;
    static char *kwlist[] = {"x", "y", "z", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|OOO:Vector3", kwlist,
                                      &xOrSequence, &y, &z))
        return -1;

    if (xOrSequence) {
        if (RealNumber_Check(xOrSequence)) {
            self->coords[0] = PyFloat_AsDouble(xOrSequence);
        } 
        else if (PyVectorCompatible_Check(xOrSequence, self->dim)) {
            self->coords[0] = PySequence_GetItem_AsDouble(xOrSequence, 0);
            self->coords[1] = PySequence_GetItem_AsDouble(xOrSequence, 1);
            self->coords[2] = PySequence_GetItem_AsDouble(xOrSequence, 2);
            /* successful initialization from sequence type */
            return 0;
        } 
        else if (Text_Check(xOrSequence)) {
            /* This should make "Vector3(Vector3().__repr__())" possible */
            char *endptr;
            char *str = Bytes_AsString(xOrSequence);
            if (str == NULL)
                return -1;
            if (strncmp(str, "<Vector3(", strlen("<Vector3(")) != 0)
                goto error;
            str += strlen("<Vector3(");
            self->coords[0] = PyOS_ascii_strtod(str, &endptr);
            if (endptr == str) {
                goto error;
            }
            str = endptr + strlen(", ");
            self->coords[1] = PyOS_ascii_strtod(str, &endptr);
            if (endptr == str)
                goto error;
            str = endptr + strlen(", ");
            self->coords[2] = PyOS_ascii_strtod(str, &endptr);
            if (endptr == str)
                goto error;
            /* successful conversion from string */
            return 0;
        }
        else {
            goto error;
        }
    } 
    else {
        self->coords[0] = 0.;
    }

    if (y) {
        if (RealNumber_Check(y)) {
            self->coords[1] = PyFloat_AsDouble(y);
        } 
        else {
            goto error;
        }
    } 
    else {
        self->coords[1] = 0.;
    }

    if (z) {
        if (RealNumber_Check(z)) {
            self->coords[2] = PyFloat_AsDouble(z);
        } 
        else {
            goto error;
        }
    } 
    else {
        self->coords[2] = 0.;
    }
    /* success initialization */
    return 0;
error:
    PyErr_SetString(PyExc_ValueError,
                    "Vector3 must be initialized with 3 real numbers or a sequence of 3 real numbers");
    return -1;
}


static void
_vector3_do_rotate(double *dst_coords, const double *src_coords, 
                   const double *axis_coords, 
                   double angle, double epsilon)
{
    double sinValue, cosValue, cosComplement;
    double normalizationFactor;
    double axisLength2 = 0;
    double axis[3];
    int i;

    /* make sure angle is in range [0, 360) */
    angle = fmod(angle, 360.);
    if (angle < 0)
        angle += 360.;

    for (i = 0; i < 3; ++i) {
        axisLength2 += axis_coords[i] * axis_coords[i];
        axis[i] = axis_coords[i];
    }

    /* normalize the axis */
    if (axisLength2 - 1 > epsilon) {
        normalizationFactor = 1. / sqrt(axisLength2);
        for (i = 0; i < 3; ++i)
            axis[i] *= normalizationFactor;
    }

    /* special-case rotation by 0, 90, 180 and 270 degrees */
    if (fmod(angle + epsilon, 90.) < 2 * epsilon) {
        switch ((int)((angle + epsilon) / 90)) {
        case 0: /* 0 degrees */
            memcpy(dst_coords, src_coords, 3 * sizeof(src_coords[0]));
            break;
        case 1: /* 90 degrees */
            dst_coords[0] = (src_coords[0] * (axis[0] * axis[0]) +
                             src_coords[1] * (axis[0] * axis[1] - axis[2]) +
                             src_coords[2] * (axis[0] * axis[2] + axis[1]));
            dst_coords[1] = (src_coords[0] * (axis[0] * axis[1] + axis[2]) +
                             src_coords[1] * (axis[1] * axis[1]) +
                             src_coords[2] * (axis[1] * axis[2] - axis[0]));
            dst_coords[2] = (src_coords[0] * (axis[0] * axis[2] - axis[1]) +
                             src_coords[1] * (axis[1] * axis[2] + axis[0]) +
                             src_coords[2] * (axis[2] * axis[2]));
            break;
        case 2: /* 180 degrees */
            dst_coords[0] = (src_coords[0] * (-1 + axis[0] * axis[0] * 2) +
                             src_coords[1] * (axis[0] * axis[1] * 2) +
                             src_coords[2] * (axis[0] * axis[2] * 2));
            dst_coords[1] = (src_coords[0] * (axis[0] * axis[1] * 2) +
                             src_coords[1] * (-1 + axis[1] * axis[1] * 2) +
                             src_coords[2] * (axis[1] * axis[2] * 2));
            dst_coords[2] = (src_coords[0] * (axis[0] * axis[2] * 2) +
                             src_coords[1] * (axis[1] * axis[2] * 2) +
                             src_coords[2] * (-1 + axis[2] * axis[2] * 2));
            break;
        case 3: /* 270 degrees */
            dst_coords[0] = (src_coords[0] * (axis[0] * axis[0]) +
                             src_coords[1] * (axis[0] * axis[1] + axis[2]) +
                             src_coords[2] * (axis[0] * axis[2] - axis[1]));
            dst_coords[1] = (src_coords[0] * (axis[0] * axis[1] - axis[2]) +
                             src_coords[1] * (axis[1] * axis[1]) +
                             src_coords[2] * (axis[1] * axis[2] + axis[0]));
            dst_coords[2] = (src_coords[0] * (axis[0] * axis[2] + axis[1]) +
                             src_coords[1] * (axis[1] * axis[2] - axis[0]) +
                             src_coords[2] * (axis[2] * axis[2]));
            break;
        }
    }
    else {
        angle = DEG2RAD(angle);
        sinValue = sin(angle);
        cosValue = cos(angle);
        cosComplement = 1 - cosValue;

        dst_coords[0] = (src_coords[0] * (cosValue + axis[0] * axis[0] * cosComplement) +
                         src_coords[1] * (axis[0] * axis[1] * cosComplement - axis[2] * sinValue) +
                         src_coords[2] * (axis[0] * axis[2] * cosComplement + axis[1] * sinValue));
        dst_coords[1] = (src_coords[0] * (axis[0] * axis[1] * cosComplement + axis[2] * sinValue) +
                         src_coords[1] * (cosValue + axis[1] * axis[1] * cosComplement) +
                         src_coords[2] * (axis[1] * axis[2] * cosComplement - axis[0] * sinValue));
        dst_coords[2] = (src_coords[0] * (axis[0] * axis[2] * cosComplement - axis[1] * sinValue) +
                         src_coords[1] * (axis[1] * axis[2] * cosComplement + axis[0] * sinValue) +
                         src_coords[2] * (cosValue + axis[2] * axis[2] * cosComplement));
    }
}

static PyObject *
vector3_rotate(PyVector *self, PyObject *args)
{
    PyVector *ret;
    PyObject *axis;
    double axis_coords[3];
    double angle;

    if (!PyArg_ParseTuple(args, "dO:rotate", &angle, &axis)) {
        return NULL;
    }
    if (!PyVectorCompatible_Check(axis, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "axis must be a 3D Vector");
        return NULL;
    }
    
    PySequence_AsVectorCoords(axis, axis_coords, 3);
    ret = (PyVector*)PyVector_NEW(self->dim);
    _vector3_do_rotate(ret->coords, self->coords, axis_coords,
                       angle, self->epsilon);
    return (PyObject*)ret;
}

static PyObject *
vector3_rotate_ip(PyVector *self, PyObject *args)
{
    PyObject *axis;
    double axis_coords[3];
    double angle;
    double tmp[3];
    
    if (!PyArg_ParseTuple(args, "dO:rotate_ip", &angle, &axis)) {
        return NULL;
    }
    if (!PyVectorCompatible_Check(axis, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "axis must be a 3D Vector");
        return NULL;
    }

    PySequence_AsVectorCoords(axis, axis_coords, 3);
    memcpy(tmp, self->coords, 3 * sizeof(self->coords[0]));
    _vector3_do_rotate(self->coords, tmp, axis_coords, angle, self->epsilon);
    Py_RETURN_NONE;
}

static PyObject *
vector3_rotate_x(PyVector *self, PyObject *angleObject)
{
    PyVector *ret;
    double sinValue, cosValue;
    double angle;

    if (!RealNumber_Check(angleObject)) {
        PyErr_SetString(PyExc_TypeError, "angle must be a real number");
        return NULL;
    }
    angle = DEG2RAD(PyFloat_AsDouble(angleObject));
    sinValue = sin(angle);
    cosValue = cos(angle);

    ret = (PyVector*)PyVector_NEW(self->dim);
    ret->coords[0] = self->coords[0];
    ret->coords[1] = self->coords[1] * cosValue - self->coords[2] * sinValue;
    ret->coords[2] = self->coords[1] * sinValue + self->coords[2] * cosValue;
    return (PyObject*)ret;
}

static PyObject *
vector3_rotate_x_ip(PyVector *self, PyObject *angleObject)
{
    double tmp_coords[3];
    double sinValue, cosValue;
    double angle;

    if (!RealNumber_Check(angleObject)) {
        PyErr_SetString(PyExc_TypeError, "angle must be a real number");
        return NULL;
    }
    angle = DEG2RAD(PyFloat_AsDouble(angleObject));
    sinValue = sin(angle);
    cosValue = cos(angle);
    memcpy(tmp_coords, self->coords, 3 * sizeof(tmp_coords[0]));

    self->coords[1] = tmp_coords[1] * cosValue - tmp_coords[2] * sinValue;
    self->coords[2] = tmp_coords[1] * sinValue + tmp_coords[2] * cosValue;
    Py_RETURN_NONE;
}

static PyObject *
vector3_rotate_y(PyVector *self, PyObject *angleObject)
{
    PyVector *ret;
    double sinValue, cosValue;
    double angle;

    if (!RealNumber_Check(angleObject)) {
        PyErr_SetString(PyExc_TypeError, "angle must be a real number");
        return NULL;
    }
    angle = DEG2RAD(PyFloat_AsDouble(angleObject));
    sinValue = sin(angle);
    cosValue = cos(angle);

    ret = (PyVector*)PyVector_NEW(self->dim);
    ret->coords[0] = self->coords[0] * cosValue + self->coords[2] * sinValue;
    ret->coords[1] = self->coords[1];
    ret->coords[2] = -self->coords[0] * sinValue + self->coords[2] * cosValue;

    return (PyObject*)ret;
}

static PyObject *
vector3_rotate_y_ip(PyVector *self, PyObject *angleObject)
{
    double tmp_coords[3];
    double sinValue, cosValue;
    double angle;

    if (!RealNumber_Check(angleObject)) {
        PyErr_SetString(PyExc_TypeError, "angle must be a real number");
        return NULL;
    }
    angle = DEG2RAD(PyFloat_AsDouble(angleObject));
    sinValue = sin(angle);
    cosValue = cos(angle);
    memcpy(tmp_coords, self->coords, 3 * sizeof(tmp_coords[0]));

    self->coords[0] = tmp_coords[0] * cosValue + tmp_coords[2] * sinValue;
    self->coords[2] = -tmp_coords[0] * sinValue + tmp_coords[2] * cosValue;
    Py_RETURN_NONE;
}

static PyObject *
vector3_rotate_z(PyVector *self, PyObject *angleObject)
{
    PyVector *ret;
    double sinValue, cosValue;
    double angle;

    if (!RealNumber_Check(angleObject)) {
        PyErr_SetString(PyExc_TypeError, "angle must be a real number");
        return NULL;
    }
    angle = DEG2RAD(PyFloat_AsDouble(angleObject));
    sinValue = sin(angle);
    cosValue = cos(angle);

    ret = (PyVector*)PyVector_NEW(self->dim);
    ret->coords[0] = self->coords[0] * cosValue - self->coords[1] * sinValue;
    ret->coords[1] = self->coords[0] * sinValue + self->coords[1] * cosValue;
    ret->coords[2] = self->coords[2];

    return (PyObject*)ret;
}

static PyObject *
vector3_rotate_z_ip(PyVector *self, PyObject *angleObject)
{
    double tmp_coords[3];
    double sinValue, cosValue;
    double angle;

    if (!RealNumber_Check(angleObject)) {
        PyErr_SetString(PyExc_TypeError, "angle must be a real number");
        return NULL;
    }
    angle = DEG2RAD(PyFloat_AsDouble(angleObject));
    sinValue = sin(angle);
    cosValue = cos(angle);
    memcpy(tmp_coords, self->coords, 3 * sizeof(tmp_coords[0]));

    self->coords[0] = tmp_coords[0] * cosValue - tmp_coords[1] * sinValue;
    self->coords[1] = tmp_coords[0] * sinValue + tmp_coords[1] * cosValue;
    Py_RETURN_NONE;
}


static PyObject *
vector3_cross(PyVector *self, PyObject *other)
{
    PyVector *ret;
    double *ret_coords;

    if (!PyVectorCompatible_Check(other, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "cannot calculate cross Product");
        return NULL;
    }
    
    ret = (PyVector*)PyVector_NEW(self->dim);
    ret_coords = ret->coords;
    if (PyVector_Check(other)) {
        double *other_coords = ((PyVector *)other)->coords;
        ret_coords[0] = (self->coords[1] * other_coords[2]) - (self->coords[2] * other_coords[1]);
        ret_coords[1] = (self->coords[2] * other_coords[0]) - (self->coords[0] * other_coords[2]);
        ret_coords[2] = (self->coords[0] * other_coords[1]) - (self->coords[1] * other_coords[0]);
    }
    else {
        double other_coords[3];
        PySequence_AsVectorCoords(other, other_coords, 3);
        ret_coords[0] = (self->coords[1] * other_coords[2]) - (self->coords[2] * other_coords[1]);
        ret_coords[1] = (self->coords[2] * other_coords[0]) - (self->coords[0] * other_coords[2]);
        ret_coords[2] = (self->coords[0] * other_coords[1]) - (self->coords[1] * other_coords[0]);
    }
    return (PyObject*)ret;
}

static PyObject *
vector3_angle_to(PyVector *self, PyObject *other)
{
    double angle, tmp1, tmp2;
    double other_coords[VECTOR_MAX_SIZE];

    if (!PyVectorCompatible_Check(other, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "expected an vector.");
        return NULL;
    }
    
    PySequence_AsVectorCoords(other, other_coords, self->dim);
    tmp1 = _scalar_product(self->coords, self->coords, self->dim);
    tmp2 = _scalar_product(other_coords, other_coords, self->dim);
    angle = acos(_scalar_product(self->coords, other_coords, self->dim) /
                 sqrt(tmp1 * tmp2));
    return PyFloat_FromDouble(RAD2DEG(angle));
}

static PyObject *
vector3_as_spherical(PyVector *self)
{
    double r, theta, phi;
    r = sqrt(_scalar_product(self->coords, self->coords, self->dim));
    theta = acos(self->coords[2] / r);
    phi = atan2(self->coords[1], self->coords[0]);
    return Py_BuildValue("(ddd)", r, theta, phi);
}

static PyObject *
vector3_from_spherical(PyVector *self, PyObject *args)
{
    PyObject *spherical_obj, *r_obj, *theta_obj, *phi_obj;
    double r, theta, phi;
    theta_obj = NULL;
    phi_obj = NULL;
    if (!PyArg_ParseTuple(args, "O:Vector3.from_spherical", &spherical_obj)) {
        return NULL;
    }
    
    if (!PySequence_Check(spherical_obj) || PySequence_Length(spherical_obj) != 3) {
        PyErr_SetString(PyExc_TypeError, "3-tuple containing r, theta and phi is expected.");
        return NULL;
    }
    r_obj = PySequence_GetItem(spherical_obj, 0);
    theta_obj = PySequence_GetItem(spherical_obj, 1);
    phi_obj = PySequence_GetItem(spherical_obj, 2);
    if (!PyNumber_Check(r_obj) || !PyNumber_Check(theta_obj) ||
        !PyNumber_Check(phi_obj)) {
        PyErr_SetString(PyExc_TypeError, 
                        "expected numerical values");
        Py_DECREF(r_obj);
        Py_DECREF(theta_obj);
        Py_DECREF(phi_obj);
        return NULL;
    }
    r = PyFloat_AsDouble(r_obj);
    theta = PyFloat_AsDouble(theta_obj);
    phi = PyFloat_AsDouble(phi_obj);
    Py_DECREF(r_obj);
    Py_DECREF(theta_obj);
    Py_DECREF(phi_obj);

    self->coords[0] = r * sin(theta) * cos(phi);
    self->coords[1] = r * sin(theta) * sin(phi);
    self->coords[2] = r * cos(theta);

    Py_RETURN_NONE;
}

static PyMethodDef vector3_methods[] = {
    {"length", (PyCFunction)vector_length, METH_NOARGS,
     DOC_VECTOR3LENGTH
    },
    {"length_squared", (PyCFunction)vector_length_squared, METH_NOARGS,
     DOC_VECTOR3LENGTHSQUARED
    },
    {"rotate", (PyCFunction)vector3_rotate, METH_VARARGS,
     DOC_VECTOR3ROTATE
    },
    {"rotate_ip", (PyCFunction)vector3_rotate_ip, METH_VARARGS,
     DOC_VECTOR3ROTATEIP
    },
    {"rotate_x", (PyCFunction)vector3_rotate_x, METH_O,
     DOC_VECTOR3ROTATEX
    },
    {"rotate_x_ip", (PyCFunction)vector3_rotate_x_ip, METH_O,
     DOC_VECTOR3ROTATEXIP
    },
    {"rotate_y", (PyCFunction)vector3_rotate_y, METH_O,
     DOC_VECTOR3ROTATEY
    },
    {"rotate_y_ip", (PyCFunction)vector3_rotate_y_ip, METH_O,
     DOC_VECTOR3ROTATEYIP
    },
    {"rotate_z", (PyCFunction)vector3_rotate_z, METH_O,
     DOC_VECTOR3ROTATEZ
    },
    {"rotate_z_ip", (PyCFunction)vector3_rotate_z_ip, METH_O,
     DOC_VECTOR3ROTATEZIP
    },
    {"slerp", (PyCFunction)vector_slerp, METH_VARARGS,
     DOC_VECTOR3SLERP
    },
    {"lerp", (PyCFunction)vector_lerp, METH_VARARGS,
     DOC_VECTOR3LERP
    },
    {"normalize", (PyCFunction)vector_normalize, METH_NOARGS,
     DOC_VECTOR3NORMALIZE
    },
    {"normalize_ip", (PyCFunction)vector_normalize_ip, METH_NOARGS,
     DOC_VECTOR3NORMALIZEIP
    },
    {"is_normalized", (PyCFunction)vector_is_normalized, METH_NOARGS,
     DOC_VECTOR3ISNORMALIZED
    },
    {"cross", (PyCFunction)vector3_cross, METH_O,
     DOC_VECTOR3CROSS
    },
    {"dot", (PyCFunction)vector_dot, METH_O,
     DOC_VECTOR3DOT
    },
    {"angle_to", (PyCFunction)vector3_angle_to, METH_O,
     DOC_VECTOR3ANGLETO
    },
    {"scale_to_length", (PyCFunction)vector_scale_to_length, METH_O,
     DOC_VECTOR3SCALETOLENGTH
    },
    {"reflect", (PyCFunction)vector_reflect, METH_O,
     DOC_VECTOR3REFLECT
    },
    {"reflect_ip", (PyCFunction)vector_reflect_ip, METH_O,
     DOC_VECTOR3REFLECTIP
    },
    {"distance_to", (PyCFunction)vector_distance_to, METH_O,
     DOC_VECTOR3DISTANCETO
    },
    {"distance_squared_to", (PyCFunction)vector_distance_squared_to, METH_O,
     DOC_VECTOR3DISTANCESQUAREDTO
    },
    {"elementwise", (PyCFunction)vector_elementwise, METH_NOARGS,
     DOC_VECTOR3ELEMENTWISE
    },
    {"as_spherical", (PyCFunction)vector3_as_spherical, METH_NOARGS,
     DOC_VECTOR3ASSPHERICAL
    },
    {"from_spherical", (PyCFunction)vector3_from_spherical, METH_VARARGS,
     DOC_VECTOR3FROMSPHERICAL
    },
    
    {NULL}  /* Sentinel */
};

static PyGetSetDef vector3_getsets[] = {
    { "x", (getter)vector_getx, (setter)vector_setx, NULL, NULL },
    { "y", (getter)vector_gety, (setter)vector_sety, NULL, NULL },
    { "z", (getter)vector_getz, (setter)vector_setz, NULL, NULL },
    { NULL, 0, NULL, NULL, NULL }  /* Sentinel */
};

/********************************
 * PyVector3 type definition
 ********************************/

static PyTypeObject PyVector3_Type = {
    TYPE_HEAD (NULL, 0)
    "pygame.math.Vector3",     /* tp_name */
    sizeof(PyVector),          /* tp_basicsize */
    0,                         /* tp_itemsize */
    /* Methods to implement standard operations */
    (destructor)vector_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_compare */
    (reprfunc)vector_repr,     /* tp_repr */
    /* Method suites for standard classes */
    &vector_as_number,         /* tp_as_number */
    &vector_as_sequence,       /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    /* More standard operations (here for binary compatibility) */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    (reprfunc)vector_str,      /* tp_str */
    (getattrofunc)vector_getAttr_swizzle, /* tp_getattro */
    (setattrofunc)vector_setAttr_swizzle, /* tp_setattro */
    /* Functions to access object as input/output buffer */
    0,                         /* tp_as_buffer */
    /* Flags to define presence of optional/expanded features */
#if PY3
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
#else
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | 
    Py_TPFLAGS_CHECKTYPES, /* tp_flags */
#endif
    /* Documentation string */
    DOC_PYGAMEMATHVECTOR3,         /* tp_doc */

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
    vector_iter,               /* tp_iter */
    0,                         /* tp_iternext */
    /* Attribute descriptor and subclassing stuff */
    vector3_methods,           /* tp_methods */
    vector_members,            /* tp_members */
    vector3_getsets,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)vector3_init,    /* tp_init */
    0,                         /* tp_alloc */
    (newfunc)vector3_new,      /* tp_new */
    0,                         /* tp_free */
    0,                         /* tp_is_gc */
    0,                         /* tp_bases */
    0,                         /* tp_mro */
    0,                         /* tp_cache */
    0,                         /* tp_subclasses */
    0,                         /* tp_weaklist */
};












/********************************************
 * PyVectorIterator type definition
 ********************************************/

static void
vectoriter_dealloc(vectoriter *it)
{
    Py_XDECREF(it->vec);
    PyObject_Del(it);
}

static PyObject *
vectoriter_next(vectoriter *it)
{
    assert(it != NULL);
    if (it->vec == NULL)
        return NULL;
    assert(PyVector_Check(it->vec));

    if (it->it_index < it->vec->dim) {
        double item = it->vec->coords[it->it_index];
        ++(it->it_index);
        return PyFloat_FromDouble(item);
    }
    
    Py_DECREF(it->vec);
    it->vec = NULL;
    return NULL;
}

static PyObject *
vectoriter_len(vectoriter *it)
{
    Py_ssize_t len = 0;
    if (it && it->vec) {
        len = it->vec->dim - it->it_index;
    }
#if PY_VERSION_HEX >= 0x02050000
    return PyInt_FromSsize_t(len);
#else
    return PyInt_FromLong((long unsigned int)len);
#endif
}

static PyMethodDef vectoriter_methods[] = {
    {"__length_hint__", (PyCFunction)vectoriter_len, METH_NOARGS,
    },
    {NULL, NULL} /* sentinel */
};

static PyTypeObject PyVectorIter_Type = {
    TYPE_HEAD (NULL, 0)
    "pygame.math.VectorIterator", /* tp_name */
    sizeof(vectoriter),        /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)vectoriter_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_compare */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    0,                         /* tp_str */
    PyObject_GenericGetAttr,   /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    0,                         /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    PyObject_SelfIter,         /* tp_iter */
    (iternextfunc)vectoriter_next, /* tp_iternext */
    vectoriter_methods,        /* tp_methods */
    0,                         /* tp_members */
};

static PyObject *
vector_iter(PyObject *vec)
{
    vectoriter *it;
    if (!PyVector_Check(vec)) {
        PyErr_BadInternalCall();
        return NULL;
    }

    it = PyObject_New(vectoriter, &PyVectorIter_Type);
    if (it == NULL)
        return NULL;
    it->it_index = 0;
    Py_INCREF(vec);
    it->vec = (PyVector *)vec;
    return (PyObject *)it;
}



/********************************************
 * PyVector_SlerpIterator type definition
 ********************************************/
static void
vector_slerpiter_dealloc(vector_slerpiter *it)
{
    PyObject_Del(it);
}

static PyObject *
vector_slerpiter_next(vector_slerpiter *it)
{
    int i, j;
    PyVector *ret;
    double tmp;
    assert(it != NULL);

    if (it->it_index < it->steps) {
        ret = (PyVector*)PyVector_NEW(it->dim);
        if (ret != NULL) {
            for (i = 0; i < it->dim; ++i) {
                tmp = 0;
                for (j = 0; j < it->dim; ++j)
                    tmp += it->coords[j] * it->matrix[j][i];
                ret->coords[i] = tmp * it->radial_factor;
            }
            memcpy(it->coords, ret->coords, sizeof(it->coords[0]) * it->dim);
        }
        ++(it->it_index);
        return (PyObject*)ret;
    }
    
    return NULL;
}

static PyObject *
vector_slerpiter_len(vector_slerpiter *it)
{
    Py_ssize_t len = 0;
    if (it) {
        len = it->steps - it->it_index;
    }
#if PY_VERSION_HEX >= 0x02050000
    return PyInt_FromSsize_t(len);
#else
    return PyInt_FromLong((long unsigned int)len);
#endif
}

static PyMethodDef vector_slerpiter_methods[] = {
    {"__length_hint__", (PyCFunction)vector_slerpiter_len, METH_NOARGS,
    },
    {NULL, NULL} /* sentinel */
};

static PyTypeObject PyVector_SlerpIter_Type = {
    TYPE_HEAD (NULL, 0)
    "pygame.math.VectorSlerpIterator", /* tp_name */
    sizeof(vector_slerpiter),        /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)vector_slerpiter_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_compare */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    0,                         /* tp_str */
    PyObject_GenericGetAttr,   /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    0,                         /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    PyObject_SelfIter,         /* tp_iter */
    (iternextfunc)vector_slerpiter_next, /* tp_iternext */
    vector_slerpiter_methods,        /* tp_methods */
    0,                         /* tp_members */
};

static PyObject *
vector_slerp(PyVector *self, PyObject *args)
{
    vector_slerpiter *it;
    PyObject *other, *steps_object;
    double vec1_coords[VECTOR_MAX_SIZE], vec2_coords[VECTOR_MAX_SIZE];
    double angle, length1, length2;

    if (!PyArg_ParseTuple(args, "OO:Vector.slerp", &other, &steps_object)) {
        return NULL;
    }
    if (!PyInt_Check(steps_object)) {
        PyErr_SetString(PyExc_TypeError, "Expected Int as argument 2");
        return NULL;
    }
    if (!PySequence_AsVectorCoords((PyObject*)self, vec1_coords, self->dim)) {
        PyErr_BadInternalCall();
        return NULL;
    }
    if (!PySequence_AsVectorCoords(other, vec2_coords, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "Expected Vector as argument 1");
        return NULL;
    }

    it = PyObject_New(vector_slerpiter, &PyVector_SlerpIter_Type);
    if (it == NULL)
        return NULL;
    it->it_index = 0;
    it->dim = self->dim;
    memcpy(it->coords, vec1_coords, sizeof(vec1_coords[0]) * it->dim);

    length1 = sqrt(_scalar_product(vec1_coords, vec1_coords, it->dim));
    length2 = sqrt(_scalar_product(vec2_coords, vec2_coords, it->dim));
    if ((length1 < self->epsilon) || (length2 < self->epsilon)) {
        PyErr_SetString(PyExc_ZeroDivisionError,
                        "can't use slerp with Zero-Vector");
        Py_DECREF(it);
        return NULL;
    }
    angle = acos(_scalar_product(vec1_coords, vec2_coords, self->dim) /
                 (length1 * length2));
    it->steps = PyInt_AsLong(steps_object);
    if (it->steps < 0) {
        angle -= 2 * M_PI;
        it->steps = -it->steps;
    }
    angle /= it->steps;

    if (fabs(length1 - length2) > self->epsilon)
        it->radial_factor = pow(length2 / length1, 1./it->steps);
    else
        it->radial_factor = 1;

    switch (self->dim) {
    case 2:
        _make_vector2_slerp_matrix(it, vec1_coords, vec2_coords, angle);
        break;
    case 3:
        _make_vector3_slerp_matrix(it, vec1_coords, vec2_coords, angle);
        break;
    default:
        PyErr_BadInternalCall();
        Py_DECREF(it);
        return NULL;        
    }
    return (PyObject *)it;
}

static void
_make_vector2_slerp_matrix(vector_slerpiter *it,
                           const double *vec1_coords, const double *vec2_coords,
                           double angle)
{
    double sin_value, cos_value;
    if (vec1_coords[0] * vec2_coords[1] < vec1_coords[1] * vec2_coords[0])
        angle *= -1;
    cos_value = cos(angle);
    sin_value = sin(angle);
    it->matrix[0][0] = cos_value;
    it->matrix[0][1] = sin_value;
    it->matrix[1][0] = -sin_value;
    it->matrix[1][1] = cos_value;
}

static void
_make_vector3_slerp_matrix(vector_slerpiter *it, 
                           const double *vec1_coords, const double *vec2_coords,
                           double angle)
{
    int i;
    double axis[3];
    double norm_factor, sin_value, cos_value, cos_complement;

    /* calculate rotation axis via cross-product */
    for (i = 0; i < 3; ++i)
        axis[i] = ((vec1_coords[(i+1)%3] * vec2_coords[(i+2)%3]) -
                   (vec1_coords[(i+2)%3] * vec2_coords[(i+1)%3]));
    /* normalize the rotation axis */
    norm_factor = 1. / sqrt(_scalar_product(axis, axis, 3));
    for (i = 0; i < 3; ++i)
        axis[i] *= norm_factor;

    sin_value = sin(angle);
    cos_value = cos(angle);
    cos_complement = 1 - cos_value;
    /* calculate the rotation matrix */
    it->matrix[0][0] = cos_value + axis[0] * axis[0] * cos_complement;
    it->matrix[0][1] = axis[0] * axis[1] * cos_complement + axis[2] * sin_value;
    it->matrix[0][2] = axis[0] * axis[2] * cos_complement - axis[1] * sin_value;
    it->matrix[1][0] = axis[0] * axis[1] * cos_complement - axis[2] * sin_value;
    it->matrix[1][1] = cos_value + axis[1] * axis[1] * cos_complement;
    it->matrix[1][2] = axis[1] * axis[2] * cos_complement + axis[0] * sin_value;
    it->matrix[2][0] = axis[0] * axis[2] * cos_complement + axis[1] * sin_value;
    it->matrix[2][1] = axis[1] * axis[2] * cos_complement - axis[0] * sin_value;
    it->matrix[2][2] = cos_value + axis[2] * axis[2] * cos_complement;
}

/*
static PyObject *
vector2_slerp(PyVector *self, PyObject *args)
{
    PyObject *other;
    PyVector *ret;
    double *other_coords;
    double t;
    if (!PyArg_ParseTuple("Od:slerp", &other, &t)) {
        return NULL;
    }
    if (!PyVectorCompatible_Check(end_vector, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "Argument 1 must be a vector.");
        return NULL;
    }
    if (!PySequence_AsVectorCoords(other, other_coords, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "Argument 1 must be a vector.");
        return NULL;
    }
    if (fabs(t) > 1) {
        PyErr_SetString(PyExc_ValueError, "Argument 2 must be in range [-1, 1].");
        return NULL;
    }
    
    length1 = sqrt(_scalar_product(self->coords, self->coords, self->dim));
    length2 = sqrt(_scalar_product(other_coords, other_coords, self->dim));
    if ((length1 < self->epsilon) || (length2 < self->epsilon)) {
        PyErr_SetString(PyExc_ZeroDivisionError,
                        "can't use slerp with Zero-Vector");
        return NULL;
    }
    angle = acos(_scalar_product(self->coords, other_coords, self->dim) /
                 (length1 * length2));
    if (t < 0) {
        angle -= 2 * M_PI;
        t = -t;
    }
    angle *= t;
    if (vec1_coords[0] * vec2_coords[1] < vec1_coords[1] * vec2_coords[0])
        angle *= -1;

    if (fabs(length1 - length2) > self->epsilon)
        it->radial_factor = t * length2 / length1;
    else
        it->radial_factor = 1;

    ret = PyVector_NEW(self->dim);
    cos_value = cos(angle);
    sin_value = sin(angle);
    ret->coords[0] = cos_value * self->coords[0] - sin_value * self->coords[1];
    ret->coords[1] = sin_value * self->coords[0] + cos_value * self->coords[1];
    n1 = 1. / length1;
    n2 = 1. / length2;
    s0 = 1. / sin(angle);
    s1 = sin(angle * t);
    s2 = sin(angle * (1-t));
    for (i = 0; i < self->dim; ++i)
        ret->coords[i] = (self->coords[i] * s1 + other_coords[i] * s2) * s0;
    return ret;
}
*/


/********************************************
 * PyVector_LerpIterator type definition
 ********************************************/
static void
vector_lerpiter_dealloc(vector_lerpiter *it)
{
    PyObject_Del(it);
}

static PyObject *
vector_lerpiter_next(vector_lerpiter *it)
{
    int i;
    PyVector *ret;
    assert(it != NULL);

    if (it->it_index < it->steps) {
        ret = (PyVector*)PyVector_NEW(it->dim);
        if (ret != NULL) {
            for (i = 0; i < it->dim; ++i) {
                it->coords[i] += it->step_vec[i];
                ret->coords[i] = it->coords[i];
            }
        }
        ++(it->it_index);
        return (PyObject*)ret;
    }
    
    return NULL;
}

static PyObject *
vector_lerpiter_len(vector_lerpiter *it)
{
    Py_ssize_t len = 0;
    if (it) {
        len = it->steps - it->it_index;
    }
#if PY_VERSION_HEX >= 0x02050000
    return PyInt_FromSsize_t(len);
#else
    return PyInt_FromLong((long unsigned int)len);
#endif
}

static PyMethodDef vector_lerpiter_methods[] = {
    {"__length_hint__", (PyCFunction)vector_lerpiter_len, METH_NOARGS,
    },
    {NULL, NULL} /* sentinel */
};

static PyTypeObject PyVector_LerpIter_Type = {
    TYPE_HEAD (NULL, 0)
    "pygame.math.VectorLerpIterator", /* tp_name */
    sizeof(vector_lerpiter),        /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)vector_lerpiter_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_compare */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    0,                         /* tp_str */
    PyObject_GenericGetAttr,   /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    0,                         /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    PyObject_SelfIter,         /* tp_iter */
    (iternextfunc)vector_lerpiter_next, /* tp_iternext */
    vector_lerpiter_methods,        /* tp_methods */
    0,                         /* tp_members */
};


static PyObject *
vector_lerp(PyVector *self, PyObject *args)
{
    vector_lerpiter *it;
    PyObject *other, *steps_object;
    long int i;
    double vec1_coords[VECTOR_MAX_SIZE], vec2_coords[VECTOR_MAX_SIZE];

    if (!PyArg_ParseTuple(args, "OO:Vector.lerp", &other, &steps_object)) {
        return NULL;
    }
    if (!PyInt_Check(steps_object)) {
        PyErr_SetString(PyExc_TypeError, "Expected Int as argument 2");
        return NULL;
    }
    if (!PySequence_AsVectorCoords((PyObject*)self, vec1_coords, self->dim)) {
        PyErr_BadInternalCall();
        return NULL;
    }
    if (!PySequence_AsVectorCoords(other, vec2_coords, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "Expected Vector as argument 1");
        return NULL;
    }

    it = PyObject_New(vector_lerpiter, &PyVector_LerpIter_Type);
    if (it == NULL)
        return NULL;
    it->it_index = 0;
    it->dim = self->dim;
    it->steps = PyInt_AsLong(steps_object);
    memcpy(it->coords, vec1_coords, sizeof(vec1_coords[0]) * it->dim);

    for (i = 0; i < it->dim; ++i)
        it->step_vec[i] = (vec2_coords[i] - vec1_coords[i]) / it->steps;
    return (PyObject *)it;
}









/*****************************************
 * elementwiseproxy
 *****************************************/



static void
vector_elementwiseproxy_dealloc(vector_elementwiseproxy *it)
{
    Py_XDECREF(it->vec);
    PyObject_Del(it);
}

static PyObject *
vector_elementwiseproxy_richcompare(PyObject *o1, PyObject *o2, int op)
{
    int i, dim;
    double diff;
    PyVector *vec;
    PyObject *other;

    if (vector_elementwiseproxy_Check(o1)) {
        vec = ((vector_elementwiseproxy*)o1)->vec;
        other = o2;
    }
    else {
        vec = ((vector_elementwiseproxy*)o2)->vec;
        other = o1;
        /* flip op */
        if (op == Py_LT)
            op = Py_GE;
        else if (op == Py_LE)
            op = Py_GT;
        else if (op == Py_GT)
            op = Py_LE;
        else if (op == Py_GE)
            op = Py_LT;
    }
    if (vector_elementwiseproxy_Check(other))
        other = (PyObject*)((vector_elementwiseproxy*)o2)->vec;
    dim = vec->dim;

    if (PyVectorCompatible_Check(other, dim)) {
        /* use diff == diff to check for NaN */
        /* TODO: how should NaN be handled with LT/LE/GT/GE? */
        switch (op) {
        case Py_EQ:
            for (i = 0; i < dim; i++) {
                diff = vec->coords[i] - PySequence_GetItem_AsDouble(other, i);
                if ((diff != diff) || (fabs(diff) >= vec->epsilon)) {
                    Py_RETURN_FALSE;
                }
            }
            Py_RETURN_TRUE;
        case Py_NE:
            for (i = 0; i < dim; i++) {
                diff = vec->coords[i] - PySequence_GetItem_AsDouble(other, i);
                if ((diff == diff) && (fabs(diff) < vec->epsilon)) {
                    Py_RETURN_FALSE;
                }
            }
            Py_RETURN_TRUE;
        case Py_LT:
            for (i = 0; i < dim; i++) {
                if (vec->coords[i] >= PySequence_GetItem_AsDouble(other, i))
                    Py_RETURN_FALSE;
            }
            Py_RETURN_TRUE;
        case Py_LE:
            for (i = 0; i < dim; i++) {
                if (vec->coords[i] > PySequence_GetItem_AsDouble(other, i))
                    Py_RETURN_FALSE;
            }
            Py_RETURN_TRUE;
        case Py_GT:
            for (i = 0; i < dim; i++) {
                if (vec->coords[i] <= PySequence_GetItem_AsDouble(other, i))
                    Py_RETURN_FALSE;
            }
            Py_RETURN_TRUE;
        case Py_GE:
            for (i = 0; i < dim; i++) {
                if (vec->coords[i] < PySequence_GetItem_AsDouble(other, i))
                    Py_RETURN_FALSE;
            }
            Py_RETURN_TRUE;
        }
    }
    else if (RealNumber_Check(other)) {
        double value = PyFloat_AsDouble(other);
        switch (op) {
        case Py_EQ:
            for (i = 0; i < dim; i++) {
                diff = vec->coords[i] - value;
                if (diff != diff || fabs(diff) >= vec->epsilon) {
                    Py_RETURN_FALSE;
                }
            }
            Py_RETURN_TRUE;
        case Py_NE:
            for (i = 0; i < dim; i++) {
                diff = vec->coords[i] - value;
                if (diff == diff && fabs(diff) < vec->epsilon) {
                    Py_RETURN_FALSE;
                }
            }
            Py_RETURN_TRUE;
        case Py_LT:
            for (i = 0; i < dim; i++) {
                if (vec->coords[i] >= value) 
                    Py_RETURN_FALSE;
            }
            Py_RETURN_TRUE;
        case Py_LE:
            for (i = 0; i < dim; i++) {
                if (vec->coords[i] > value) 
                    Py_RETURN_FALSE;
            }
            Py_RETURN_TRUE;
        case Py_GT:
            for (i = 0; i < dim; i++) {
                if (vec->coords[i] <= value) 
                    Py_RETURN_FALSE;
            }
            Py_RETURN_TRUE;
        case Py_GE:
            for (i = 0; i < dim; i++) {
                if (vec->coords[i] < value) 
                    Py_RETURN_FALSE;
            }
            Py_RETURN_TRUE;
        }
    }

    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}




/*******************************************************
 * vector_elementwiseproxy PyNumber emulation routines
 *******************************************************/

static PyObject *
vector_elementwiseproxy_generic_math(PyObject *o1, PyObject *o2, int op)
{
    /* TODO: DIV, FLOOR_DIV and MOD should check for ZeroDivision */
    int i, dim;
    double tmp, mod;
    PyObject *other;
    PyVector *vec, *ret;
    if (vector_elementwiseproxy_Check(o1)) {
        vec = ((vector_elementwiseproxy*)o1)->vec;
        other = o2;
    }
    else {
        other = o1;
        vec = ((vector_elementwiseproxy*)o2)->vec;
        op |= OP_ARG_REVERSE;
    }

    dim = vec->dim;

    if (vector_elementwiseproxy_Check(other)) {
        other = (PyObject*)((vector_elementwiseproxy*)other)->vec;
        op |= OP_ARG_VECTOR;
    }
    else if (PyVectorCompatible_Check(other, dim))
        op |= OP_ARG_VECTOR;
    else if (RealNumber_Check(other))
        op |= OP_ARG_NUMBER;
    else
        op |= OP_ARG_UNKNOWN;

    switch (op) {
    case OP_ADD | OP_ARG_VECTOR:
    case OP_ADD | OP_ARG_VECTOR | OP_ARG_REVERSE:
        return vector_add((PyObject*)vec, other);
    case OP_ADD | OP_ARG_NUMBER:
    case OP_ADD | OP_ARG_NUMBER | OP_ARG_REVERSE:
        tmp = PyFloat_AsDouble(other);
        ret = (PyVector*)PyVector_NEW(dim);
        for (i = 0; i < dim; i++)
            ret->coords[i] = vec->coords[i] + tmp;
        return (PyObject*)ret;
    case OP_SUB | OP_ARG_VECTOR:
        return vector_sub((PyObject*)vec, other);
    case OP_SUB | OP_ARG_VECTOR | OP_ARG_REVERSE:
        return vector_sub(other, (PyObject*)vec);
    case OP_SUB | OP_ARG_NUMBER:
        tmp = PyFloat_AsDouble(other);
        ret = (PyVector*)PyVector_NEW(dim);
        for (i = 0; i < dim; i++)
            ret->coords[i] = vec->coords[i] - tmp;
        return (PyObject*)ret;
    case OP_SUB | OP_ARG_NUMBER | OP_ARG_REVERSE:
        tmp = PyFloat_AsDouble(other);
        ret = (PyVector*)PyVector_NEW(dim);
        for (i = 0; i < dim; i++)
            ret->coords[i] = tmp - vec->coords[i];
        return (PyObject*)ret;
    case OP_MUL | OP_ARG_VECTOR:
    case OP_MUL | OP_ARG_VECTOR | OP_ARG_REVERSE:
        ret = (PyVector*)PyVector_NEW(vec->dim);
        for (i = 0; i < vec->dim; i++) {
            ret->coords[i] = (vec->coords[i] * 
                              PySequence_GetItem_AsDouble(other, i));
        }
        return (PyObject*)ret;
    case OP_MUL | OP_ARG_NUMBER:
    case OP_MUL | OP_ARG_NUMBER | OP_ARG_REVERSE:
        return vector_mul((PyObject*)vec, other);
    case OP_DIV | OP_ARG_VECTOR:
        ret = (PyVector*)PyVector_NEW(vec->dim);
        for (i = 0; i < vec->dim; i++) {
            ret->coords[i] = (vec->coords[i] /
                              PySequence_GetItem_AsDouble(other, i));
        }
        return (PyObject*)ret;
    case OP_DIV | OP_ARG_VECTOR | OP_ARG_REVERSE:
        ret = (PyVector*)PyVector_NEW(vec->dim);
        for (i = 0; i < vec->dim; i++) {
            ret->coords[i] = (PySequence_GetItem_AsDouble(o1, i) /
                              vec->coords[i]);
        }
        return (PyObject*)ret;
    case OP_DIV | OP_ARG_NUMBER:
        return vector_div(vec, other);
    case OP_DIV | OP_ARG_NUMBER | OP_ARG_REVERSE:
        tmp = PyFloat_AsDouble(o1);
        ret = (PyVector*)PyVector_NEW(vec->dim);
        for (i = 0; i < vec->dim; i++)
            ret->coords[i] = tmp / vec->coords[i];
        return (PyObject*)ret;
    case OP_FLOOR_DIV | OP_ARG_VECTOR:
        ret = (PyVector*)PyVector_NEW(vec->dim);
        for (i = 0; i < vec->dim; i++) {
            ret->coords[i] = floor(vec->coords[i] /
                                   PySequence_GetItem_AsDouble(other, i));
        }
        return (PyObject*)ret;
    case OP_FLOOR_DIV | OP_ARG_VECTOR | OP_ARG_REVERSE:
        ret = (PyVector*)PyVector_NEW(vec->dim);
        for (i = 0; i < vec->dim; i++) {
            ret->coords[i] = floor(PySequence_GetItem_AsDouble(o1, i) /
                                   vec->coords[i]);
        }
        return (PyObject*)ret;
    case OP_FLOOR_DIV | OP_ARG_NUMBER:
        return vector_floor_div(vec, other);
    case OP_FLOOR_DIV | OP_ARG_NUMBER | OP_ARG_REVERSE:
        tmp = PyFloat_AsDouble(other);
        ret = (PyVector*)PyVector_NEW(vec->dim);
        for (i = 0; i < vec->dim; i++)
            ret->coords[i] = floor(tmp / vec->coords[i]);
        return (PyObject*)ret;
    case OP_MOD | OP_ARG_VECTOR:
        ret = (PyVector*)PyVector_NEW(vec->dim);
        for (i = 0; i < vec->dim; i++) {
            tmp = PySequence_GetItem_AsDouble(other, i);
            mod = fmod(vec->coords[i], tmp);
            /* note: checking mod*value < 0 is incorrect -- underflows to
               0 if value < sqrt(smallest nonzero double) */
            if (mod && ((tmp < 0) != (mod < 0))) {
                mod += tmp;
            }
            ret->coords[i] = mod;
        }
        return (PyObject*)ret;
    case OP_MOD | OP_ARG_VECTOR | OP_ARG_REVERSE:
        ret = (PyVector*)PyVector_NEW(vec->dim);
        for (i = 0; i < vec->dim; i++) {
            tmp = PySequence_GetItem_AsDouble(other, i);
            mod = fmod(tmp, vec->coords[i]);
            /* note: see above */
            if (mod && ((vec->coords[i] < 0) != (mod < 0))) {
                mod += vec->coords[i];
            }
            ret->coords[i] = mod;
        }
        return (PyObject*)ret;
    case OP_MOD | OP_ARG_NUMBER:
        ret = (PyVector*)PyVector_NEW(vec->dim);
        tmp = PyFloat_AsDouble(other);
        for (i = 0; i < vec->dim; i++) {
            mod = fmod(vec->coords[i], tmp);
            /* note: see above */
            if (mod && ((tmp < 0) != (mod < 0))) {
                mod += tmp;
            }
            ret->coords[i] = mod;
        }
        return (PyObject*)ret;
    case OP_MOD | OP_ARG_NUMBER | OP_ARG_REVERSE:
        ret = (PyVector*)PyVector_NEW(vec->dim);
        tmp = PyFloat_AsDouble(other);
        for (i = 0; i < vec->dim; i++) {
            mod = fmod(tmp, vec->coords[i]);
            /* note: see above */
            if (mod && ((vec->coords[i] < 0) != (mod < 0))) {
                mod += vec->coords[i];
            }
            ret->coords[i] = mod;
        }
        return (PyObject*)ret;
    default:
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
}

static PyObject *
vector_elementwiseproxy_add(PyObject *o1, PyObject *o2)
{
    return vector_elementwiseproxy_generic_math(o1, o2, OP_ADD);
}
static PyObject *
vector_elementwiseproxy_sub(PyObject *o1, PyObject *o2)
{
    return vector_elementwiseproxy_generic_math(o1, o2, OP_SUB);
}
static PyObject *
vector_elementwiseproxy_mul(PyObject *o1, PyObject *o2)
{
    return vector_elementwiseproxy_generic_math(o1, o2, OP_MUL);
}
static PyObject *
vector_elementwiseproxy_div(PyObject *o1, PyObject *o2)
{
    return vector_elementwiseproxy_generic_math(o1, o2, OP_DIV);
}
static PyObject *
vector_elementwiseproxy_floor_div(PyObject *o1, PyObject *o2)
{
    return vector_elementwiseproxy_generic_math(o1, o2, OP_FLOOR_DIV);
}
static PyObject *
vector_elementwiseproxy_mod(PyObject *o1, PyObject *o2)
{
    return vector_elementwiseproxy_generic_math(o1, o2, OP_MOD);
}

static PyObject *
vector_elementwiseproxy_pow(PyObject *baseObj, PyObject *expoObj, PyObject *mod)
{
    int i, dim;
    double tmp;
    double bases[VECTOR_MAX_SIZE];
    double expos[VECTOR_MAX_SIZE];
    PyVector *ret;
    PyObject *base, *expo, *result;
    if (mod != Py_None) {
        PyErr_SetString(PyExc_TypeError, "pow() 3rd argument not "
                        "allowed for vectors");
        return NULL;
    }

    if (vector_elementwiseproxy_Check(baseObj)) {
        dim = ((vector_elementwiseproxy*)baseObj)->vec->dim;
        memcpy(bases, ((vector_elementwiseproxy*)baseObj)->vec->coords,
               sizeof(double) * dim);
        if (vector_elementwiseproxy_Check(expoObj)) {
            memcpy(expos, ((vector_elementwiseproxy*)expoObj)->vec->coords,
                   sizeof(double) * dim);
        }
        else if (PyVectorCompatible_Check(expoObj, dim)) {
            PySequence_AsVectorCoords(expoObj, expos, dim);
        }
        else if (RealNumber_Check(expoObj)) {
            tmp = PyFloat_AsDouble(expoObj);
            for (i = 0; i < dim; i++)
                expos[i] = tmp;
        }
        else
            goto NOT_IMPLEMENTED;
    }
    else {
        dim = ((vector_elementwiseproxy*)expoObj)->vec->dim;
        memcpy(expos, ((vector_elementwiseproxy*)expoObj)->vec->coords,
               sizeof(double) * dim);
        if (PyVectorCompatible_Check(baseObj, dim)) {
            PySequence_AsVectorCoords(baseObj, bases, dim);
        }
        else if (RealNumber_Check(baseObj)) {
            tmp = PyFloat_AsDouble(baseObj);
            for (i = 0; i < dim; i++)
                bases[i] = tmp;
        }
        else
            goto NOT_IMPLEMENTED;
    }

    ret = (PyVector*)PyVector_NEW(dim);
    /* there are many special cases so we let python do the work for now */
    for (i = 0; i < dim; i++) {
        base = PyFloat_FromDouble(bases[i]);
        expo = PyFloat_FromDouble(expos[i]);
        result = PyNumber_Power(base, expo, Py_None);
        if (!result)
            return NULL;
        ret->coords[i] = PyFloat_AsDouble(result);
        Py_DECREF(result);
        Py_DECREF(expo);
        Py_DECREF(base);
    }
    return (PyObject*)ret;

NOT_IMPLEMENTED:
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}


static PyObject *
vector_elementwiseproxy_abs(vector_elementwiseproxy *self)
{
    int i;
    PyVector *ret = (PyVector*)PyVector_NEW(self->vec->dim);
    for (i = 0; i < self->vec->dim; i++) {
        ret->coords[i] = fabs(self->vec->coords[i]);
    }
    return (PyObject*)ret;
}

static PyObject *
vector_elementwiseproxy_neg(vector_elementwiseproxy *self)
{
    int i;
    PyVector *ret = (PyVector*)PyVector_NEW(self->vec->dim);
    for (i = 0; i < self->vec->dim; i++) {
        ret->coords[i] = -self->vec->coords[i];
    }
    return (PyObject*)ret;
}

static PyObject *
vector_elementwiseproxy_pos(vector_elementwiseproxy *self)
{
    PyVector *ret = (PyVector*)PyVector_NEW(self->vec->dim);
    memcpy(ret->coords, self->vec->coords, sizeof(ret->coords[0]) * ret->dim);
    return (PyObject*)ret;
}

static int
vector_elementwiseproxy_nonzero(vector_elementwiseproxy *self)
{
    int i;
    for (i = 0; i < self->vec->dim; i++) {
        if (fabs(self->vec->coords[i]) > self->vec->epsilon) {
            return 1;
        }
    }
    return 0;
}

static PyNumberMethods vector_elementwiseproxy_as_number = {
    (binaryfunc)vector_elementwiseproxy_add,      /* nb_add;       __add__ */
    (binaryfunc)vector_elementwiseproxy_sub,      /* nb_subtract;  __sub__ */
    (binaryfunc)vector_elementwiseproxy_mul,      /* nb_multiply;  __mul__ */
#if !PY3
    (binaryfunc)vector_elementwiseproxy_div,      /* nb_divide;    __div__ */
#endif
    (binaryfunc)vector_elementwiseproxy_mod,      /* nb_remainder; __mod__ */
    (binaryfunc)0,                                /* nb_divmod;    __divmod__ */
    (ternaryfunc)vector_elementwiseproxy_pow,     /* nb_power;     __pow__ */
    (unaryfunc)vector_elementwiseproxy_neg, /* nb_negative;  __neg__ */
    (unaryfunc)vector_elementwiseproxy_pos, /* nb_positive;  __pos__ */
    (unaryfunc)vector_elementwiseproxy_abs, /* nb_absolute;  __abs__ */
    (inquiry)vector_elementwiseproxy_nonzero, /* nb_nonzero;   __nonzero__ */
    (unaryfunc)0,                   /* nb_invert;    __invert__ */
    (binaryfunc)0,                  /* nb_lshift;    __lshift__ */
    (binaryfunc)0,                  /* nb_rshift;    __rshift__ */
    (binaryfunc)0,                  /* nb_and;       __and__ */
    (binaryfunc)0,                  /* nb_xor;       __xor__ */
    (binaryfunc)0,                  /* nb_or;        __or__ */
#if !PY3
    (coercion)0,                    /* nb_coerce;    __coerce__ */
#endif
    (unaryfunc)0,                   /* nb_int;       __int__ */
    (unaryfunc)0,                   /* nb_long;      __long__ */
    (unaryfunc)0,                   /* nb_float;     __float__ */
#if !PY3
    (unaryfunc)0,                   /* nb_oct;       __oct__ */
    (unaryfunc)0,                   /* nb_hex;       __hex__ */
#endif
    /* Added in release 2.0 */
    (binaryfunc)0,                  /* nb_inplace_add;       __iadd__ */
    (binaryfunc)0,                  /* nb_inplace_subtract;  __isub__ */
    (binaryfunc)0,                  /* nb_inplace_multiply;  __imul__ */
#if !PY3
    (binaryfunc)0,                  /* nb_inplace_divide;    __idiv__ */
#endif
    (binaryfunc)0,                  /* nb_inplace_remainder; __imod__ */
    (ternaryfunc)0,                 /* nb_inplace_power;     __pow__ */
    (binaryfunc)0,                  /* nb_inplace_lshift;    __ilshift__ */
    (binaryfunc)0,                  /* nb_inplace_rshift;    __irshift__ */
    (binaryfunc)0,                  /* nb_inplace_and;       __iand__ */
    (binaryfunc)0,                  /* nb_inplace_xor;       __ixor__ */
    (binaryfunc)0,                  /* nb_inplace_or;        __ior__ */

    /* Added in release 2.2 */
    (binaryfunc)vector_elementwiseproxy_floor_div, /* nb_floor_divide;         __floor__ */
    (binaryfunc)vector_elementwiseproxy_div, /* nb_true_divide;          __truediv__ */
    (binaryfunc)0,                  /* nb_inplace_floor_divide; __ifloor__ */
    (binaryfunc)0,                  /* nb_inplace_true_divide;  __itruediv__ */
};




static PyTypeObject PyVectorElementwiseProxy_Type = {
    TYPE_HEAD (NULL, 0)
    "pygame.math.VectorElementwiseProxy", /* tp_name */
    sizeof(vector_elementwiseproxy), /* tp_basicsize */
    0,                         /* tp_itemsize */
    /* Methods to implement standard operations */
    (destructor)vector_elementwiseproxy_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_compare */
    (reprfunc)0,               /* tp_repr */
    /* Method suites for standard classes */
    &vector_elementwiseproxy_as_number, /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    /* More standard operations (here for binary compatibility) */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    (reprfunc)0,               /* tp_str */
    (getattrofunc)0,           /* tp_getattro */
    (setattrofunc)0,           /* tp_setattro */
    /* Functions to access object as input/output buffer */
    0,                         /* tp_as_buffer */
    /* Flags to define presence of optional/expanded features */
#if PY3
    Py_TPFLAGS_DEFAULT,
#else
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES, /* tp_flags */
#endif
    /* Documentation string */
    0,                         /* tp_doc */

    /* Assigned meaning in release 2.0 */
    /* call function for all accessible objects */
    0,                         /* tp_traverse */
    /* delete references to contained objects */
    0,                         /* tp_clear */

    /* Assigned meaning in release 2.1 */
    /* rich comparisons */
    (richcmpfunc)vector_elementwiseproxy_richcompare, /* tp_richcompare */
    /* weak reference enabler */
    0,                         /* tp_weaklistoffset */

    /* Added in release 2.2 */
    /* Iterators */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    /* Attribute descriptor and subclassing stuff */
    0,                         /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)0,               /* tp_init */
    0,                         /* tp_alloc */
    (newfunc)0,                /* tp_new */
    0,                         /* tp_free */
    0,                         /* tp_is_gc */
    0,                         /* tp_bases */
    0,                         /* tp_mro */
    0,                         /* tp_cache */
    0,                         /* tp_subclasses */
    0,                         /* tp_weaklist */
};



static PyObject *
vector_elementwise(PyVector *vec)
{
    vector_elementwiseproxy *proxy;
    if (!PyVector_Check(vec)) {
        PyErr_BadInternalCall();
        return NULL;
    }

    proxy = PyObject_New(vector_elementwiseproxy, 
                         &PyVectorElementwiseProxy_Type);
    if (proxy == NULL)
        return NULL;
    Py_INCREF(vec);
    proxy->vec = (PyVector *)vec;
    return (PyObject *)proxy;
}







static PyObject *
math_enable_swizzling(PyVector *self)
{
    swizzling_enabled = 1;
    Py_RETURN_NONE;
}

static PyObject *
math_disable_swizzling(PyVector *self)
{
    swizzling_enabled = 0;
    Py_RETURN_NONE;
}

static PyMethodDef _math_methods[] =
{
    {"enable_swizzling", (PyCFunction)math_enable_swizzling, METH_NOARGS,
     "enables swizzling."
    },
    {"disable_swizzling", (PyCFunction)math_disable_swizzling, METH_NOARGS,
     "disables swizzling."
    },
    {NULL, NULL, 0, NULL}
};


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
        DOC_PYGAMEMATH,
        -1,
        _math_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    /* initialize the extension types */
    if ((PyType_Ready(&PyVector2_Type) < 0) || 
        (PyType_Ready(&PyVector3_Type) < 0) ||
        (PyType_Ready(&PyVectorElementwiseProxy_Type) < 0) ||
        (PyType_Ready(&PyVectorIter_Type) < 0) ||
        (PyType_Ready(&PyVector_SlerpIter_Type) < 0) /*||
        (PyType_Ready(&PyVector4_Type) < 0)*/) {
        MODINIT_ERROR;
    }

    /* initialize the module */
#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 (MODPREFIX "math", _math_methods, DOC_PYGAMEMATH);
#endif

    if (module == NULL) {
        MODINIT_ERROR;
    }

    /* add extension types to module */
    Py_INCREF(&PyVector2_Type);
    Py_INCREF(&PyVector3_Type);
    Py_INCREF(&PyVector3_Type);
    Py_INCREF(&PyVectorElementwiseProxy_Type);
    Py_INCREF(&PyVectorIter_Type);
    Py_INCREF(&PyVector_SlerpIter_Type);
//    Py_INCREF(&PyVector4_Type);
    if ((PyModule_AddObject(module, "Vector2", (PyObject *)&PyVector2_Type) != 0) ||
        (PyModule_AddObject(module, "Vector3", (PyObject *)&PyVector3_Type) != 0) ||
        (PyModule_AddObject(module, "VectorElementwiseProxy", (PyObject *)&PyVectorElementwiseProxy_Type) != 0) ||
        (PyModule_AddObject(module, "VectorIterator", (PyObject *)&PyVectorIter_Type) != 0) ||
        (PyModule_AddObject(module, "VectorSlerpIterator", (PyObject *)&PyVector_SlerpIter_Type) != 0) /*||
        (PyModule_AddObject(module, "Vector4", (PyObject *)&PyVector4_Type) != 0)*/) {
        Py_DECREF(&PyVector2_Type);
        Py_DECREF(&PyVector3_Type);
        Py_DECREF(&PyVectorElementwiseProxy_Type);
        Py_DECREF(&PyVectorIter_Type);
        Py_DECREF(&PyVector_SlerpIter_Type);
//        Py_DECREF(&PyVector4_Type);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    /* export the C api */
    c_api[0] = &PyVector2_Type;
    c_api[1] = &PyVector3_Type;
//    c_api[2] = &PyVector4_Type;
//    c_api[3] = PyVector_NEW;
//    c_api[4] = PyVectorCompatible_Check;
    apiobj = PyCObject_FromVoidPtr(c_api, NULL);
    if (apiobj == NULL) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj) != 0) {
        Py_DECREF (apiobj);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    MODINIT_RETURN (module);
}
