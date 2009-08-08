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

#define VECTOR_MAX_SIZE (4)
#define STRING_BUF_SIZE (100)
#define SWIZZLE_ERR_NO_ERR         0
#define SWIZZLE_ERR_DOUBLE_IDX     1
#define SWIZZLE_ERR_EXTRACTION_ERR 2

static PyTypeObject PyVector2_Type;
static PyTypeObject PyVectorElementwiseProxy_Type;
#define PyVector2_Check(x) ((x)->ob_type == &PyVector2_Type)
#define PyVector_Check(x) (PyVector2_Check(x))
#define vector_elementwiseproxy_Check(x) \
    ((x)->ob_type == &PyVectorElementwiseProxy_Type)

#define DEG2RAD(angle) ((angle) * M_PI / 180.)
#define RAD2DEG(angle) ((angle) * 180. / M_PI)



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
PySequence_AsVectorCoords(PyObject *seq, double *coords, const int size)
{
    int i;
    PyObject *item;
    PyObject *fltobj;

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
        else if (RealNumber_Check(o2)) {
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
        else if (RealNumber_Check(o1)) {
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
    if (RealNumber_Check(other)) {
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
    if (RealNumber_Check(other)) {
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
    if (RealNumber_Check(other)) {
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
    if (RealNumber_Check(other)) {
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
    if (RealNumber_Check(other)) {
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


PySequenceMethods vector_as_sequence = {
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
    /* NOTE: this is never called when swizzling is enabled */
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
    /* NOTE: this is never called when swizzling is enabled */
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
    double diff;
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
            diff = self->coords[i] - PySequence_GetItem_AsDouble(other, i);
            // test diff != diff to catch NaN
            if ((diff != diff) || (fabs(diff) >= self->epsilon)) {
                Py_RETURN_FALSE;
            }
        }
        Py_RETURN_TRUE;
        break;
    case Py_NE:
        for (i = 0; i < self->dim; i++) {
            diff = self->coords[i] - PySequence_GetItem_AsDouble(other, i);
            if ((diff != diff) || (fabs(diff) >= self->epsilon)) {
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

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|OO:Vector2", kwlist,
                                      &xOrSequence, &y))
        return -1;

    if (xOrSequence) {
        if (RealNumber_Check(xOrSequence)) {
            self->coords[0] = PyFloat_AsDouble(xOrSequence);
        } 
        else if (checkPyVectorCompatible(xOrSequence, self->dim)) {
            self->coords[0] = PySequence_GetItem_AsDouble(xOrSequence, 0);
            self->coords[1] = PySequence_GetItem_AsDouble(xOrSequence, 1);
            /* successful initialization from sequence type */
            return 0;
        } 
        else if (PyString_Check(xOrSequence)) {
            /* This should make "Vector2(Vector2().__repr__())" possible */
            char buffer[STRING_BUF_SIZE];
            char *endptr;
            char *str = PyString_AsString(xOrSequence);
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

static void
vector2_do_rotate(double *dst_coords, const double *src_coords, double angle)
{
    /* make sure angle is in range [0, 360) */
    angle = fmod(angle, 360.);
    if (angle < 0)
        angle += 360.;

    /* special-case rotation by 0, 90, 180 and 270 degrees */
    if (angle == 0.) {
        dst_coords[0] = src_coords[0];
        dst_coords[1] = src_coords[1];
    }
    else if (angle == 90.) {
        dst_coords[0] = -src_coords[1];
        dst_coords[1] = src_coords[0];
    }
    else if (angle == 180.) {
        dst_coords[0] = -src_coords[0];
        dst_coords[1] = -src_coords[1];
    }
    else if (angle == 270.) {
        dst_coords[0] = src_coords[1];
        dst_coords[1] = -src_coords[0];
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
    vector2_do_rotate(ret->coords, self->coords, angle);
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
    vector2_do_rotate(self->coords, tmp, angle);
    Py_RETURN_NONE;
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
vector2_cross(PyVector *self, PyObject *other)
{
    if (!checkPyVectorCompatible(other, self->dim)) {
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
vector_angle_to(PyVector *self, PyObject *other)
{
    double angle;
    if (!checkPyVectorCompatible(other, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "expected an vector.");
        return NULL;
    }
    
    angle = (atan2(PySequence_GetItem_AsDouble(other, 1),
                   PySequence_GetItem_AsDouble(other, 0)) - 
             atan2(self->coords[1], self->coords[0]));
    return PyFloat_FromDouble(RAD2DEG(angle));
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

static PyObject *
vector_reflect(PyVector *self, PyObject *normal)
{
    int i, dim = self->dim;
    PyVector *ret;
    double dot_product;
    double norm_length;
    /* allocate enough space for 2, 3 and 4 dim vectors */
    double norm_coords[4];

    if (!checkPyVectorCompatible(normal, dim)) {
        PyErr_SetString(PyExc_TypeError, "Expected a vector.");
        return NULL;
    }

    /* normalize the normal */
    norm_length = 0;
    for (i = 0; i < dim; ++i) {
        norm_coords[i] = PySequence_GetItem_AsDouble(normal, i);
        norm_length += norm_coords[i] * norm_coords[i];
    }
    if (norm_length < self->epsilon) {
        PyErr_SetString(PyExc_ZeroDivisionError, "Normal must not be of length zero.");
        return NULL;
    }
    if (norm_length != 1) {
        norm_length = sqrt(norm_length);
        for (i = 0; i < dim; ++i)
            norm_coords[i] /= norm_length;
    }
    
    /* calculate the dot_product for the projection */
    dot_product = 0;
    for (i = 0; i < dim; ++i)
        dot_product += self->coords[i] * norm_coords[i];
    
    ret = (PyVector *)PyVector_NEW(dim);
    for (i = 0; i < dim; ++i)
        ret->coords[i] = self->coords[i] - 2 * norm_coords[i] * dot_product;

    return (PyObject *)ret;
}

static PyObject *
vector_reflect_ip(PyVector *self, PyObject *normal)
{
    int i, dim = self->dim;
    double dot_product;
    double norm_length;
    /* allocate enough space for 2, 3 and 4 dim vectors */
    double norm_coords[4];

    if (!checkPyVectorCompatible(normal, dim)) {
        PyErr_SetString(PyExc_TypeError, "Expected a vector.");
        return NULL;
    }

    /* normalize the normal */
    norm_length = 0;
    for (i = 0; i < dim; ++i) {
        norm_coords[i] = PySequence_GetItem_AsDouble(normal, i);
        norm_length += norm_coords[i] * norm_coords[i];
    }
    if (norm_length < self->epsilon) {
        PyErr_SetString(PyExc_ZeroDivisionError, "Normal must not be of length zero.");
        return NULL;
    }
    if (norm_length != 1) {
        norm_length = sqrt(norm_length);
        for (i = 0; i < dim; ++i)
            norm_coords[i] /= norm_length;
    }
    
    /* calculate the dot_product for the projection */
    dot_product = 0;
    for (i = 0; i < dim; ++i)
        dot_product += self->coords[i] * norm_coords[i];
    
    for (i = 0; i < dim; ++i)
        self->coords[i] -= 2 * norm_coords[i] * dot_product;

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
        distance_squared +=  tmp * tmp;
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
        distance_squared +=  tmp * tmp;
    }
    
    return PyFloat_FromDouble(distance_squared);
}


static int 
vector_setAttr_swizzle(PyVector *self, PyObject *attr_name, PyObject *val);
static PyObject *
vector_getAttr_swizzle(PyVector *self, PyObject *attr_name);

static PyObject *
vector_enable_swizzle(PyVector *self)
{
    self->ob_type->tp_getattro = (getattrofunc)vector_getAttr_swizzle;
    self->ob_type->tp_setattro = (setattrofunc)vector_setAttr_swizzle;
    Py_RETURN_NONE;
}

static PyObject *
vector_disable_swizzle(PyVector *self)
{
    self->ob_type->tp_getattro = PyObject_GenericGetAttr;
    self->ob_type->tp_setattro = PyObject_GenericSetAttr;
    Py_RETURN_NONE;
}


static PyObject *vector_elementwise(PyVector *self);

static PyMethodDef vector2_methods[] = {
    {"enable_swizzle", (PyCFunction)vector_enable_swizzle, METH_NOARGS,
     "enables swizzling."
    },
    {"disable_swizzle", (PyCFunction)vector_disable_swizzle, METH_NOARGS,
     "disables swizzling."
    },
    {"length", (PyCFunction)vector_length, METH_NOARGS,
     "returns the length/magnitude of the vector."
    },
    {"length_squared", (PyCFunction)vector_length_squared, METH_NOARGS,
     "returns the length/magnitude of the vector."
    },
    {"rotate", (PyCFunction)vector2_rotate, METH_VARARGS,
     "returns a new vector rotated counterclockwise by the angle given in degrees."
    },
    {"rotate_ip", (PyCFunction)vector2_rotate_ip, METH_VARARGS,
     "rotates the vector counterclockwise by the angle given in degrees."
    },
    {"normalize", (PyCFunction)vector_normalize, METH_NOARGS,
     "returns a vector that has length == 1 and the same direction as self."
    },
    {"normalize_ip", (PyCFunction)vector_normalize_ip, METH_NOARGS,
     "Normalizes the vector so that it has length == 1."
    },
    {"is_normalized", (PyCFunction)vector_is_normalized, METH_NOARGS,
     "returns True if the vector has length == 1. otherwise it returns False."
    },
    {"cross", (PyCFunction)vector2_cross, METH_O,
     "calculates the cross product."
    },
    {"dot", (PyCFunction)vector_dot, METH_O,
     "calculates the dot product."
    },
    {"angle_to", (PyCFunction)vector_angle_to, METH_O,
     "returns the angle between self and the given vector."
    },
    {"scale_to_length", (PyCFunction)vector_scale_to_length, METH_O,
     "scales the vector to the given length."
    },
    {"reflect", (PyCFunction)vector_reflect, METH_O,
     "reflects the vector on the surface characterized by the given normal."
    },
    {"reflect_ip", (PyCFunction)vector_reflect_ip, METH_O,
     "reflects the vector in-place on the surface characterized by the given normal."
    },
    {"distance_to", (PyCFunction)vector_distance_to, METH_O,
     "returns the distance to the given vector."
    },
    {"distance_squared_to", (PyCFunction)vector_distance_squared_to, METH_O,
     "returns the squared distance to the given vector."
    },
    {"elementwise", (PyCFunction)vector_elementwise, METH_NOARGS,
     "applies the following operation to each element of the vector."
    },
    
    {NULL}  /* Sentinel */
};

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
    return PyString_FromString(buffer[bufferIdx % 2]); 
}

/*
static PyObject *
vector2_repr(PyVector *self)
{
    char buffer[STRING_BUF_SIZE];

    PyOS_snprintf(buffer, STRING_BUF_SIZE, "<Vector2(%g, %g)>",
                  self->coords[0], self->coords[1]);
    return PyString_FromString(buffer); 
}
*/

static PyObject *
vector_str(PyVector *self)
{
    int i;
    int bufferIdx;
    char buffer[2][STRING_BUF_SIZE];
    
    bufferIdx = 1;
    PyOS_snprintf(buffer[0], STRING_BUF_SIZE, "[", self->dim);
    for (i = 0; i < self->dim - 1; ++i) {
        PyOS_snprintf(buffer[bufferIdx % 2], STRING_BUF_SIZE, "%s%g, ", 
                      buffer[(bufferIdx + 1) % 2], self->coords[i]);
        bufferIdx++;
    }
    PyOS_snprintf(buffer[bufferIdx % 2], STRING_BUF_SIZE, "%s%g]", 
                  buffer[(bufferIdx + 1) % 2], self->coords[i]);
    return PyString_FromString(buffer[bufferIdx % 2]); 
}

/*
static PyObject *
vector2_str(PyVector *self)
{
    char buffer[STRING_BUF_SIZE];
    PyOS_snprintf(buffer, STRING_BUF_SIZE, "[%g, %g]",
                  self->coords[0], self->coords[1]);
    return PyString_FromString(buffer); 
}
*/

static PyObject*
vector_getAttr_swizzle(PyVector *self, PyObject *attr_name)
{
    PyObject *res = PyObject_GenericGetAttr((PyObject*)self, attr_name);
    /* if normal lookup failed try to swizzle */
    if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        Py_ssize_t len = PySequence_Length(attr_name);
        const char *attr = PyString_AsString(attr_name);
        double *coords = self->coords;
        Py_ssize_t i;
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
    // first try swizzle
    const char *attr = PyString_AsString(attr_name);
    Py_ssize_t len = PySequence_Length(attr_name);
    double entry[self->dim];
    int entry_was_set[self->dim];
    int swizzle_err = SWIZZLE_ERR_NO_ERR;
    int i;

    for (i = 0; i < self->dim; ++i)
        entry_was_set[i] = 0;

    for (i = 0; i < len; ++i) {
        int idx;
        switch (attr[i]) {
        case 'x':
        case 'y':
/*        case 'z': */
            idx = attr[i] - 'x';
            break;
/*        case 'w':
            idx = 3;
            break; */
        default:
            // swizzle failed. attempt normal attribute setting
            return PyObject_GenericSetAttr((PyObject*)self, attr_name, val);
        }
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
        /* this should NOT happen */
        PyErr_SetString(PyExc_RuntimeError, "Unhandled error in swizzle code");
        return -1;
    }
} 


static PyGetSetDef vector2_getsets[] = {
    { "x", (getter)vector_getx, (setter)vector_setx, NULL, NULL },
    { "y", (getter)vector_gety, (setter)vector_sety, NULL, NULL },
    { NULL, 0, NULL, NULL, NULL }  /* Sentinel */
};


/********************************
 * PyVector2 type definition
 ********************************/
static PyObject *vector_iter(PyObject *vec);

static PyTypeObject PyVector2_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /* ob_size */
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
    (getattrofunc)PyObject_GenericGetAttr, /* tp_getattro */
    (setattrofunc)PyObject_GenericSetAttr, /* tp_setattro */
    /* Functions to access object as input/output buffer */
    0,                         /* tp_as_buffer */
    /* Flags to define presence of optional/expanded features */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | 
    Py_TPFLAGS_CHECKTYPES, /* tp_flags */
    /* Documentation string */
    DOC_PYGAMEVECTOR2,         /* tp_doc */

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


/********************************
 * PyVector2Iterator type definition
 ********************************/

typedef struct {
    PyObject_HEAD
    long it_index;
    PyVector *vec;
} vectoriter;

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
    Py_ssize_t len;
    if (it && it->vec) {
        len = it->vec->dim - it->it_index;
        if (len >= 0)
            return PyInt_FromSsize_t(len);
    }
    return PyInt_FromLong(0);
}

static PyMethodDef vectoriter_methods[] = {
    {"__length_hint__", (PyCFunction)vectoriter_len, METH_NOARGS,
    },
    {NULL, NULL} /* sentinel */
};

static PyTypeObject PyVectorIter_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /* ob_size */
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



















typedef struct {
    PyObject_HEAD
    PyVector *vec;
} vector_elementwiseproxy;

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

    if (checkPyVectorCompatible(other, dim)) {
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
vector_elementwiseproxy_add(PyObject *o1, PyObject *o2)
{
    int i, dim;
    double value;
    PyObject *other;
    PyVector *vec, *ret;
    if (vector_elementwiseproxy_Check(o1)) {
        vec = ((vector_elementwiseproxy*)o1)->vec;
        other = o2;
    }
    else {
        other = o1;
        vec = ((vector_elementwiseproxy*)o2)->vec;
    }
    if (vector_elementwiseproxy_Check(other))
        other = (PyObject*)((vector_elementwiseproxy*)other)->vec;
    dim = vec->dim;

    if (checkPyVectorCompatible(other, dim)) {
        return vector_add((PyObject*)vec, other);
    }
    else if (RealNumber_Check(other)) {
        value = PyFloat_AsDouble(other);
        ret = (PyVector*)PyVector_NEW(dim);
        for (i = 0; i < dim; i++) {
            ret->coords[i] = vec->coords[i] + value;
        }
        return (PyObject*)ret;
    }

    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}


static PyObject *
vector_elementwiseproxy_sub(PyObject *o1, PyObject *o2)
{
    int i, dim, reverse;
    double value;
    PyVector *vec, *ret;
    PyObject *other;

    if (vector_elementwiseproxy_Check(o1)) {
        vec = ((vector_elementwiseproxy*)o1)->vec;
        other = o2;
        reverse = 0;
    }
    else {
        vec = ((vector_elementwiseproxy*)o2)->vec;
        other = o1;
        reverse = 1;
    }
    if (vector_elementwiseproxy_Check(other))
        other = (PyObject*)((vector_elementwiseproxy*)other)->vec;
    dim = vec->dim;
    
    if (checkPyVectorCompatible(other, dim)) {
        if (reverse)
            return vector_sub(other, (PyObject*)vec);
        else
            return vector_sub((PyObject*)vec, other);
    }
    else if (RealNumber_Check(other)) {
        value = PyFloat_AsDouble(other);
        ret = (PyVector*)PyVector_NEW(dim);
        for (i = 0; i < dim; i++) {
            ret->coords[i] = (vec->coords[i] - value) * (reverse ? -1 : 1);
        }
        return (PyObject*)ret;
    }

    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}


static PyObject *
vector_elementwiseproxy_mul(PyObject *o1, PyObject *o2)
{
    int i;
    PyVector *vec, *ret;
    PyObject *other;

    if (vector_elementwiseproxy_Check(o1)) {
        vec = ((vector_elementwiseproxy*)o1)->vec;
        other = o2;
    }
    else {
        other = o1;
        vec = ((vector_elementwiseproxy*)o2)->vec;
    }
    if (vector_elementwiseproxy_Check(other))
        other = (PyObject*)((vector_elementwiseproxy*)other)->vec;

    /* elementwiseproxy * vector ? */
    if (checkPyVectorCompatible(other, vec->dim)) {
        ret = (PyVector*)PyVector_NEW(vec->dim);
        for (i = 0; i < vec->dim; i++) {
            ret->coords[i] = (vec->coords[i] * 
                              PySequence_GetItem_AsDouble(other, i));
        }
        return (PyObject*)ret;
    }
    /* elementwise * scalar ? */
    else if (RealNumber_Check(other)) {
        return vector_mul((PyObject*)vec, other);
    }

    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}


static PyObject *
vector_elementwiseproxy_div(PyObject *o1, PyObject *o2)
{
    int i;
    PyVector *vec, *ret;

    if (vector_elementwiseproxy_Check(o1)) {
        vec = ((vector_elementwiseproxy*)o1)->vec;
        if (vector_elementwiseproxy_Check(o2))
            o2 = (PyObject*)((vector_elementwiseproxy*)o2)->vec;
        if (checkPyVectorCompatible(o2, vec->dim)) {
            ret = (PyVector*)PyVector_NEW(vec->dim);
            for (i = 0; i < vec->dim; i++) {
                ret->coords[i] = (vec->coords[i] /
                                  PySequence_GetItem_AsDouble(o2, i));
            }
            return (PyObject*)ret;
        }
        else if (RealNumber_Check(o2)) {
            return vector_div(vec, o2);
        }
    }
    else {
        vec = ((vector_elementwiseproxy*)o2)->vec;
        if (checkPyVectorCompatible(o1, vec->dim)) {
            ret = (PyVector*)PyVector_NEW(vec->dim);
            for (i = 0; i < vec->dim; i++) {
                ret->coords[i] = (PySequence_GetItem_AsDouble(o1, i) /
                                  vec->coords[i]);
            }
            return (PyObject*)ret;
        }
        else if (RealNumber_Check(o1)) {
            double value = PyFloat_AsDouble(o1);
            ret = (PyVector*)PyVector_NEW(vec->dim);
            for (i = 0; i < vec->dim; i++) {
                ret->coords[i] = value / vec->coords[i];
            }
            return (PyObject*)ret;
        }
    }

    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
vector_elementwiseproxy_floor_div(PyObject *o1, PyObject *o2)
{
    int i;
    PyVector *vec, *ret;

    if (vector_elementwiseproxy_Check(o1)) {
        vec = ((vector_elementwiseproxy*)o1)->vec;
        if (vector_elementwiseproxy_Check(o2))
            o2 = (PyObject*)((vector_elementwiseproxy*)o2)->vec;
        if (checkPyVectorCompatible(o2, vec->dim)) {
            ret = (PyVector*)PyVector_NEW(vec->dim);
            for (i = 0; i < vec->dim; i++) {
                ret->coords[i] = floor(vec->coords[i] / 
                                       PySequence_GetItem_AsDouble(o2, i));
            }
            return (PyObject*)ret;
        }
        else if (RealNumber_Check(o2)) {
            return vector_floor_div(vec, o2);
        }
    }
    else {
        vec = ((vector_elementwiseproxy*)o2)->vec;
        if (checkPyVectorCompatible(o1, vec->dim)) {
            ret = (PyVector*)PyVector_NEW(vec->dim);
            for (i = 0; i < vec->dim; i++) {
                ret->coords[i] = floor(PySequence_GetItem_AsDouble(o1, i) /
                                       vec->coords[i]);
            }
            return (PyObject*)ret;
        }
        else if (RealNumber_Check(o1)) {
            double value = PyFloat_AsDouble(o1);
            ret = (PyVector*)PyVector_NEW(vec->dim);
            for (i = 0; i < vec->dim; i++) {
                ret->coords[i] = floor(value / vec->coords[i]);
            }
            return (PyObject*)ret;
        }
    }

    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
vector_elementwiseproxy_pow(PyObject *baseObj, PyObject *expoObj, PyObject *mod)
{
    int i, dim;
    double tmp;
    double bases[VECTOR_MAX_SIZE];
    double expos[VECTOR_MAX_SIZE];
    PyVector *ret, *vec;
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
        else if (checkPyVectorCompatible(expoObj, dim)) {
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
        if (checkPyVectorCompatible(baseObj, dim)) {
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
vector_elementwiseproxy_mod(PyObject *o1, PyObject *o2)
{
    /* TODO: There wasn't put much thought behind this implementation and
             it is late so this should probably be thoughly reviewed. */
    int i;
    PyObject *entry, *divisor, *result;
    PyVector *res, *vec;
    if (vector_elementwiseproxy_Check(o1)) {
        vec = ((vector_elementwiseproxy*)o1)->vec;
        if (vector_elementwiseproxy_Check(o2))
            o2 = (PyObject*)((vector_elementwiseproxy*)o2)->vec;
        if (checkPyVectorCompatible(o2, vec->dim)) { 
            res = (PyVector*)PyVector_NEW(vec->dim);
            for (i = 0; i < vec->dim; i++) {
                entry = PyFloat_FromDouble(vec->coords[i]);
                divisor = PyFloat_FromDouble(PySequence_GetItem_AsDouble(o2, i));
                result = PyNumber_Remainder(entry, divisor);
                res->coords[i] = PyFloat_AsDouble(result);
                Py_DECREF(entry);
                Py_DECREF(divisor);
                Py_DECREF(result);
            }
            return (PyObject*)res;
        }
        else if (RealNumber_Check(o2)) {
            res = (PyVector*)PyVector_NEW(vec->dim);
            for (i = 0; i < vec->dim; i++) {
                entry = PyFloat_FromDouble(vec->coords[i]);
                result = PyNumber_Remainder(entry, o2);
                res->coords[i] = PyFloat_AsDouble(result);
                Py_DECREF(entry);
                Py_DECREF(result);
            }
            return (PyObject*)res;
        }
    }
    else {
        vec = ((vector_elementwiseproxy*)o2)->vec;
        if (checkPyVectorCompatible(o1, vec->dim)) { 
            res = (PyVector*)PyVector_NEW(vec->dim);
            for (i = 0; i < vec->dim; i++) {
                divisor = PyFloat_FromDouble(vec->coords[i]);
                entry = PyFloat_FromDouble(PySequence_GetItem_AsDouble(o1, i));
                result = PyNumber_Remainder(entry, divisor);
                res->coords[i] = PyFloat_AsDouble(result);
                Py_DECREF(entry);
                Py_DECREF(divisor);
                Py_DECREF(result);
            }
            return (PyObject*)res;
        }
        else if (RealNumber_Check(o1)) {
            res = (PyVector*)PyVector_NEW(vec->dim);
            for (i = 0; i < vec->dim; i++) {
                divisor = PyFloat_FromDouble(vec->coords[i]);
                result = PyNumber_Remainder(o1, divisor);
                res->coords[i] = PyFloat_AsDouble(result);
                Py_DECREF(divisor);
                Py_DECREF(result);
            }
            return (PyObject*)res;
        }
    }

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
    int i;
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

PyNumberMethods vector_elementwiseproxy_as_number = {
    (binaryfunc)vector_elementwiseproxy_add,      /* nb_add;       __add__ */
    (binaryfunc)vector_elementwiseproxy_sub,      /* nb_subtract;  __sub__ */
    (binaryfunc)vector_elementwiseproxy_mul,      /* nb_multiply;  __mul__ */
    (binaryfunc)vector_elementwiseproxy_div,      /* nb_divide;    __div__ */
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
    (coercion)0,                    /* nb_coerce;    __coerce__ */
    (unaryfunc)0,                   /* nb_int;       __int__ */
    (unaryfunc)0,                   /* nb_long;      __long__ */
    (unaryfunc)0,                   /* nb_float;     __float__ */
    (unaryfunc)0,                   /* nb_oct;       __oct__ */
    (unaryfunc)0,                   /* nb_hex;       __hex__ */

    /* Added in release 2.0 */
    (binaryfunc)0,                  /* nb_inplace_add;       __iadd__ */
    (binaryfunc)0,                  /* nb_inplace_subtract;  __isub__ */
    (binaryfunc)0,                  /* nb_inplace_multiply;  __imul__ */
    (binaryfunc)0,                  /* nb_inplace_divide;    __idiv__ */
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

    /* Added in release 2.5 */
    (unaryfunc)0,                   /* nb_index;  __index__ */
};




static PyTypeObject PyVectorElementwiseProxy_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /* ob_size */
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES, /* tp_flags */
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
        Py_DECREF (apiobj);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    MODINIT_RETURN (module);
}
