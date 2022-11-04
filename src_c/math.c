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

/* Adjust gcc 4.4 optimization for floating point on x86-32 PCs running Linux.
 * This addresses bug 52:
 * https://github.com/pygame/pygame/issues/52
 * With this option, floats have consistent precision regardless of optimize
 * level.
 */
#if defined(__GNUC__) && defined(__linux__) && defined(__i386__) && \
    __SIZEOF_POINTER__ == 4 && __GNUC__ == 4 && __GNUC_MINOR__ >= 4
#pragma GCC optimize("float-store")
#endif

#define PYGAMEAPI_MATH_INTERNAL
#define NO_PYGAME_C_API
#include "doc/math_doc.h"

#include "pygame.h"

#include "structmember.h"

#include "pgcompat.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

/* on some windows platforms math.h doesn't define M_PI */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TWO_PI (2. * M_PI)

#ifndef M_PI_2
#define M_PI_2 (M_PI / 2.0)
#endif /* M_PI_2 */

#define VECTOR_EPSILON (1e-6)
#define VECTOR_MAX_SIZE (4)
#define STRING_BUF_SIZE_REPR (112)
#define STRING_BUF_SIZE_STR (103)
#define SWIZZLE_ERR_NO_ERR 0
#define SWIZZLE_ERR_DOUBLE_IDX 1
#define SWIZZLE_ERR_EXTRACTION_ERR 2

#define OP_ADD 1
/* #define OP_IADD         2 */
#define OP_SUB 3
/* #define OP_ISUB         4 */
#define OP_MUL 5
/* #define OP_IMUL         6 */
#define OP_DIV 7
/* #define OP_IDIV         8 */
#define OP_FLOOR_DIV 9
/* #define OP_IFLOOR_DIV  10 */
#define OP_MOD 11
#define OP_INPLACE 16
#define OP_ARG_REVERSE 32
#define OP_ARG_UNKNOWN 64
#define OP_ARG_VECTOR 128
#define OP_ARG_NUMBER 256

static PyTypeObject pgVector2_Type;
static PyTypeObject pgVector3_Type;
static PyTypeObject pgVectorElementwiseProxy_Type;
static PyTypeObject pgVectorIter_Type;

#define pgVector2_Check(x) (PyType_IsSubtype(Py_TYPE(x), &pgVector2_Type))
#define pgVector3_Check(x) (PyType_IsSubtype(Py_TYPE(x), &pgVector3_Type))
#define pgVector_Check(x) (pgVector2_Check(x) || pgVector3_Check(x))
#define vector_elementwiseproxy_Check(x) \
    (Py_TYPE(x) == &pgVectorElementwiseProxy_Type)
#define _vector_subtype_new(x) \
    ((pgVector *)(Py_TYPE(x)->tp_new(Py_TYPE(x), NULL, NULL)))

#define DEG2RAD(angle) ((angle)*M_PI / 180.)
#define RAD2DEG(angle) ((angle)*180. / M_PI)

typedef struct {
    PyObject_HEAD double *coords; /* Coordinates */
    Py_ssize_t dim;               /* Dimension of the vector */
    double epsilon;               /* Small value for comparisons */
} pgVector;

typedef struct {
    PyObject_HEAD long it_index;
    pgVector *vec;
} vectoriter;

typedef struct {
    PyObject_HEAD pgVector *vec;
} vector_elementwiseproxy;

/* further forward declarations */
/* math functions */
static PyObject *
math_clamp(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PG_DECLARE_FASTCALL_FUNC(math_clamp, PyObject);

/* generic helper functions */
static int
RealNumber_Check(PyObject *obj);
static double
PySequence_GetItem_AsDouble(PyObject *seq, Py_ssize_t index);
static int
PySequence_AsVectorCoords(PyObject *seq, double *const coords,
                          const Py_ssize_t size);
static int
pgVectorCompatible_Check(PyObject *obj, Py_ssize_t dim);
static double
_scalar_product(const double *coords1, const double *coords2, Py_ssize_t size);
static int
get_double_from_unicode_slice(PyObject *unicode_obj, Py_ssize_t idx1,
                              Py_ssize_t idx2, double *val);
static Py_ssize_t
_vector_find_string_helper(PyObject *str_obj, const char *substr,
                           Py_ssize_t start, Py_ssize_t end);
static Py_ssize_t
_vector_coords_from_string(PyObject *str, char **delimiter, double *coords,
                           Py_ssize_t dim);
static void
_vector_move_towards_helper(Py_ssize_t dim, double *origin_coords,
                            double *target_coords, double max_distance);

/* generic vector functions */
static PyObject *
pgVector_NEW(Py_ssize_t dim);
static void
vector_dealloc(pgVector *self);
static PyObject *
vector_generic_math(PyObject *o1, PyObject *o2, int op);
static PyObject *
vector_add(PyObject *o1, PyObject *o2);
static PyObject *
vector_sub(PyObject *o1, PyObject *o2);
static PyObject *
vector_mul(PyObject *o1, PyObject *o2);
static PyObject *
vector_div(pgVector *self, PyObject *other);
static PyObject *
vector_floor_div(pgVector *self, PyObject *other);
static PyObject *
vector_inplace_add(pgVector *self, PyObject *other);
static PyObject *
vector_inplace_sub(pgVector *self, PyObject *other);
static PyObject *
vector_inplace_mul(pgVector *self, PyObject *other);
static PyObject *
vector_inplace_div(pgVector *self, PyObject *other);
static PyObject *
vector_inplace_floor_div(pgVector *self, PyObject *other);
static PyObject *
vector_neg(pgVector *self);
static PyObject *
vector_pos(pgVector *self);
static int
vector_nonzero(pgVector *self);
static Py_ssize_t
vector_len(pgVector *self);
static PyObject *
vector_GetItem(pgVector *self, Py_ssize_t index);
static int
vector_SetItem(pgVector *self, Py_ssize_t index, PyObject *value);
static int
vector_contains(pgVector *self, PyObject *arg);
static PyObject *
vector_GetSlice(pgVector *self, Py_ssize_t ilow, Py_ssize_t ihigh);
static int
vector_SetSlice(pgVector *self, Py_ssize_t ilow, Py_ssize_t ihigh,
                PyObject *v);
static PyObject *
vector_getx(pgVector *self, void *closure);
static PyObject *
vector_gety(pgVector *self, void *closure);
static PyObject *
vector_getz(pgVector *self, void *closure);
#ifdef PYGAME_MATH_VECTOR_HAVE_W
static PyObject *
vector_getw(pgVector *self, void *closure);
#endif
static int
vector_setx(pgVector *self, PyObject *value, void *closure);
static int
vector_sety(pgVector *self, PyObject *value, void *closure);
static int
vector_setz(pgVector *self, PyObject *value, void *closure);
#ifdef PYGAME_MATH_VECTOR_HAVE_W
static int
vector_setw(pgVector *self, PyObject *value, void *closure);
#endif
static PyObject *
vector_richcompare(PyObject *o1, PyObject *o2, int op);
static PyObject *
vector_length(pgVector *self, PyObject *args);
static PyObject *
vector_length_squared(pgVector *self, PyObject *args);
static PyObject *
vector_normalize(pgVector *self, PyObject *args);
static PyObject *
vector_normalize_ip(pgVector *self, PyObject *args);
static PyObject *
vector_dot(pgVector *self, PyObject *other);
static PyObject *
vector_scale_to_length(pgVector *self, PyObject *length);
static PyObject *
vector_move_towards(pgVector *self, PyObject *args);
static PyObject *
vector_move_towards_ip(pgVector *self, PyObject *args);
static PyObject *
vector_slerp(pgVector *self, PyObject *args);
static PyObject *
vector_lerp(pgVector *self, PyObject *args);
static int
_vector_reflect_helper(double *dst_coords, const double *src_coords,
                       PyObject *normal, Py_ssize_t dim, double epsilon);
static PyObject *
vector_reflect(pgVector *self, PyObject *normal);
static PyObject *
vector_reflect_ip(pgVector *self, PyObject *normal);
static double
_vector_distance_helper(pgVector *self, PyObject *other);
static PyObject *
vector_distance_to(pgVector *self, PyObject *other);
static PyObject *
vector_distance_squared_to(pgVector *self, PyObject *other);
static PyObject *
vector_getAttr_swizzle(pgVector *self, PyObject *attr_name);
static int
vector_setAttr_swizzle(pgVector *self, PyObject *attr_name, PyObject *val);
static PyObject *
vector_elementwise(pgVector *self, PyObject *args);
static int
_vector_check_snprintf_success(int return_code, int max_size);
static PyObject *
vector_repr(pgVector *self);
static PyObject *
vector_str(pgVector *self);
static PyObject *
vector_project_onto(pgVector *self, PyObject *other);
static PyObject *
vector_copy(pgVector *self, PyObject *_null);
static PyObject *
vector_clamp_magnitude(pgVector *self, PyObject *const *args,
                       Py_ssize_t nargs);
PG_DECLARE_FASTCALL_FUNC(vector_clamp_magnitude, pgVector);
static PyObject *
vector_clamp_magnitude_ip(pgVector *self, PyObject *const *args,
                          Py_ssize_t nargs);
PG_DECLARE_FASTCALL_FUNC(vector_clamp_magnitude_ip, pgVector);

/*
static Py_ssize_t vector_readbuffer(pgVector *self, Py_ssize_t segment, void
**ptrptr); static Py_ssize_t vector_writebuffer(pgVector *self, Py_ssize_t
segment, void **ptrptr); static Py_ssize_t vector_segcount(pgVector *self,
Py_ssize_t *lenp);
*/

/* vector2 specific functions */
static PyObject *
vector2_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int
vector2_init(pgVector *self, PyObject *args, PyObject *kwds);
static int
_vector2_rotate_helper(double *dst_coords, const double *src_coords,
                       double angle, double epsilon);
static PyObject *
vector2_rotate(pgVector *self, PyObject *args);
static PyObject *
vector2_rotate_ip(pgVector *self, PyObject *args);
static PyObject *
vector2_cross(pgVector *self, PyObject *other);
static PyObject *
vector2_angle_to(pgVector *self, PyObject *other);
static PyObject *
vector2_as_polar(pgVector *self, PyObject *args);
static PyObject *
vector2_from_polar(pgVector *self, PyObject *args);

/* vector3 specific functions */
static PyObject *
vector3_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int
vector3_init(pgVector *self, PyObject *args, PyObject *kwds);
static int
_vector3_rotate_helper(double *dst_coords, const double *src_coords,
                       const double *axis_coords, double angle,
                       double epsilon);
static PyObject *
vector3_rotate(pgVector *self, PyObject *args);
static PyObject *
vector3_rotate_ip(pgVector *self, PyObject *args);
static PyObject *
vector3_cross(pgVector *self, PyObject *other);
static PyObject *
vector3_angle_to(pgVector *self, PyObject *other);
static PyObject *
vector3_as_spherical(pgVector *self, PyObject *args);
static PyObject *
vector3_from_spherical(pgVector *self, PyObject *args);

/* vector iterator functions */
static void
vectoriter_dealloc(vectoriter *it);
static PyObject *
vectoriter_next(vectoriter *it);
static PyObject *
vectoriter_len(vectoriter *it);
static PyObject *
vector_iter(PyObject *vec);

/* elementwiseproxy */
static void
vector_elementwiseproxy_dealloc(vector_elementwiseproxy *it);
static PyObject *
vector_elementwiseproxy_richcompare(PyObject *o1, PyObject *o2, int op);
static PyObject *
vector_elementwiseproxy_generic_math(PyObject *o1, PyObject *o2, int op);
static PyObject *
vector_elementwiseproxy_add(PyObject *o1, PyObject *o2);
static PyObject *
vector_elementwiseproxy_sub(PyObject *o1, PyObject *o2);
static PyObject *
vector_elementwiseproxy_mul(PyObject *o1, PyObject *o2);
static PyObject *
vector_elementwiseproxy_div(PyObject *o1, PyObject *o2);
static PyObject *
vector_elementwiseproxy_floor_div(PyObject *o1, PyObject *o2);
static PyObject *
vector_elementwiseproxy_pow(PyObject *baseObj, PyObject *expoObj,
                            PyObject *mod);
static PyObject *
vector_elementwiseproxy_mod(PyObject *o1, PyObject *o2);
static PyObject *
vector_elementwiseproxy_abs(vector_elementwiseproxy *self);
static PyObject *
vector_elementwiseproxy_neg(vector_elementwiseproxy *self);
static PyObject *
vector_elementwiseproxy_pos(vector_elementwiseproxy *self);
static int
vector_elementwiseproxy_nonzero(vector_elementwiseproxy *self);
static PyObject *
vector_elementwise(pgVector *vec, PyObject *args);

/********************************
 * Global helper functions
 ********************************/

static int
RealNumber_Check(PyObject *obj)
{
    if (obj && PyNumber_Check(obj) && !PyComplex_Check(obj))
        return 1;
    return 0;
}

static double
PySequence_GetItem_AsDouble(PyObject *seq, Py_ssize_t index)
{
    PyObject *item;
    double value;

    item = PySequence_GetItem(seq, index);
    if (item == NULL) {
        PyErr_SetString(PyExc_TypeError, "a sequence is expected");
        return -1;
    }
    value = PyFloat_AsDouble(item);
    Py_DECREF(item);
    if (PyErr_Occurred())
        return -1;
    return value;
}

static int
PySequence_AsVectorCoords(PyObject *seq, double *const coords,
                          const Py_ssize_t size)
{
    Py_ssize_t i;

    if (pgVector_Check(seq)) {
        memcpy(coords, ((pgVector *)seq)->coords, sizeof(double) * size);
        return 1;
    }
    if (!PySequence_Check(seq) || PySequence_Length(seq) != size) {
        PyErr_SetString(PyExc_ValueError, "Sequence has the wrong length.");
        return 0;
    }

    for (i = 0; i < size; ++i) {
        coords[i] = PySequence_GetItem_AsDouble(seq, i);
        if (PyErr_Occurred()) {
            return 0;
        }
    }
    return 1;
}

static int
pgVectorCompatible_Check(PyObject *obj, Py_ssize_t dim)
{
    Py_ssize_t i;
    PyObject *tmp;

    switch (dim) {
        case 2:
            if (pgVector2_Check(obj)) {
                return 1;
            }
            break;
        case 3:
            if (pgVector3_Check(obj)) {
                return 1;
            }
            break;
            /*
                case 4:
                    if (pgVector4_Check(obj)) {
                        return 1;
                    }
                    break;
            */
        default:
            PyErr_SetString(
                PyExc_SystemError,
                "Wrong internal call to pgVectorCompatible_Check.");
            return 0;
    }

    if (!PySequence_Check(obj) || (PySequence_Length(obj) != dim)) {
        return 0;
    }

    for (i = 0; i < dim; ++i) {
        tmp = PySequence_GetItem(obj, i);
        if (tmp == NULL || !RealNumber_Check(tmp)) {
            Py_XDECREF(tmp);
            return 0;
        }
        Py_DECREF(tmp);
    }
    return 1;
}

static double
_scalar_product(const double *coords1, const double *coords2, Py_ssize_t size)
{
    Py_ssize_t i;
    double product = 0;
    for (i = 0; i < size; ++i)
        product += coords1[i] * coords2[i];
    return product;
}

static int
get_double_from_unicode_slice(PyObject *unicode_obj, Py_ssize_t idx1,
                              Py_ssize_t idx2, double *val)
{
    PyObject *float_obj;
    PyObject *slice = PySequence_GetSlice(unicode_obj, idx1, idx2);
    if (slice == NULL) {
        PyErr_SetString(PyExc_SystemError,
                        "internal error while converting str slice to float");
        return -1;
    }
    float_obj = PyFloat_FromString(slice);
    Py_DECREF(slice);
    if (float_obj == NULL)
        return 0;
    *val = PyFloat_AsDouble(float_obj);
    Py_DECREF(float_obj);
    return 1;
}

static Py_ssize_t
_vector_find_string_helper(PyObject *str_obj, const char *substr,
                           Py_ssize_t start, Py_ssize_t end)
{
    /* This is implemented using Python's Unicode C-API rather than
     * plain C to handle both char* and wchar_t* especially because
     * wchar.h was only added to the standard in 95 and we want to be
     * ISO C 90 compatible.
     */
    PyObject *substr_obj;
    Py_ssize_t pos;
    substr_obj = PyUnicode_FromString(substr);
    if (substr_obj == NULL) {
        return -2;
    }
    pos = PyUnicode_Find(str_obj, substr_obj, start, end, 1);
    Py_DECREF(substr_obj);
    return pos;
}

/* Extracts coords from a str/unicode Object <str>.
 * <delimiter> should be an array of <dim>+1 C-Strings of the strings before,
 *   between and after the coords.
 * The extracted numbers will be stored in <coords>.
 * return
 *    0 on success.
 *   -1 if conversion was unsuccessful
 *   -2 if an internal error occurred and an exception was set
 */
static Py_ssize_t
_vector_coords_from_string(PyObject *str, char **delimiter, double *coords,
                           Py_ssize_t dim)
{
    int error_code;
    Py_ssize_t i, start_pos, end_pos, length;
    PyObject *vector_string;
    vector_string = PyUnicode_FromObject(str);
    if (vector_string == NULL) {
        return -2;
    }
    length = PySequence_Length(vector_string);
    /* find the starting point of the first coordinate in the string */
    start_pos =
        _vector_find_string_helper(vector_string, delimiter[0], 0, length);
    if (start_pos < 0) {
        return start_pos;
    }
    start_pos += strlen(delimiter[0]);
    for (i = 0; i < dim; i++) {
        /* find the end point of the current coordinate in the string */
        end_pos = _vector_find_string_helper(vector_string, delimiter[i + 1],
                                             start_pos, length);
        if (end_pos < 0)
            return end_pos;
        /* try to convert the current coordinate */
        error_code = get_double_from_unicode_slice(vector_string, start_pos,
                                                   end_pos, &coords[i]);
        if (error_code < 0) {
            return -2;
        }
        else if (error_code == 0) {
            return -1;
        }
        /* move starting point to the next coordinate */
        start_pos = end_pos + strlen(delimiter[i + 1]);
    }
    return 0;
}

static PyMemberDef vector_members[] = {
    {"epsilon", T_DOUBLE, offsetof(pgVector, epsilon), 0,
     "small value used in comparisons"},
    {NULL} /* Sentinel */
};

static PyObject *
pgVector_NEW(Py_ssize_t dim)
{
    switch (dim) {
        case 2:
            return vector2_new(&pgVector2_Type, NULL, NULL);
        case 3:
            return vector3_new(&pgVector3_Type, NULL, NULL);
            /*
                case 4:
                    return vector4_new(&pgVector4_Type, NULL, NULL);
            */
        default:
            PyErr_SetString(PyExc_SystemError,
                            "Wrong internal call to pgVector_NEW.\n");
            return NULL;
    }
}

static void
vector_dealloc(pgVector *self)
{
    PyMem_Free(self->coords);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/**********************************************
 * Generic vector PyNumber emulation routines
 **********************************************/

static PyObject *
vector_generic_math(PyObject *o1, PyObject *o2, int op)
{
    Py_ssize_t i, dim;
    double *vec_coords;
    double other_coords[VECTOR_MAX_SIZE] = {0};
    double tmp;
    PyObject *other;
    pgVector *vec, *ret = NULL;
    if (pgVector_Check(o1)) {
        vec = (pgVector *)o1;
        other = o2;
    }
    else {
        vec = (pgVector *)o2;
        other = o1;
        op |= OP_ARG_REVERSE;
    }

    dim = vec->dim;
    vec_coords = vec->coords;

    if (other == NULL) {
        PyErr_BadInternalCall();
        return NULL;
    }

    if (pgVectorCompatible_Check(other, dim)) {
        op |= OP_ARG_VECTOR;
        if (!PySequence_AsVectorCoords(other, other_coords, dim))
            return NULL;
    }
    else if (RealNumber_Check(other))
        op |= OP_ARG_NUMBER;
    else
        op |= OP_ARG_UNKNOWN;

    if (op & OP_INPLACE) {
        ret = vec;
        Py_INCREF(ret);
    }
    else if (op != (OP_MUL | OP_ARG_VECTOR) &&
             op != (OP_MUL | OP_ARG_VECTOR | OP_ARG_REVERSE)) {
        ret = _vector_subtype_new(vec);
        if (ret == NULL)
            return NULL;
    }

    switch (op) {
        case OP_ADD | OP_ARG_VECTOR:
        case OP_ADD | OP_ARG_VECTOR | OP_ARG_REVERSE:
        case OP_ADD | OP_ARG_VECTOR | OP_INPLACE:
            for (i = 0; i < dim; i++)
                ret->coords[i] = vec_coords[i] + other_coords[i];
            break;
        case OP_SUB | OP_ARG_VECTOR:
        case OP_SUB | OP_ARG_VECTOR | OP_INPLACE:
            for (i = 0; i < dim; i++)
                ret->coords[i] = vec_coords[i] - other_coords[i];
            break;
        case OP_SUB | OP_ARG_VECTOR | OP_ARG_REVERSE:
            for (i = 0; i < dim; i++)
                ret->coords[i] = other_coords[i] - vec_coords[i];
            break;
        case OP_MUL | OP_ARG_VECTOR:
        case OP_MUL | OP_ARG_VECTOR | OP_ARG_REVERSE:
            tmp = 0.;
            for (i = 0; i < dim; i++)
                tmp += vec_coords[i] * other_coords[i];
            ret = (pgVector *)PyFloat_FromDouble(tmp);
            break;
        case OP_MUL | OP_ARG_NUMBER:
        case OP_MUL | OP_ARG_NUMBER | OP_ARG_REVERSE:
        case OP_MUL | OP_ARG_NUMBER | OP_INPLACE:
            tmp = PyFloat_AsDouble(other);
            for (i = 0; i < dim; i++)
                ret->coords[i] = vec_coords[i] * tmp;
            break;
        case OP_DIV | OP_ARG_NUMBER:
        case OP_DIV | OP_ARG_NUMBER | OP_INPLACE:
            tmp = PyFloat_AsDouble(other);
            if (tmp == 0.) {
                PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
                Py_DECREF(ret);
                return NULL;
            }
            tmp = 1. / tmp;
            for (i = 0; i < dim; i++)
                ret->coords[i] = vec_coords[i] * tmp;
            break;
        case OP_FLOOR_DIV | OP_ARG_NUMBER:
        case OP_FLOOR_DIV | OP_ARG_NUMBER | OP_INPLACE:
            tmp = PyFloat_AsDouble(other);
            if (tmp == 0.) {
                PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
                Py_DECREF(ret);
                return NULL;
            }
            tmp = 1. / tmp;
            for (i = 0; i < dim; i++)
                ret->coords[i] = floor(vec_coords[i] * tmp);
            break;
        default:
            Py_XDECREF(ret);
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
    }
    return (PyObject *)ret;
}

static PyObject *
vector_add(PyObject *o1, PyObject *o2)
{
    return vector_generic_math(o1, o2, OP_ADD);
}
static PyObject *
vector_inplace_add(pgVector *o1, PyObject *o2)
{
    return vector_generic_math((PyObject *)o1, o2, OP_ADD | OP_INPLACE);
}
static PyObject *
vector_sub(PyObject *o1, PyObject *o2)
{
    return vector_generic_math(o1, o2, OP_SUB);
}
static PyObject *
vector_inplace_sub(pgVector *o1, PyObject *o2)
{
    return vector_generic_math((PyObject *)o1, o2, OP_SUB | OP_INPLACE);
}
static PyObject *
vector_mul(PyObject *o1, PyObject *o2)
{
    return vector_generic_math(o1, o2, OP_MUL);
}
static PyObject *
vector_inplace_mul(pgVector *o1, PyObject *o2)
{
    return vector_generic_math((PyObject *)o1, o2, OP_MUL | OP_INPLACE);
}
static PyObject *
vector_div(pgVector *o1, PyObject *o2)
{
    return vector_generic_math((PyObject *)o1, o2, OP_DIV);
}
static PyObject *
vector_inplace_div(pgVector *o1, PyObject *o2)
{
    return vector_generic_math((PyObject *)o1, o2, OP_DIV | OP_INPLACE);
}
static PyObject *
vector_floor_div(pgVector *o1, PyObject *o2)
{
    return vector_generic_math((PyObject *)o1, o2, OP_FLOOR_DIV);
}
static PyObject *
vector_inplace_floor_div(pgVector *o1, PyObject *o2)
{
    return vector_generic_math((PyObject *)o1, o2, OP_FLOOR_DIV | OP_INPLACE);
}

static PyObject *
vector_neg(pgVector *self)
{
    pgVector *ret = _vector_subtype_new(self);

    if (ret) {
        Py_ssize_t i;

        for (i = 0; i < self->dim; i++) {
            ret->coords[i] = -self->coords[i];
        }
    }
    return (PyObject *)ret;
}

static PyObject *
vector_pos(pgVector *self)
{
    pgVector *ret = _vector_subtype_new(self);

    if (ret) {
        memcpy(ret->coords, self->coords, sizeof(ret->coords[0]) * ret->dim);
    }
    return (PyObject *)ret;
}

static int
vector_nonzero(pgVector *self)
{
    Py_ssize_t i;
    for (i = 0; i < self->dim; i++) {
        if (self->coords[i] != 0) {
            return 1;
        }
    }
    return 0;
}

static PyObject *
vector_copy(pgVector *self, PyObject *_null)
{
    Py_ssize_t i;
    pgVector *ret = _vector_subtype_new(self);

    if (!ret) {
        return NULL;
    }

    for (i = 0; i < self->dim; i++) {
        ret->coords[i] = self->coords[i];
    }
    return (PyObject *)ret;
}

static PyObject *
vector_clamp_magnitude(pgVector *self, PyObject *const *args, Py_ssize_t nargs)
{
    Py_ssize_t i;
    pgVector *ret;

    ret = _vector_subtype_new(self);
    if (ret == NULL)
        return NULL;

    for (i = 0; i < self->dim; ++i)
        ret->coords[i] = self->coords[i];

    PyObject *ret_val = vector_clamp_magnitude_ip(ret, args, nargs);
    if (!ret_val) {
        return NULL;
    }
    Py_DECREF(ret_val);

    return (PyObject *)ret;
}

PG_WRAP_FASTCALL_FUNC(vector_clamp_magnitude, pgVector);

static PyObject *
vector_clamp_magnitude_ip(pgVector *self, PyObject *const *args,
                          Py_ssize_t nargs)
{
    Py_ssize_t i;
    double max_length, old_length_sq, fraction = 1, min_length = 0;
    switch (nargs) {
        case 2:
            min_length = PyFloat_AsDouble(args[0]);
            if (min_length == -1.0 && PyErr_Occurred()) {
                return NULL;
            }
            /* Fall-through */
        case 1:
            max_length = PyFloat_AsDouble(args[nargs - 1]);
            if (max_length == -1.0 && PyErr_Occurred()) {
                return NULL;
            }
            break;
        default:
            return RAISE(PyExc_TypeError,
                         "Vector clamp function must take one or two floats");
    }

    if (min_length > max_length) {
        return RAISE(PyExc_ValueError,
                     "Argument min_length cannot exceed max_length");
    }

    if (max_length < 0 || min_length < 0) {
        return RAISE(PyExc_ValueError,
                     "Arguments to Vector clamp must be non-negative");
    }

    /* Get magnitude of Vector */
    old_length_sq = _scalar_product(self->coords, self->coords, self->dim);
    if (old_length_sq == 0) {
        return RAISE(PyExc_ValueError,
                     "Cannot clamp a vector with zero length");
    }

    /* Notes for other contributors reading this code:
     * The numerator for the fraction is different.
     */
    if (old_length_sq > max_length * max_length) {
        /* Scale to length */
        fraction = max_length / sqrt(old_length_sq);
    }

    if (old_length_sq < min_length * min_length) {
        /* Scale to length */
        fraction = min_length / sqrt(old_length_sq);
    }

    for (i = 0; i < self->dim; ++i)
        self->coords[i] *= fraction;

    Py_RETURN_NONE;
}

PG_WRAP_FASTCALL_FUNC(vector_clamp_magnitude_ip, pgVector);

static PyNumberMethods vector_as_number = {
    .nb_add = (binaryfunc)vector_add,
    .nb_subtract = (binaryfunc)vector_sub,
    .nb_multiply = (binaryfunc)vector_mul,
    .nb_negative = (unaryfunc)vector_neg,
    .nb_positive = (unaryfunc)vector_pos,
    .nb_bool = (inquiry)vector_nonzero,
    .nb_inplace_add = (binaryfunc)vector_inplace_add,
    .nb_inplace_subtract = (binaryfunc)vector_inplace_sub,
    .nb_inplace_multiply = (binaryfunc)vector_inplace_mul,
    .nb_floor_divide = (binaryfunc)vector_floor_div,
    .nb_true_divide = (binaryfunc)vector_div,
    .nb_inplace_floor_divide = (binaryfunc)vector_inplace_floor_div,
    .nb_inplace_true_divide = (binaryfunc)vector_inplace_div,
};

/*************************************************
 * Generic vector PySequence emulation routines
 *************************************************/

static Py_ssize_t
vector_len(pgVector *self)
{
    return (Py_ssize_t)self->dim;
}

static PyObject *
vector_GetItem(pgVector *self, Py_ssize_t index)
{
    if (index < 0 || index >= self->dim) {
        PyErr_SetString(PyExc_IndexError, "subscript out of range.");
        return NULL;
    }
    return PyFloat_FromDouble(self->coords[index]);
}

static int
vector_SetItem(pgVector *self, Py_ssize_t index, PyObject *value)
{
    if (index < 0 || index >= self->dim) {
        PyErr_SetString(PyExc_IndexError, "subscript out of range.");
        return -1;
    }
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "item deletion is not supported");
        return -1;
    }

    self->coords[index] = PyFloat_AsDouble(value);
    if (PyErr_Occurred())
        return -1;
    return 0;
}

static int
vector_contains(pgVector *self, PyObject *arg)
{
    double f = PyFloat_AsDouble(arg);
    if (f == -1 && PyErr_Occurred()) {
        return -1;
    }

    int i;
    for (i = 0; i < self->dim; i++) {
        if (self->coords[i] == f) {
            return 1;
        }
    }

    return 0;
}

static PyObject *
vector_GetSlice(pgVector *self, Py_ssize_t ilow, Py_ssize_t ihigh)
{
    /* some code was taken from the CPython source listobject.c */
    PyListObject *slice;
    Py_ssize_t i, len;

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
    slice = (PyListObject *)PyList_New(len);
    if (slice == NULL)
        return NULL;

    for (i = 0; i < len; i++) {
        PyList_SET_ITEM(slice, i, PyFloat_FromDouble(self->coords[ilow + i]));
    }
    return (PyObject *)slice;
}

static int
vector_SetSlice(pgVector *self, Py_ssize_t ilow, Py_ssize_t ihigh, PyObject *v)
{
    Py_ssize_t i, len;
    double new_coords[VECTOR_MAX_SIZE];

    if (v == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "Vector object doesn't support item deletion");
        return -1;
    }

    if (ilow < 0)
        ilow = 0;
    else if (ilow > self->dim)
        ilow = self->dim;
    if (ihigh < ilow)
        ihigh = ilow;
    else if (ihigh > self->dim)
        ihigh = self->dim;

    len = ihigh - ilow;
    if (!PySequence_AsVectorCoords(v, new_coords, len)) {
        return -1;
    }

    for (i = 0; i < len; ++i) {
        self->coords[i + ilow] = new_coords[i];
    }
    return 0;
}

/* sq_slice and sq_ass_slice are no longer used in this struct */
static PySequenceMethods vector_as_sequence = {
    .sq_length = (lenfunc)vector_len,
    .sq_item = (ssizeargfunc)vector_GetItem,
    .sq_ass_item = (ssizeobjargproc)vector_SetItem,
    .sq_contains = (objobjproc)vector_contains,
};

/***************************************************************************
 * Generic vector PyMapping emulation routines to support extended slicing
 ***************************************************************************/

/* Great parts of this function are adapted from python's list_subscript */
static PyObject *
vector_subscript(pgVector *self, PyObject *key)
{
    Py_ssize_t i;
    if (PyIndex_Check(key)) {
        i = PyNumber_AsSsize_t(key, PyExc_IndexError);
        if (i == -1 && PyErr_Occurred())
            return NULL;
        if (i < 0)
            i += self->dim;
        return vector_GetItem(self, i);
    }
    else if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step, slicelength, cur;
        PyObject *result;
        PyObject *it;

        if (PySlice_GetIndicesEx((PyObject *)key, self->dim, &start, &stop,
                                 &step, &slicelength) < 0) {
            return NULL;
        }

        if (slicelength <= 0) {
            return PyList_New(0);
        }
        else if (step == 1) {
            return vector_GetSlice(self, start, stop);
        }
        else {
            result = PyList_New(slicelength);
            if (!result)
                return NULL;

            for (cur = start, i = 0; i < slicelength; cur += step, i++) {
                it = PyFloat_FromDouble(self->coords[cur]);
                if (it == NULL) {
                    Py_DECREF(result);
                    return NULL;
                }
                PyList_SET_ITEM(result, i, it);
            }
            return result;
        }
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "vector indices must be integers, not %.200s",
                     Py_TYPE(key)->tp_name);
        return NULL;
    }
}

/* Parts of this function are adapted from python's list_ass_subscript */
static int
vector_ass_subscript(pgVector *self, PyObject *key, PyObject *value)
{
    if (PyIndex_Check(key)) {
        Py_ssize_t i = PyNumber_AsSsize_t(key, PyExc_IndexError);
        if (i == -1 && PyErr_Occurred())
            return -1;
        if (i < 0)
            i += self->dim;
        return vector_SetItem(self, i, value);
    }
    else if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step, slicelength;

        if (PySlice_GetIndicesEx((PyObject *)key, self->dim, &start, &stop,
                                 &step, &slicelength) < 0) {
            return -1;
        }

        if (step == 1)
            return vector_SetSlice(self, start, stop, value);

        /* Make sure s[5:2] = [..] inserts at the right place:
           before 5, not before 2. */
        if ((step < 0 && start < stop) || (step > 0 && start > stop))
            stop = start;

        if (value == NULL) {
            /* delete slice not supported */
            PyErr_SetString(PyExc_TypeError,
                            "Deletion of vector components is not supported.");
            return -1;
        }
        else {
            /* assign slice */
            double seqitems[VECTOR_MAX_SIZE];
            Py_ssize_t cur, i;

            if (!PySequence_AsVectorCoords(value, seqitems, slicelength))
                return -1;
            for (cur = start, i = 0; i < slicelength; cur += step, i++) {
                self->coords[cur] = seqitems[i];
            }
            return 0;
        }
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "list indices must be integers, not %.200s",
                     Py_TYPE(key)->tp_name);
        return -1;
    }
}

static PyMappingMethods vector_as_mapping = {
    .mp_length = (lenfunc)vector_len,
    .mp_subscript = (binaryfunc)vector_subscript,
    .mp_ass_subscript = (objobjargproc)vector_ass_subscript,
};

static int
vector_set_component(pgVector *self, PyObject *value, int component)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the x attribute");
        return -1;
    }
    if (component >= self->dim) {
        PyErr_BadInternalCall();
        return -1;
    }

    self->coords[component] = PyFloat_AsDouble(value);
    if (PyErr_Occurred())
        return -1;
    return 0;
}

static PyObject *
vector_getx(pgVector *self, void *closure)
{
    return PyFloat_FromDouble(self->coords[0]);
}

static int
vector_setx(pgVector *self, PyObject *value, void *closure)
{
    return vector_set_component(self, value, 0);
}

static PyObject *
vector_gety(pgVector *self, void *closure)
{
    return PyFloat_FromDouble(self->coords[1]);
}

static int
vector_sety(pgVector *self, PyObject *value, void *closure)
{
    return vector_set_component(self, value, 1);
}

static PyObject *
vector_getz(pgVector *self, void *closure)
{
    return PyFloat_FromDouble(self->coords[2]);
}

static int
vector_setz(pgVector *self, PyObject *value, void *closure)
{
    return vector_set_component(self, value, 2);
}

#ifdef PYGAME_MATH_VECTOR_HAVE_W
static PyObject *
vector_getw(pgVector *self, void *closure)
{
    return PyFloat_FromDouble(self->coords[3]);
}

static int
vector_setw(pgVector *self, PyObject *value, void *closure)
{
    return vector_set_component(self, value, 3);
}
#endif

static PyObject *
vector_richcompare(PyObject *o1, PyObject *o2, int op)
{
    Py_ssize_t i;
    double diff;
    double other_coords[VECTOR_MAX_SIZE];
    pgVector *vec;
    PyObject *other;

    if (pgVector_Check(o1)) {
        vec = (pgVector *)o1;
        other = o2;
    }
    else {
        vec = (pgVector *)o2;
        other = o1;
    }
    if (!pgVectorCompatible_Check(other, vec->dim)) {
        if (op == Py_EQ)
            Py_RETURN_FALSE;
        else if (op == Py_NE)
            Py_RETURN_TRUE;
        else {
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
    }

    if (!PySequence_AsVectorCoords(other, other_coords, vec->dim)) {
        return NULL;
    }

    switch (op) {
        case Py_EQ:
            for (i = 0; i < vec->dim; i++) {
                diff = vec->coords[i] - other_coords[i];
                /* test diff != diff to catch NaN */
                if ((diff != diff) || (fabs(diff) >= vec->epsilon)) {
                    Py_RETURN_FALSE;
                }
            }
            Py_RETURN_TRUE;
        case Py_NE:
            for (i = 0; i < vec->dim; i++) {
                diff = vec->coords[i] - other_coords[i];
                if ((diff != diff) || (fabs(diff) >= vec->epsilon)) {
                    Py_RETURN_TRUE;
                }
            }
            Py_RETURN_FALSE;
        default:
            PyErr_SetString(PyExc_TypeError,
                            "This operation is not supported by vectors");
            return NULL;
    }
}

static PyObject *
vector_length(pgVector *self, PyObject *_null)
{
    double length_squared =
        _scalar_product(self->coords, self->coords, self->dim);
    return PyFloat_FromDouble(sqrt(length_squared));
}

static PyObject *
vector_length_squared(pgVector *self, PyObject *_null)
{
    double length_squared =
        _scalar_product(self->coords, self->coords, self->dim);
    return PyFloat_FromDouble(length_squared);
}

static PyObject *
vector_normalize(pgVector *self, PyObject *_null)
{
    pgVector *ret = _vector_subtype_new(self);
    if (ret == NULL) {
        return NULL;
    }
    memcpy(ret->coords, self->coords, sizeof(ret->coords[0]) * ret->dim);

    if (!vector_normalize_ip(ret, NULL)) {
        return NULL;
    }
    return (PyObject *)ret;
}

static PyObject *
vector_normalize_ip(pgVector *self, PyObject *_null)
{
    Py_ssize_t i;
    double length;

    length = sqrt(_scalar_product(self->coords, self->coords, self->dim));

    if (length == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Can't normalize Vector of length Zero");
        return NULL;
    }

    for (i = 0; i < self->dim; ++i)
        self->coords[i] /= length;

    Py_RETURN_NONE;
}

static PyObject *
vector_is_normalized(pgVector *self, PyObject *_null)
{
    double length_squared =
        _scalar_product(self->coords, self->coords, self->dim);
    if (fabs(length_squared - 1) < self->epsilon)
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

static PyObject *
vector_dot(pgVector *self, PyObject *other)
{
    double other_coords[VECTOR_MAX_SIZE];
    if (!PySequence_AsVectorCoords(other, other_coords, self->dim)) {
        PyErr_SetString(PyExc_TypeError,
                        "Cannot perform dot product with this type.");
        return NULL;
    }
    return PyFloat_FromDouble(
        _scalar_product(self->coords, other_coords, self->dim));
}

static PyObject *
vector_scale_to_length(pgVector *self, PyObject *length)
{
    Py_ssize_t i;
    double new_length, old_length;
    double fraction;

    new_length = PyFloat_AsDouble(length);
    if (PyErr_Occurred()) {
        return NULL;
    }

    old_length = sqrt(_scalar_product(self->coords, self->coords, self->dim));

    if (old_length < self->epsilon) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot scale a vector with zero length");
        return NULL;
    }

    fraction = new_length / old_length;
    for (i = 0; i < self->dim; ++i)
        self->coords[i] *= fraction;

    Py_RETURN_NONE;
}

static void
_vector_move_towards_helper(Py_ssize_t dim, double *origin_coords,
                            double *target_coords, double max_distance)
{
    Py_ssize_t i;
    double frac, dist, delta[VECTOR_MAX_SIZE];
    if (max_distance == 0)
        return;

    for (i = 0; i < dim; ++i)
        delta[i] = target_coords[i] - origin_coords[i];

    /* Get magnitude of Vector */
    dist = sqrt(_scalar_product(delta, delta, dim));
    if (dist == 0) {
        /* origin and target are same, return early (this also makes sure
         * that frac is never NaN) */
        return;
    }

    if (dist <= max_distance) {
        /* Return target Vector */
        for (i = 0; i < dim; ++i)
            origin_coords[i] = target_coords[i];
        return;
    }

    frac = max_distance / dist;
    for (i = 0; i < dim; ++i)
        origin_coords[i] += delta[i] * frac;
}

static PyObject *
vector_move_towards(pgVector *self, PyObject *args)
{
    Py_ssize_t i;
    PyObject *target;
    double target_coords[VECTOR_MAX_SIZE];
    double max_distance;
    pgVector *ret;

    if (!PyArg_ParseTuple(args, "Od:move_towards", &target, &max_distance))
        return NULL;

    if (!pgVectorCompatible_Check(target, self->dim)) {
        PyErr_SetString(PyExc_TypeError,
                        "Target Vector is not the same size as self");
        return NULL;
    }

    if (!PySequence_AsVectorCoords(target, target_coords, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "Expected Vector as argument 1");
        return NULL;
    }

    ret = _vector_subtype_new(self);
    if (ret == NULL)
        return NULL;

    for (i = 0; i < self->dim; ++i)
        ret->coords[i] = self->coords[i];

    _vector_move_towards_helper(self->dim, ret->coords, target_coords,
                                max_distance);

    return (PyObject *)ret;
}

static PyObject *
vector_move_towards_ip(pgVector *self, PyObject *args)
{
    PyObject *target;
    double target_coords[VECTOR_MAX_SIZE];
    double max_distance;

    if (!PyArg_ParseTuple(args, "Od:move_towards_ip", &target, &max_distance))
        return NULL;

    if (!pgVectorCompatible_Check(target, self->dim)) {
        PyErr_SetString(PyExc_TypeError,
                        "Target Vector is not the same size as self");
        return NULL;
    }

    if (!PySequence_AsVectorCoords(target, target_coords, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "Expected Vector as argument 1");
        return NULL;
    }

    _vector_move_towards_helper(self->dim, self->coords, target_coords,
                                max_distance);

    Py_RETURN_NONE;
}

static PyObject *
vector_slerp(pgVector *self, PyObject *args)
{
    Py_ssize_t i;
    PyObject *other;
    pgVector *ret;
    double other_coords[VECTOR_MAX_SIZE];
    double tmp, angle, t, length1, length2, f0, f1, f2;

    if (!PyArg_ParseTuple(args, "Od:slerp", &other, &t)) {
        return NULL;
    }
    if (!PySequence_AsVectorCoords(other, other_coords, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "Argument 1 must be a vector.");
        return NULL;
    }
    if (fabs(t) > 1) {
        PyErr_SetString(PyExc_ValueError,
                        "Argument 2 must be in range [-1, 1].");
        return NULL;
    }

    length1 = sqrt(_scalar_product(self->coords, self->coords, self->dim));
    length2 = sqrt(_scalar_product(other_coords, other_coords, self->dim));
    if ((length1 < self->epsilon) || (length2 < self->epsilon)) {
        PyErr_SetString(PyExc_ValueError, "can't use slerp with Zero-Vector");
        return NULL;
    }
    tmp = (_scalar_product(self->coords, other_coords, self->dim) /
           (length1 * length2));
    /* make sure tmp is in the range [-1:1] so acos won't return NaN */
    tmp = (tmp < -1 ? -1 : (tmp > 1 ? 1 : tmp));
    angle = acos(tmp);

    /* if t < 0 we take the long arch of the greate circle to the destiny */
    if (t < 0) {
        angle -= 2 * M_PI;
        t = -t;
    }
    if (self->coords[0] * other_coords[1] < self->coords[1] * other_coords[0])
        angle *= -1;

    ret = _vector_subtype_new(self);
    if (ret == NULL) {
        return NULL;
    }
    /* special case angle==0 and angle==360 */
    if ((fabs(angle) < self->epsilon) ||
        (fabs(fabs(angle) - 2 * M_PI) < self->epsilon)) {
        /* approximate with lerp, because slerp diverges with 1/sin(angle) */
        for (i = 0; i < self->dim; ++i)
            ret->coords[i] = self->coords[i] * (1 - t) + other_coords[i] * t;
    }
    /* special case angle==180 and angle==-180 */
    else if (fabs(fabs(angle) - M_PI) < self->epsilon) {
        PyErr_SetString(PyExc_ValueError,
                        "SLERP with 180 degrees is undefined.");
        Py_DECREF(ret);
        return NULL;
    }
    else {
        f0 = ((length2 - length1) * t + length1) / sin(angle);
        f1 = sin(angle * (1 - t)) / length1;
        f2 = sin(angle * t) / length2;
        for (i = 0; i < self->dim; ++i) {
            ret->coords[i] =
                (self->coords[i] * f1 + other_coords[i] * f2) * f0;
        }
    }
    return (PyObject *)ret;
}

static PyObject *
vector_lerp(pgVector *self, PyObject *args)
{
    Py_ssize_t i;
    PyObject *other;
    pgVector *ret;
    double t;
    double other_coords[VECTOR_MAX_SIZE];

    if (!PyArg_ParseTuple(args, "Od:Vector.lerp", &other, &t)) {
        return NULL;
    }
    if (!PySequence_AsVectorCoords(other, other_coords, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "Expected Vector as argument 1");
        return NULL;
    }
    if (t < 0 || t > 1) {
        PyErr_SetString(PyExc_ValueError,
                        "Argument 2 must be in range [0, 1]");
        return NULL;
    }

    ret = _vector_subtype_new(self);
    if (ret == NULL) {
        return NULL;
    }
    for (i = 0; i < self->dim; ++i)
        ret->coords[i] = self->coords[i] * (1 - t) + other_coords[i] * t;
    return (PyObject *)ret;
}

static int
_vector_reflect_helper(double *dst_coords, const double *src_coords,
                       PyObject *normal, Py_ssize_t dim, double epsilon)
{
    Py_ssize_t i;
    double dot_product, norm_length;
    double norm_coords[VECTOR_MAX_SIZE];

    /* normalize the normal */
    if (!PySequence_AsVectorCoords(normal, norm_coords, dim))
        return 0;

    norm_length = _scalar_product(norm_coords, norm_coords, dim);

    if (norm_length < epsilon) {
        PyErr_SetString(PyExc_ValueError,
                        "Normal must not be of length zero.");
        return 0;
    }
    if (norm_length != 1) {
        norm_length = sqrt(norm_length);
        for (i = 0; i < dim; ++i)
            norm_coords[i] /= norm_length;
    }

    /* calculate the dot_product for the projection */
    dot_product = _scalar_product(src_coords, norm_coords, dim);

    for (i = 0; i < dim; ++i)
        dst_coords[i] = src_coords[i] - 2 * norm_coords[i] * dot_product;
    return 1;
}

static PyObject *
vector_reflect(pgVector *self, PyObject *normal)
{
    pgVector *ret = _vector_subtype_new(self);

    if (ret == NULL) {
        return NULL;
    }

    if (!_vector_reflect_helper(ret->coords, self->coords, normal, self->dim,
                                self->epsilon)) {
        return NULL;
    }
    return (PyObject *)ret;
}

static PyObject *
vector_reflect_ip(pgVector *self, PyObject *normal)
{
    double tmp_coords[VECTOR_MAX_SIZE];

    if (!_vector_reflect_helper(tmp_coords, self->coords, normal, self->dim,
                                self->epsilon)) {
        return NULL;
    }
    memcpy(self->coords, tmp_coords, self->dim * sizeof(tmp_coords[0]));
    Py_RETURN_NONE;
}

static double
_vector_distance_helper(pgVector *self, PyObject *other)
{
    Py_ssize_t i, dim = self->dim;
    double distance_squared = 0;

    /* Specialised fastpath for Vector-Vector distance calculation*/
    if (pgVector_Check(other)) {
        pgVector *otherv = (pgVector *)other;
        double dx, dy;

        if (dim != otherv->dim) {
            PyErr_SetString(PyExc_ValueError, "Vectors must be the same size");
            return -1;
        }

        dx = otherv->coords[0] - self->coords[0];
        dy = otherv->coords[1] - self->coords[1];

        distance_squared = dx * dx + dy * dy;

        if (dim == 3) {
            double dz;
            dz = otherv->coords[2] - self->coords[2];
            distance_squared += dz * dz;
        }
    }
    /* Vector-Sequence distance calculation*/
    else {
        double tmp;
        PyObject *fast_seq = PySequence_Fast(other, "A sequence was expected");
        if (!fast_seq) {
            return -1;
        }

        if (PySequence_Fast_GET_SIZE(fast_seq) != dim) {
            PyErr_SetString(PyExc_ValueError,
                            "Vector and sequence must be the same size");
            return -1;
        }

        for (i = 0; i < dim; ++i) {
            tmp = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fast_seq, i)) -
                  self->coords[i];
            if (PyErr_Occurred())
                return -1;
            distance_squared += tmp * tmp;
        }
    }
    return distance_squared;
}

static PyObject *
vector_distance_to(pgVector *self, PyObject *other)
{
    double distance_squared = _vector_distance_helper(self, other);
    if (distance_squared < 0 && PyErr_Occurred())
        return NULL;
    return PyFloat_FromDouble(sqrt(distance_squared));
}

static PyObject *
vector_distance_squared_to(pgVector *self, PyObject *other)
{
    double distance_squared = _vector_distance_helper(self, other);
    if (distance_squared < 0 && PyErr_Occurred())
        return NULL;
    return PyFloat_FromDouble(distance_squared);
}

static int
_vector_check_snprintf_success(int return_code, int max_size)
{
    if (return_code < 0) {
        PyErr_SetString(PyExc_SystemError,
                        "internal snprintf call went wrong! Please report "
                        "this to github.com/pygame/pygame/issues");
        return 0;
    }
    if (return_code >= max_size) {
        PyErr_SetString(PyExc_SystemError,
                        "Internal buffer to small for snprintf! Please report "
                        "this to github.com/pygame/pygame/issues");
        return 0;
    }
    return 1;
}

static PyObject *
vector_repr(pgVector *self)
{
    /* The repr() of the largest possible Vector3 looks like
     * "<Vector3({d}, {d}, {d})>" where 'd' has a maximum size of 32 bytes
     *  so allocate a 16 + 3 * 32 == 112 byte buffer
     */
    char buffer[STRING_BUF_SIZE_REPR];
    int tmp;

    if (self->dim == 2) {
        tmp = PyOS_snprintf(buffer, STRING_BUF_SIZE_REPR, "<Vector2(%g, %g)>",
                            self->coords[0], self->coords[1]);
    }
    else if (self->dim == 3) {
        tmp = PyOS_snprintf(buffer, STRING_BUF_SIZE_REPR,
                            "<Vector3(%g, %g, %g)>", self->coords[0],
                            self->coords[1], self->coords[2]);
    }
    else {
        return RAISE(
            PyExc_NotImplementedError,
            "repr() for Vectors of higher dimensions are not implemented yet");
    }

    if (!_vector_check_snprintf_success(tmp, STRING_BUF_SIZE_REPR))
        return NULL;

    return PyUnicode_FromString(buffer);
}

static PyObject *
vector_str(pgVector *self)
{
    /* The str() of the largest possible Vector3 looks like
     * "[{d}, {d}, {d}]" where 'd' has a maximum size of 32 bytes
     *  so allocate a 7 + 3 * 32 == 103 byte buffer
     */
    char buffer[STRING_BUF_SIZE_STR];
    int tmp;

    if (self->dim == 2) {
        tmp = PyOS_snprintf(buffer, STRING_BUF_SIZE_STR, "[%g, %g]",
                            self->coords[0], self->coords[1]);
    }
    else if (self->dim == 3) {
        tmp = PyOS_snprintf(buffer, STRING_BUF_SIZE_STR, "[%g, %g, %g]",
                            self->coords[0], self->coords[1], self->coords[2]);
    }
    else {
        return RAISE(
            PyExc_NotImplementedError,
            "repr() for Vectors of higher dimensions are not implemented yet");
    }

    if (!_vector_check_snprintf_success(tmp, STRING_BUF_SIZE_STR))
        return NULL;

    return PyUnicode_FromString(buffer);
}

static PyObject *
vector_project_onto(pgVector *self, PyObject *other)
{
    Py_ssize_t i;
    pgVector *ret;
    double other_coords[VECTOR_MAX_SIZE];
    double factor;
    double a_dot_b;
    double b_dot_b;

    if (!PySequence_AsVectorCoords(other, other_coords, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "Expected Vector as argument 1");
        return NULL;
    }

    ret = _vector_subtype_new(self);
    if (ret == NULL) {
        return NULL;
    }

    a_dot_b = _scalar_product(self->coords, other_coords, self->dim);
    b_dot_b = _scalar_product(other_coords, other_coords, self->dim);

    if (b_dot_b < self->epsilon) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot project onto a vector with zero length");
        return NULL;
    }

    factor = a_dot_b / b_dot_b;

    for (i = 0; i < self->dim; ++i)
        ret->coords[i] = other_coords[i] * factor;

    return (PyObject *)ret;
}

/* This method first tries normal attribute access. If successful we're
 * done. If not we try swizzling. Here we have 3 different outcomes:
 *  1) swizzling works. we return the result as a tuple
 *  2) swizzling fails because it wasn't a valid swizzle. we return the
 *     original AttributeError
 *  3) swizzling fails due to some internal error. we return this error.
 */
static PyObject *
vector_getAttr_swizzle(pgVector *self, PyObject *attr_name)
{
    double value;
    double *coords;
    Py_ssize_t i, idx, len;
    PyObject *attr_unicode = NULL;
    const char *attr = NULL;
    PyObject *res = NULL;

    len = PySequence_Length(attr_name);

    if (len == 1) {
        return PyObject_GenericGetAttr((PyObject *)self, attr_name);
    }

    if (len < 0)
        goto swizzle_failed;
    coords = self->coords;
    attr_unicode = PyUnicode_FromObject(attr_name);
    if (attr_unicode == NULL)
        goto swizzle_failed;
    attr = PyUnicode_AsUTF8AndSize(attr_unicode, &len);
    if (attr == NULL)
        goto internal_error;
    /* If we are not a swizzle, go straight to GenericGetAttr. */
    if ((attr[0] != 'x') && (attr[0] != 'y') && (attr[0] != 'z') &&
        (attr[0] != 'w')) {
        goto swizzle_failed;
    }

    if (len == 2 || len == 3) {
        res = (PyObject *)pgVector_NEW((int)len);
    }
    else {
        /* More than 3, we return a tuple. */
        res = (PyObject *)PyTuple_New(len);
    }
    if (res == NULL)
        goto internal_error;
    for (i = 0; i < len; i++) {
        switch (attr[i]) {
            case 'x':
            case 'y':
            case 'z':
                idx = attr[i] - 'x';
                goto swizzle_idx;
            case 'w':
                idx = 3;

            swizzle_idx:
                if (idx >= self->dim) {
                    goto swizzle_failed;
                };
                value = coords[idx];
                break;

            case '0':
                value = 0.0f;
                break;
            case '1':
                value = 1.0f;
                break;

            default:
                goto swizzle_failed;
        }
        if (len == 2 || len == 3) {
            ((pgVector *)res)->coords[i] = value;
        }
        else {
            if (PyTuple_SetItem(res, i, PyFloat_FromDouble(value)) != 0)
                goto internal_error;
        }
    }
    /* swizzling succeeded! */
    Py_DECREF(attr_unicode);
    return res;

    /* swizzling failed! clean up and return NULL */
swizzle_failed:
    Py_XDECREF(res);
    Py_XDECREF(attr_unicode);
    return PyObject_GenericGetAttr((PyObject *)self, attr_name);
internal_error:
    Py_XDECREF(res);
    Py_XDECREF(attr_unicode);
    return NULL;
}

static int
vector_setAttr_swizzle(pgVector *self, PyObject *attr_name, PyObject *val)
{
    const char *attr = NULL;
    PyObject *attr_unicode;
    Py_ssize_t len = PySequence_Length(attr_name);
    double entry[VECTOR_MAX_SIZE];
    int entry_was_set[VECTOR_MAX_SIZE];
    int swizzle_err = SWIZZLE_ERR_NO_ERR;
    Py_ssize_t i;

    if (len == 1)
        return PyObject_GenericSetAttr((PyObject *)self, attr_name, val);

    /* if swizzling is enabled first try swizzle */
    for (i = 0; i < self->dim; ++i)
        entry_was_set[i] = 0;

    /* handle string and unicode uniformly */
    attr_unicode = PyUnicode_FromObject(attr_name);
    if (attr_unicode == NULL)
        return -1;
    attr = PyUnicode_AsUTF8AndSize(attr_unicode, &len);

    if (attr == NULL) {
        Py_DECREF(attr_unicode);
        return -1;
    }

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
                Py_DECREF(attr_unicode);
                return PyObject_GenericSetAttr((PyObject *)self, attr_name,
                                               val);
        }
        if (idx >= self->dim) {
            /* swizzle failed. attempt generic attribute setting */
            Py_DECREF(attr_unicode);
            return PyObject_GenericSetAttr((PyObject *)self, attr_name, val);
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
    Py_DECREF(attr_unicode);

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
            PyErr_SetString(PyExc_RuntimeError,
                            "Unhandled error in swizzle code. Please report "
                            "this bug to github.com/pygame/pygame/issues");
            return -1;
    }
}

#if 0
static Py_ssize_t
vector_readbuffer(pgVector *self, Py_ssize_t segment, void **ptrptr)
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
vector_writebuffer(pgVector *self, Py_ssize_t segment, void **ptrptr)
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
vector_segcount(pgVector *self, Py_ssize_t *lenp)
{
    if (lenp) {
        *lenp = self->dim * sizeof(self->coords[0]);
    }
    return 1;
}

static int
vector_getbuffer(pgVector *self, Py_buffer *view, int flags)
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
vector_releasebuffer(pgVector *self, Py_buffer *view)
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
#endif

/*********************************************************************
 * vector2 specific functions
 *********************************************************************/

static PyObject *
vector2_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    pgVector *vec = (pgVector *)type->tp_alloc(type, 0);

    if (vec != NULL) {
        vec->dim = 2;
        vec->epsilon = VECTOR_EPSILON;
        vec->coords = PyMem_New(double, vec->dim);
        if (vec->coords == NULL) {
            Py_TYPE(vec)->tp_free((PyObject *)vec);
            return NULL;
        }
    }

    return (PyObject *)vec;
}

static int
_vector2_set(pgVector *self, PyObject *xOrSequence, PyObject *y)
{
    if (xOrSequence) {
        if (RealNumber_Check(xOrSequence)) {
            self->coords[0] = PyFloat_AsDouble(xOrSequence);
            /* scalar constructor. */
            if (y == NULL) {
                self->coords[1] = self->coords[0];
                return 0;
            }
        }
        else if (pgVectorCompatible_Check(xOrSequence, self->dim)) {
            if (!PySequence_AsVectorCoords(xOrSequence, self->coords, 2))
                return -1;
            else
                return 0;
        }
        else if (PyUnicode_Check(xOrSequence)) {
            char *delimiter[3] = {"<Vector2(", ", ", ")>"};
            Py_ssize_t error_code;
            error_code = _vector_coords_from_string(xOrSequence, delimiter,
                                                    self->coords, self->dim);
            if (error_code == -2) {
                return -1;
            }
            else if (error_code == -1) {
                goto error;
            }
            return 0;
        }
        else {
            goto error;
        }
    }
    else {
        self->coords[0] = 0.;
        if (y == NULL) {
            self->coords[1] = 0.;
            return 0;
        }
    }

    if (RealNumber_Check(y)) {
        self->coords[1] = PyFloat_AsDouble(y);
    }
    else {
        goto error;
    }

    /* success initialization */
    return 0;
error:
    PyErr_SetString(PyExc_ValueError,
                    "Vector2 must be set with 2 real numbers, a "
                    "sequence of 2 real numbers, or "
                    "another Vector2 instance");
    return -1;
}

static int
vector2_init(pgVector *self, PyObject *args, PyObject *kwds)
{
    PyObject *xOrSequence = NULL, *y = NULL;
    static char *kwlist[] = {"x", "y", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO:Vector2", kwlist,
                                     &xOrSequence, &y))
        return -1;

    return _vector2_set(self, xOrSequence, y);
}

static PyObject *
vector2_update(pgVector *self, PyObject *args, PyObject *kwds)
{
    PyObject *xOrSequence = NULL, *y = NULL;
    static char *kwlist[] = {"x", "y", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO:Vector2", kwlist,
                                     &xOrSequence, &y))
        return NULL;

    if (_vector2_set(self, xOrSequence, y) == 0)
        Py_RETURN_NONE;
    return NULL;
}

static int
_vector2_rotate_helper(double *dst_coords, const double *src_coords,
                       double angle, double epsilon)
{
    /* make sure angle is in range [0, 2 PI) */
    angle = fmod(angle, TWO_PI);
    if (angle < 0)
        angle += TWO_PI;

    /* special-case rotation by 0, 90, 180 and 270 degrees */
    if (fmod(angle + epsilon, M_PI_2) < 2 * epsilon) {
        switch ((int)((angle + epsilon) / M_PI_2)) {
            case 0: /* 0 degrees */
            case 4: /* 360 degree (see issue 214) */
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
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Please report this bug in vector2_rotate_helper to "
                    "the developers at github.com/pygame/pygame/issues");
                return 0;
        }
    }
    else {
        double sinValue, cosValue;
        sinValue = sin(angle);
        cosValue = cos(angle);

        dst_coords[0] = cosValue * src_coords[0] - sinValue * src_coords[1];
        dst_coords[1] = sinValue * src_coords[0] + cosValue * src_coords[1];
    }
    return 1;
}

static PyObject *
vector2_rotate_rad(pgVector *self, PyObject *angleObject)
{
    double angle;
    pgVector *ret;

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }

    ret = _vector_subtype_new(self);
    if (ret == NULL || !_vector2_rotate_helper(ret->coords, self->coords,
                                               angle, self->epsilon)) {
        Py_XDECREF(ret);
        return NULL;
    }
    return (PyObject *)ret;
}

static PyObject *
vector2_rotate_rad_ip(pgVector *self, PyObject *angleObject)
{
    double angle;
    double tmp[2];

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }

    memcpy(tmp, self->coords, 2 * sizeof(double));
    if (!_vector2_rotate_helper(self->coords, tmp, angle, self->epsilon)) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
vector2_rotate_ip_rad(pgVector *self, PyObject *angleObject)
{
    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "vector2_rotate_rad_ip() now has all the functionality "
                     "of vector2_rotate_ip_rad(), so vector2_rotate_ip_rad() "
                     "will be deprecated in pygame 2.1.1",
                     1) == -1) {
        return NULL;
    }
    return vector2_rotate_rad_ip(self, angleObject);
}

static PyObject *
vector2_rotate(pgVector *self, PyObject *angleObject)
{
    double angle;
    pgVector *ret;

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    angle = DEG2RAD(angle);

    ret = _vector_subtype_new(self);
    if (ret == NULL || !_vector2_rotate_helper(ret->coords, self->coords,
                                               angle, self->epsilon)) {
        Py_XDECREF(ret);
        return NULL;
    }
    return (PyObject *)ret;
}

static PyObject *
vector2_rotate_ip(pgVector *self, PyObject *angleObject)
{
    double angle;
    double tmp[2];

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    angle = DEG2RAD(angle);

    memcpy(tmp, self->coords, 2 * sizeof(double));
    if (!_vector2_rotate_helper(self->coords, tmp, angle, self->epsilon)) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
vector2_cross(pgVector *self, PyObject *other)
{
    double other_coords[2];
    if (self == (pgVector *)other)
        return PyFloat_FromDouble(0.0);

    if (!pgVectorCompatible_Check(other, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "cannot calculate cross Product");
        return NULL;
    }

    if (!PySequence_AsVectorCoords(other, other_coords, 2)) {
        return NULL;
    }

    return PyFloat_FromDouble((self->coords[0] * other_coords[1]) -
                              (self->coords[1] * other_coords[0]));
}

static PyObject *
vector2_project(pgVector *self, PyObject *other)
{
    return (PyObject *)vector_project_onto(self, other);
}

static PyObject *
vector2_angle_to(pgVector *self, PyObject *other)
{
    double angle;
    double other_coords[2];

    if (!pgVectorCompatible_Check(other, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "expected an vector.");
        return NULL;
    }

    if (!PySequence_AsVectorCoords(other, other_coords, 2)) {
        return NULL;
    }

    angle = (atan2(other_coords[1], other_coords[0]) -
             atan2(self->coords[1], self->coords[0]));
    return PyFloat_FromDouble(RAD2DEG(angle));
}

static PyObject *
vector2_as_polar(pgVector *self, PyObject *_null)
{
    double r, phi;
    r = sqrt(_scalar_product(self->coords, self->coords, self->dim));
    phi = RAD2DEG(atan2(self->coords[1], self->coords[0]));
    return Py_BuildValue("(dd)", r, phi);
}

static PyObject *
vector2_from_polar(pgVector *self, PyObject *args)
{
    double r, phi;
    if (!PyArg_ParseTuple(args, "(dd):Vector2.from_polar", &r, &phi)) {
        return NULL;
    }
    phi = DEG2RAD(phi);
    self->coords[0] = r * cos(phi);
    self->coords[1] = r * sin(phi);

    Py_RETURN_NONE;
}
static PyObject *
vector_getsafepickle(pgRectObject *self, void *_null)
{
    Py_RETURN_TRUE;
}
/* for pickling */
static PyObject *
vector2_reduce(PyObject *oself, PyObject *_null)
{
    pgVector *self = (pgVector *)oself;
    return Py_BuildValue("(O(dd))", Py_TYPE(oself), self->coords[0],
                         self->coords[1]);
}

static PyMethodDef vector2_methods[] = {
    {"length", (PyCFunction)vector_length, METH_NOARGS, DOC_VECTOR2LENGTH},
    {"length_squared", (PyCFunction)vector_length_squared, METH_NOARGS,
     DOC_VECTOR2LENGTHSQUARED},
    {"magnitude", (PyCFunction)vector_length, METH_NOARGS,
     DOC_VECTOR2MAGNITUDE},
    {"magnitude_squared", (PyCFunction)vector_length_squared, METH_NOARGS,
     DOC_VECTOR2MAGNITUDESQUARED},
    {"rotate", (PyCFunction)vector2_rotate, METH_O, DOC_VECTOR2ROTATE},
    {"rotate_ip", (PyCFunction)vector2_rotate_ip, METH_O, DOC_VECTOR2ROTATEIP},
    {"rotate_rad", (PyCFunction)vector2_rotate_rad, METH_O,
     DOC_VECTOR2ROTATERAD},
    {"rotate_rad_ip", (PyCFunction)vector2_rotate_rad_ip, METH_O,
     DOC_VECTOR2ROTATERADIP},
    {"rotate_ip_rad", (PyCFunction)vector2_rotate_ip_rad, METH_O,
     DOC_VECTOR2ROTATEIPRAD},
    {"move_towards", (PyCFunction)vector_move_towards, METH_VARARGS,
     DOC_VECTOR2MOVETOWARDS},
    {"move_towards_ip", (PyCFunction)vector_move_towards_ip, METH_VARARGS,
     DOC_VECTOR2MOVETOWARDSIP},
    {"slerp", (PyCFunction)vector_slerp, METH_VARARGS, DOC_VECTOR2SLERP},
    {"lerp", (PyCFunction)vector_lerp, METH_VARARGS, DOC_VECTOR2LERP},
    {"normalize", (PyCFunction)vector_normalize, METH_NOARGS,
     DOC_VECTOR2NORMALIZE},
    {"normalize_ip", (PyCFunction)vector_normalize_ip, METH_NOARGS,
     DOC_VECTOR2NORMALIZEIP},
    {"is_normalized", (PyCFunction)vector_is_normalized, METH_NOARGS,
     DOC_VECTOR2ISNORMALIZED},
    {"cross", (PyCFunction)vector2_cross, METH_O, DOC_VECTOR2CROSS},
    {"dot", (PyCFunction)vector_dot, METH_O, DOC_VECTOR2DOT},
    {"angle_to", (PyCFunction)vector2_angle_to, METH_O, DOC_VECTOR2ANGLETO},
    {"update", (PyCFunction)vector2_update, METH_VARARGS | METH_KEYWORDS,
     DOC_VECTOR2UPDATE},
    {"scale_to_length", (PyCFunction)vector_scale_to_length, METH_O,
     DOC_VECTOR2SCALETOLENGTH},
    {"reflect", (PyCFunction)vector_reflect, METH_O, DOC_VECTOR2REFLECT},
    {"reflect_ip", (PyCFunction)vector_reflect_ip, METH_O,
     DOC_VECTOR2REFLECTIP},
    {"distance_to", (PyCFunction)vector_distance_to, METH_O,
     DOC_VECTOR2DISTANCETO},
    {"distance_squared_to", (PyCFunction)vector_distance_squared_to, METH_O,
     DOC_VECTOR2DISTANCESQUAREDTO},
    {"elementwise", (PyCFunction)vector_elementwise, METH_NOARGS,
     DOC_VECTOR2ELEMENTWISE},
    {"as_polar", (PyCFunction)vector2_as_polar, METH_NOARGS,
     DOC_VECTOR2ASPOLAR},
    {"from_polar", (PyCFunction)vector2_from_polar, METH_VARARGS,
     DOC_VECTOR2FROMPOLAR},
    {"project", (PyCFunction)vector2_project, METH_O, DOC_VECTOR2PROJECT},
    {"copy", (PyCFunction)vector_copy, METH_NOARGS, DOC_VECTOR2COPY},
    {"__copy__", (PyCFunction)vector_copy, METH_NOARGS, NULL},
    {"clamp_magnitude", (PyCFunction)PG_FASTCALL_NAME(vector_clamp_magnitude),
     PG_FASTCALL, DOC_VECTOR2CLAMPMAGNITUDE},
    {"clamp_magnitude_ip",
     (PyCFunction)PG_FASTCALL_NAME(vector_clamp_magnitude_ip), PG_FASTCALL,
     DOC_VECTOR2CLAMPMAGNITUDEIP},
    {"__safe_for_unpickling__", (PyCFunction)vector_getsafepickle, METH_NOARGS,
     NULL},
    {"__reduce__", (PyCFunction)vector2_reduce, METH_NOARGS, NULL},

    {NULL} /* Sentinel */
};

static PyGetSetDef vector2_getsets[] = {
    {"x", (getter)vector_getx, (setter)vector_setx, NULL, NULL},
    {"y", (getter)vector_gety, (setter)vector_sety, NULL, NULL},
    {NULL, 0, NULL, NULL, NULL} /* Sentinel */
};

/********************************
 * pgVector2 type definition
 ********************************/

static PyTypeObject pgVector2_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.math.Vector2",
    .tp_basicsize = sizeof(pgVector),
    .tp_dealloc = (destructor)vector_dealloc,
    .tp_repr = (reprfunc)vector_repr,
    .tp_as_number = &vector_as_number,
    .tp_as_sequence = &vector_as_sequence,
    .tp_as_mapping = &vector_as_mapping,
    .tp_str = (reprfunc)vector_str,
    .tp_getattro = (getattrofunc)vector_getAttr_swizzle,
    .tp_setattro = (setattrofunc)vector_setAttr_swizzle,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = DOC_PYGAMEMATHVECTOR2,
    .tp_richcompare = (richcmpfunc)vector_richcompare,
    .tp_iter = vector_iter,
    .tp_methods = vector2_methods,
    .tp_members = vector_members,
    .tp_getset = vector2_getsets,
    .tp_init = (initproc)vector2_init,
    .tp_new = (newfunc)vector2_new,
};

/*************************************************************
 *  pgVector3 specific functions
 *************************************************************/

static PyObject *
vector3_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    pgVector *vec = (pgVector *)type->tp_alloc(type, 0);

    if (vec != NULL) {
        vec->dim = 3;
        vec->epsilon = VECTOR_EPSILON;
        vec->coords = PyMem_New(double, vec->dim);
        if (vec->coords == NULL) {
            Py_TYPE(vec)->tp_free((PyObject *)vec);
            return NULL;
        }
    }

    return (PyObject *)vec;
}

static int
_vector3_set(pgVector *self, PyObject *xOrSequence, PyObject *y, PyObject *z)
{
    if (xOrSequence) {
        if (RealNumber_Check(xOrSequence)) {
            self->coords[0] = PyFloat_AsDouble(xOrSequence);
            /* scalar constructor. */
            if (y == NULL && z == NULL) {
                self->coords[1] = self->coords[0];
                self->coords[2] = self->coords[0];
                return 0;
            }
        }
        else if (pgVectorCompatible_Check(xOrSequence, self->dim)) {
            if (!PySequence_AsVectorCoords(xOrSequence, self->coords, 3))
                return -1;
            else
                return 0;
        }
        else if (PyUnicode_Check(xOrSequence)) {
            char *delimiter[4] = {"<Vector3(", ", ", ", ", ")>"};
            Py_ssize_t error_code;
            error_code = _vector_coords_from_string(xOrSequence, delimiter,
                                                    self->coords, self->dim);
            if (error_code == -2) {
                return -1;
            }
            else if (error_code == -1) {
                goto error;
            }
            return 0;
        }
        else {
            goto error;
        }
    }
    else {
        self->coords[0] = 0.;
        self->coords[1] = 0.;
        self->coords[2] = 0.;
        return 0;
    }
    if (y && !z) {
        goto error;
    }
    else if (y && z) {
        if (RealNumber_Check(y) && RealNumber_Check(z)) {
            self->coords[1] = PyFloat_AsDouble(y);
            self->coords[2] = PyFloat_AsDouble(z);
        }
        else {
            goto error;
        }
    }
    /* success initialization */
    return 0;
error:
    PyErr_SetString(PyExc_ValueError,
                    "Vector3 must be set with 3 real numbers, a "
                    "sequence of 3 real numbers, or "
                    "another Vector3 instance");
    return -1;
}

static int
vector3_init(pgVector *self, PyObject *args, PyObject *kwds)
{
    PyObject *xOrSequence = NULL, *y = NULL, *z = NULL;
    static char *kwlist[] = {"x", "y", "z", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO:Vector3", kwlist,
                                     &xOrSequence, &y, &z))
        return -1;

    return _vector3_set(self, xOrSequence, y, z);
}

static PyObject *
vector3_update(pgVector *self, PyObject *args, PyObject *kwds)
{
    PyObject *xOrSequence = NULL, *y = NULL, *z = NULL;
    static char *kwlist[] = {"x", "y", "z", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO:Vector3", kwlist,
                                     &xOrSequence, &y, &z))
        return NULL;

    if (_vector3_set(self, xOrSequence, y, z) == 0)
        Py_RETURN_NONE;
    return NULL;
}

static int
_vector3_rotate_helper(double *dst_coords, const double *src_coords,
                       const double *axis_coords, double angle, double epsilon)
{
    double normalizationFactor;
    double axisLength2 = 0;
    double axis[3];
    int i;

    /* make sure angle is in range [0, 2 PI) */
    angle = fmod(angle, TWO_PI);
    if (angle < 0)
        angle += TWO_PI;

    for (i = 0; i < 3; ++i) {
        axisLength2 += axis_coords[i] * axis_coords[i];
        axis[i] = axis_coords[i];
    }

    /* Rotation axis may not be Zero */
    if (axisLength2 < epsilon) {
        PyErr_SetString(PyExc_ValueError, "Rotation Axis is to close to Zero");
        return 0;
    }

    /* normalize the axis */
    if (fabs(axisLength2 - 1) > epsilon) {
        normalizationFactor = 1. / sqrt(axisLength2);
        for (i = 0; i < 3; ++i)
            axis[i] *= normalizationFactor;
    }

    /* special-case rotation by 0, 90, 180 and 270 degrees */
    if (fmod(angle + epsilon, M_PI_2) < 2 * epsilon) {
        switch ((int)((angle + epsilon) / M_PI_2)) {
            case 0: /* 0 degrees */
            case 4: /* 360 degrees (see issue 214) */
                memcpy(dst_coords, src_coords, 3 * sizeof(src_coords[0]));
                break;
            case 1: /* 90 degrees */
                dst_coords[0] =
                    (src_coords[0] * (axis[0] * axis[0]) +
                     src_coords[1] * (axis[0] * axis[1] - axis[2]) +
                     src_coords[2] * (axis[0] * axis[2] + axis[1]));
                dst_coords[1] =
                    (src_coords[0] * (axis[0] * axis[1] + axis[2]) +
                     src_coords[1] * (axis[1] * axis[1]) +
                     src_coords[2] * (axis[1] * axis[2] - axis[0]));
                dst_coords[2] =
                    (src_coords[0] * (axis[0] * axis[2] - axis[1]) +
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
                dst_coords[0] =
                    (src_coords[0] * (axis[0] * axis[0]) +
                     src_coords[1] * (axis[0] * axis[1] + axis[2]) +
                     src_coords[2] * (axis[0] * axis[2] - axis[1]));
                dst_coords[1] =
                    (src_coords[0] * (axis[0] * axis[1] - axis[2]) +
                     src_coords[1] * (axis[1] * axis[1]) +
                     src_coords[2] * (axis[1] * axis[2] + axis[0]));
                dst_coords[2] =
                    (src_coords[0] * (axis[0] * axis[2] + axis[1]) +
                     src_coords[1] * (axis[1] * axis[2] - axis[0]) +
                     src_coords[2] * (axis[2] * axis[2]));
                break;
            default:
                /* this should NEVER happen and means a bug in the code */
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Please report this bug in vector3_rotate_helper to "
                    "the developers at github.com/pygame/pygame/issues");
                return 0;
        }
    }
    else {
        double sinValue = sin(angle);
        double cosValue = cos(angle);
        double cosComplement = 1 - cosValue;

        dst_coords[0] =
            (src_coords[0] * (cosValue + axis[0] * axis[0] * cosComplement) +
             src_coords[1] *
                 (axis[0] * axis[1] * cosComplement - axis[2] * sinValue) +
             src_coords[2] *
                 (axis[0] * axis[2] * cosComplement + axis[1] * sinValue));
        dst_coords[1] =
            (src_coords[0] *
                 (axis[0] * axis[1] * cosComplement + axis[2] * sinValue) +
             src_coords[1] * (cosValue + axis[1] * axis[1] * cosComplement) +
             src_coords[2] *
                 (axis[1] * axis[2] * cosComplement - axis[0] * sinValue));
        dst_coords[2] =
            (src_coords[0] *
                 (axis[0] * axis[2] * cosComplement - axis[1] * sinValue) +
             src_coords[1] *
                 (axis[1] * axis[2] * cosComplement + axis[0] * sinValue) +
             src_coords[2] * (cosValue + axis[2] * axis[2] * cosComplement));
    }
    return 1;
}

static PyObject *
vector3_rotate_rad(pgVector *self, PyObject *args)
{
    pgVector *ret;
    PyObject *axis;
    double axis_coords[3];
    double angle;

    if (!PyArg_ParseTuple(args, "dO:rotate", &angle, &axis)) {
        return NULL;
    }
    if (!pgVectorCompatible_Check(axis, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "axis must be a 3D Vector");
        return NULL;
    }
    if (!PySequence_AsVectorCoords(axis, axis_coords, 3)) {
        return NULL;
    }

    ret = _vector_subtype_new(self);
    if (ret == NULL ||
        !_vector3_rotate_helper(ret->coords, self->coords, axis_coords, angle,
                                self->epsilon)) {
        Py_XDECREF(ret);
        return NULL;
    }
    return (PyObject *)ret;
}

static PyObject *
vector3_rotate_rad_ip(pgVector *self, PyObject *args)
{
    PyObject *axis;
    double axis_coords[3];
    double angle;
    double tmp[3];

    if (!PyArg_ParseTuple(args, "dO:rotate", &angle, &axis)) {
        return NULL;
    }
    if (!pgVectorCompatible_Check(axis, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "axis must be a 3D Vector");
        return NULL;
    }
    if (!PySequence_AsVectorCoords(axis, axis_coords, 3)) {
        return NULL;
    }

    memcpy(tmp, self->coords, 3 * sizeof(self->coords[0]));
    if (!_vector3_rotate_helper(self->coords, tmp, axis_coords, angle,
                                self->epsilon)) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
vector3_rotate_ip_rad(pgVector *self, PyObject *angleObject)
{
    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "vector3_rotate_rad_ip() now has all the functionality "
                     "of vector3_rotate_ip_rad(), so vector3_rotate_ip_rad() "
                     "will be deprecated in pygame 2.1.1",
                     1) == -1) {
        return NULL;
    }
    return vector3_rotate_rad_ip(self, angleObject);
}

static PyObject *
vector3_rotate(pgVector *self, PyObject *args)
{
    pgVector *ret;
    PyObject *axis;
    double axis_coords[3];
    double angle;

    if (!PyArg_ParseTuple(args, "dO:rotate", &angle, &axis)) {
        return NULL;
    }
    angle = DEG2RAD(angle);
    if (!pgVectorCompatible_Check(axis, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "axis must be a 3D Vector");
        return NULL;
    }
    if (!PySequence_AsVectorCoords(axis, axis_coords, 3)) {
        return NULL;
    }

    ret = _vector_subtype_new(self);
    if (ret == NULL ||
        !_vector3_rotate_helper(ret->coords, self->coords, axis_coords, angle,
                                self->epsilon)) {
        Py_XDECREF(ret);
        return NULL;
    }
    return (PyObject *)ret;
}

static PyObject *
vector3_rotate_ip(pgVector *self, PyObject *args)
{
    PyObject *axis;
    double axis_coords[3];
    double angle;
    double tmp[3];

    if (!PyArg_ParseTuple(args, "dO:rotate_ip", &angle, &axis)) {
        return NULL;
    }
    angle = DEG2RAD(angle);
    if (!pgVectorCompatible_Check(axis, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "axis must be a 3D Vector");
        return NULL;
    }
    if (!PySequence_AsVectorCoords(axis, axis_coords, 3)) {
        return NULL;
    }

    memcpy(tmp, self->coords, 3 * sizeof(self->coords[0]));
    if (!_vector3_rotate_helper(self->coords, tmp, axis_coords, angle,
                                self->epsilon)) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
vector3_rotate_x_rad(pgVector *self, PyObject *angleObject)
{
    pgVector *ret;
    double sinValue, cosValue;
    double angle;

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    sinValue = sin(angle);
    cosValue = cos(angle);

    ret = _vector_subtype_new(self);
    if (ret == NULL) {
        return NULL;
    }
    ret->coords[0] = self->coords[0];
    ret->coords[1] = self->coords[1] * cosValue - self->coords[2] * sinValue;
    ret->coords[2] = self->coords[1] * sinValue + self->coords[2] * cosValue;
    return (PyObject *)ret;
}

static PyObject *
vector3_rotate_x_rad_ip(pgVector *self, PyObject *angleObject)
{
    double tmp_coords[3];
    double sinValue, cosValue;
    double angle;

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    sinValue = sin(angle);
    cosValue = cos(angle);
    memcpy(tmp_coords, self->coords, 3 * sizeof(tmp_coords[0]));

    self->coords[1] = tmp_coords[1] * cosValue - tmp_coords[2] * sinValue;
    self->coords[2] = tmp_coords[1] * sinValue + tmp_coords[2] * cosValue;
    Py_RETURN_NONE;
}

static PyObject *
vector3_rotate_x_ip_rad(pgVector *self, PyObject *angleObject)
{
    if (PyErr_WarnEx(
            PyExc_DeprecationWarning,
            "vector3_rotate_x_rad_ip() now has all the functionality of "
            "vector3_rotate_x_ip_rad(), so vector3_rotate_x_ip_rad() will "
            "be deprecated in pygame 2.1.1",
            1) == -1) {
        return NULL;
    }
    return vector3_rotate_x_rad_ip(self, angleObject);
}

static PyObject *
vector3_rotate_x(pgVector *self, PyObject *angleObject)
{
    pgVector *ret;
    double sinValue, cosValue;
    double angle;

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    angle = DEG2RAD(angle);
    sinValue = sin(angle);
    cosValue = cos(angle);

    ret = _vector_subtype_new(self);
    if (ret == NULL) {
        return NULL;
    }
    ret->coords[0] = self->coords[0];
    ret->coords[1] = self->coords[1] * cosValue - self->coords[2] * sinValue;
    ret->coords[2] = self->coords[1] * sinValue + self->coords[2] * cosValue;
    return (PyObject *)ret;
}

static PyObject *
vector3_rotate_x_ip(pgVector *self, PyObject *angleObject)
{
    double tmp_coords[3];
    double sinValue, cosValue;
    double angle;

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    angle = DEG2RAD(angle);
    sinValue = sin(angle);
    cosValue = cos(angle);
    memcpy(tmp_coords, self->coords, 3 * sizeof(tmp_coords[0]));

    self->coords[1] = tmp_coords[1] * cosValue - tmp_coords[2] * sinValue;
    self->coords[2] = tmp_coords[1] * sinValue + tmp_coords[2] * cosValue;
    Py_RETURN_NONE;
}

static PyObject *
vector3_rotate_y_rad(pgVector *self, PyObject *angleObject)
{
    pgVector *ret;
    double sinValue, cosValue;
    double angle;

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    sinValue = sin(angle);
    cosValue = cos(angle);

    ret = _vector_subtype_new(self);
    if (ret == NULL) {
        return NULL;
    }
    ret->coords[0] = self->coords[0] * cosValue + self->coords[2] * sinValue;
    ret->coords[1] = self->coords[1];
    ret->coords[2] = -self->coords[0] * sinValue + self->coords[2] * cosValue;

    return (PyObject *)ret;
}

static PyObject *
vector3_rotate_y_rad_ip(pgVector *self, PyObject *angleObject)
{
    double tmp_coords[3];
    double sinValue, cosValue;
    double angle;

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    sinValue = sin(angle);
    cosValue = cos(angle);
    memcpy(tmp_coords, self->coords, 3 * sizeof(tmp_coords[0]));

    self->coords[0] = tmp_coords[0] * cosValue + tmp_coords[2] * sinValue;
    self->coords[2] = -tmp_coords[0] * sinValue + tmp_coords[2] * cosValue;
    Py_RETURN_NONE;
}

static PyObject *
vector3_rotate_y_ip_rad(pgVector *self, PyObject *angleObject)
{
    if (PyErr_WarnEx(
            PyExc_DeprecationWarning,
            "vector3_rotate_y_rad_ip() now has all the functionality of "
            "vector3_rotate_y_ip_rad(), so vector3_rotate_y_ip_rad() will "
            "be deprecated in pygame 2.1.1",
            1) == -1) {
        return NULL;
    }
    return vector3_rotate_x_rad_ip(self, angleObject);
}

static PyObject *
vector3_rotate_y(pgVector *self, PyObject *angleObject)
{
    pgVector *ret;
    double sinValue, cosValue;
    double angle;

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    angle = DEG2RAD(angle);
    sinValue = sin(angle);
    cosValue = cos(angle);

    ret = _vector_subtype_new(self);
    if (ret == NULL) {
        return NULL;
    }
    ret->coords[0] = self->coords[0] * cosValue + self->coords[2] * sinValue;
    ret->coords[1] = self->coords[1];
    ret->coords[2] = -self->coords[0] * sinValue + self->coords[2] * cosValue;

    return (PyObject *)ret;
}

static PyObject *
vector3_rotate_y_ip(pgVector *self, PyObject *angleObject)
{
    double tmp_coords[3];
    double sinValue, cosValue;
    double angle;

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    angle = DEG2RAD(angle);
    sinValue = sin(angle);
    cosValue = cos(angle);
    memcpy(tmp_coords, self->coords, 3 * sizeof(tmp_coords[0]));

    self->coords[0] = tmp_coords[0] * cosValue + tmp_coords[2] * sinValue;
    self->coords[2] = -tmp_coords[0] * sinValue + tmp_coords[2] * cosValue;
    Py_RETURN_NONE;
}

static PyObject *
vector3_rotate_z_rad(pgVector *self, PyObject *angleObject)
{
    pgVector *ret;
    double sinValue, cosValue;
    double angle;

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    sinValue = sin(angle);
    cosValue = cos(angle);

    ret = _vector_subtype_new(self);
    if (ret == NULL) {
        return NULL;
    }
    ret->coords[0] = self->coords[0] * cosValue - self->coords[1] * sinValue;
    ret->coords[1] = self->coords[0] * sinValue + self->coords[1] * cosValue;
    ret->coords[2] = self->coords[2];

    return (PyObject *)ret;
}

static PyObject *
vector3_rotate_z_rad_ip(pgVector *self, PyObject *angleObject)
{
    double tmp_coords[3];
    double sinValue, cosValue;
    double angle;

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    sinValue = sin(angle);
    cosValue = cos(angle);
    memcpy(tmp_coords, self->coords, 3 * sizeof(tmp_coords[0]));

    self->coords[0] = tmp_coords[0] * cosValue - tmp_coords[1] * sinValue;
    self->coords[1] = tmp_coords[0] * sinValue + tmp_coords[1] * cosValue;
    Py_RETURN_NONE;
}

static PyObject *
vector3_rotate_z_ip_rad(pgVector *self, PyObject *angleObject)
{
    if (PyErr_WarnEx(
            PyExc_DeprecationWarning,
            "vector3_rotate_z_rad_ip() now has all the functionality of "
            "vector3_rotate_z_ip_rad(), so vector3_rotate_z_ip_rad() will "
            "be deprecated in pygame 2.1.1",
            1) == -1) {
        return NULL;
    }
    return vector3_rotate_x_rad_ip(self, angleObject);
}

static PyObject *
vector3_rotate_z(pgVector *self, PyObject *angleObject)
{
    pgVector *ret;
    double sinValue, cosValue;
    double angle;

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    angle = DEG2RAD(angle);
    sinValue = sin(angle);
    cosValue = cos(angle);

    ret = _vector_subtype_new(self);
    if (ret == NULL) {
        return NULL;
    }
    ret->coords[0] = self->coords[0] * cosValue - self->coords[1] * sinValue;
    ret->coords[1] = self->coords[0] * sinValue + self->coords[1] * cosValue;
    ret->coords[2] = self->coords[2];

    return (PyObject *)ret;
}

static PyObject *
vector3_rotate_z_ip(pgVector *self, PyObject *angleObject)
{
    double tmp_coords[3];
    double sinValue, cosValue;
    double angle;

    angle = PyFloat_AsDouble(angleObject);
    if (angle == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    angle = DEG2RAD(angle);
    sinValue = sin(angle);
    cosValue = cos(angle);
    memcpy(tmp_coords, self->coords, 3 * sizeof(tmp_coords[0]));

    self->coords[0] = tmp_coords[0] * cosValue - tmp_coords[1] * sinValue;
    self->coords[1] = tmp_coords[0] * sinValue + tmp_coords[1] * cosValue;
    Py_RETURN_NONE;
}

static PyObject *
vector3_cross(pgVector *self, PyObject *other)
{
    pgVector *ret;
    double *ret_coords;
    double *self_coords;
    double *other_coords;

    if (!pgVectorCompatible_Check(other, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "cannot calculate cross Product");
        return NULL;
    }

    self_coords = self->coords;
    if (pgVector_Check(other)) {
        other_coords = ((pgVector *)other)->coords;
    }
    else {
        other_coords = PyMem_New(double, self->dim);
        if (!other_coords) {
            return PyErr_NoMemory();
        }

        if (!PySequence_AsVectorCoords(other, other_coords, 3)) {
            PyMem_Free(other_coords);
            return NULL;
        }
    }

    ret = _vector_subtype_new(self);
    if (ret == NULL) {
        if (!pgVector_Check(other))
            PyMem_Free(other_coords);
        return NULL;
    }
    ret_coords = ret->coords;
    ret_coords[0] = ((self_coords[1] * other_coords[2]) -
                     (self_coords[2] * other_coords[1]));
    ret_coords[1] = ((self_coords[2] * other_coords[0]) -
                     (self_coords[0] * other_coords[2]));
    ret_coords[2] = ((self_coords[0] * other_coords[1]) -
                     (self_coords[1] * other_coords[0]));

    if (!pgVector_Check(other))
        PyMem_Free(other_coords);

    return (PyObject *)ret;
}

static PyObject *
vector3_angle_to(pgVector *self, PyObject *other)
{
    double angle, tmp, squared_length1, squared_length2;
    double other_coords[VECTOR_MAX_SIZE];

    if (!pgVectorCompatible_Check(other, self->dim)) {
        PyErr_SetString(PyExc_TypeError, "expected an vector.");
        return NULL;
    }

    if (!PySequence_AsVectorCoords(other, other_coords, self->dim)) {
        return NULL;
    }
    squared_length1 = _scalar_product(self->coords, self->coords, self->dim);
    squared_length2 = _scalar_product(other_coords, other_coords, self->dim);
    tmp = sqrt(squared_length1 * squared_length2);
    if (tmp == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "angle to zero vector is undefined.");
        return NULL;
    }
    angle = acos(_scalar_product(self->coords, other_coords, self->dim) / tmp);
    return PyFloat_FromDouble(RAD2DEG(angle));
}

static PyObject *
vector3_as_spherical(pgVector *self, PyObject *_null)
{
    double r, theta, phi;
    r = sqrt(_scalar_product(self->coords, self->coords, self->dim));
    if (r == 0.) {
        return Py_BuildValue("(ddd)", 0., 0., 0.);
    }
    theta = RAD2DEG(acos(self->coords[2] / r));
    phi = RAD2DEG(atan2(self->coords[1], self->coords[0]));
    return Py_BuildValue("(ddd)", r, theta, phi);
}

static PyObject *
vector3_from_spherical(pgVector *self, PyObject *args)
{
    double r, theta, phi;

    if (!PyArg_ParseTuple(args, "(ddd):vector3_from_spherical", &r, &theta,
                          &phi)) {
        return NULL;
    }
    theta = DEG2RAD(theta);
    phi = DEG2RAD(phi);
    self->coords[0] = r * sin(theta) * cos(phi);
    self->coords[1] = r * sin(theta) * sin(phi);
    self->coords[2] = r * cos(theta);

    Py_RETURN_NONE;
}

static PyObject *
vector3_project(pgVector *self, PyObject *other)
{
    return (PyObject *)vector_project_onto(self, other);
}

/* For pickling. */
static PyObject *
vector3_reduce(PyObject *oself, PyObject *_null)
{
    pgVector *self = (pgVector *)oself;
    return Py_BuildValue("(O(ddd))", Py_TYPE(oself), self->coords[0],
                         self->coords[1], self->coords[2]);
}

static PyMethodDef vector3_methods[] = {
    {"length", (PyCFunction)vector_length, METH_NOARGS, DOC_VECTOR3LENGTH},
    {"length_squared", (PyCFunction)vector_length_squared, METH_NOARGS,
     DOC_VECTOR3LENGTHSQUARED},
    {"magnitude", (PyCFunction)vector_length, METH_NOARGS,
     DOC_VECTOR3MAGNITUDE},
    {"magnitude_squared", (PyCFunction)vector_length_squared, METH_NOARGS,
     DOC_VECTOR3MAGNITUDESQUARED},
    {"rotate", (PyCFunction)vector3_rotate, METH_VARARGS, DOC_VECTOR3ROTATE},
    {"rotate_ip", (PyCFunction)vector3_rotate_ip, METH_VARARGS,
     DOC_VECTOR3ROTATEIP},
    {"rotate_rad", (PyCFunction)vector3_rotate_rad, METH_VARARGS,
     DOC_VECTOR3ROTATERAD},
    {"rotate_rad_ip", (PyCFunction)vector3_rotate_rad_ip, METH_VARARGS,
     DOC_VECTOR3ROTATERADIP},
    {"rotate_ip_rad", (PyCFunction)vector3_rotate_ip_rad, METH_VARARGS,
     DOC_VECTOR3ROTATEIPRAD},
    {"rotate_x", (PyCFunction)vector3_rotate_x, METH_O, DOC_VECTOR3ROTATEX},
    {"rotate_x_ip", (PyCFunction)vector3_rotate_x_ip, METH_O,
     DOC_VECTOR3ROTATEXIP},
    {"rotate_x_rad", (PyCFunction)vector3_rotate_x_rad, METH_O,
     DOC_VECTOR3ROTATEXRAD},
    {"rotate_x_rad_ip", (PyCFunction)vector3_rotate_x_rad_ip, METH_O,
     DOC_VECTOR3ROTATEXRADIP},
    {"rotate_x_ip_rad", (PyCFunction)vector3_rotate_x_ip_rad, METH_O,
     DOC_VECTOR3ROTATEXIPRAD},
    {"rotate_y", (PyCFunction)vector3_rotate_y, METH_O, DOC_VECTOR3ROTATEY},
    {"rotate_y_ip", (PyCFunction)vector3_rotate_y_ip, METH_O,
     DOC_VECTOR3ROTATEYIP},
    {"rotate_y_rad", (PyCFunction)vector3_rotate_y_rad, METH_O,
     DOC_VECTOR3ROTATEYRAD},
    {"rotate_y_rad_ip", (PyCFunction)vector3_rotate_y_rad_ip, METH_O,
     DOC_VECTOR3ROTATEYRADIP},
    {"rotate_y_ip_rad", (PyCFunction)vector3_rotate_y_ip_rad, METH_O,
     DOC_VECTOR3ROTATEYIPRAD},
    {"rotate_z", (PyCFunction)vector3_rotate_z, METH_O, DOC_VECTOR3ROTATEZ},
    {"rotate_z_ip", (PyCFunction)vector3_rotate_z_ip, METH_O,
     DOC_VECTOR3ROTATEZIP},
    {"rotate_z_rad", (PyCFunction)vector3_rotate_z_rad, METH_O,
     DOC_VECTOR3ROTATEZRAD},
    {"rotate_z_rad_ip", (PyCFunction)vector3_rotate_z_rad_ip, METH_O,
     DOC_VECTOR3ROTATEZRADIP},
    {"rotate_z_ip_rad", (PyCFunction)vector3_rotate_z_ip_rad, METH_O,
     DOC_VECTOR3ROTATEZIPRAD},
    {"move_towards", (PyCFunction)vector_move_towards, METH_VARARGS,
     DOC_VECTOR3MOVETOWARDS},
    {"move_towards_ip", (PyCFunction)vector_move_towards_ip, METH_VARARGS,
     DOC_VECTOR3MOVETOWARDSIP},
    {"slerp", (PyCFunction)vector_slerp, METH_VARARGS, DOC_VECTOR3SLERP},
    {"lerp", (PyCFunction)vector_lerp, METH_VARARGS, DOC_VECTOR3LERP},
    {"normalize", (PyCFunction)vector_normalize, METH_NOARGS,
     DOC_VECTOR3NORMALIZE},
    {"normalize_ip", (PyCFunction)vector_normalize_ip, METH_NOARGS,
     DOC_VECTOR3NORMALIZEIP},
    {"is_normalized", (PyCFunction)vector_is_normalized, METH_NOARGS,
     DOC_VECTOR3ISNORMALIZED},
    {"cross", (PyCFunction)vector3_cross, METH_O, DOC_VECTOR3CROSS},
    {"dot", (PyCFunction)vector_dot, METH_O, DOC_VECTOR3DOT},
    {"angle_to", (PyCFunction)vector3_angle_to, METH_O, DOC_VECTOR3ANGLETO},
    {"update", (PyCFunction)vector3_update, METH_VARARGS | METH_KEYWORDS,
     DOC_VECTOR3UPDATE},
    {"scale_to_length", (PyCFunction)vector_scale_to_length, METH_O,
     DOC_VECTOR3SCALETOLENGTH},
    {"reflect", (PyCFunction)vector_reflect, METH_O, DOC_VECTOR3REFLECT},
    {"reflect_ip", (PyCFunction)vector_reflect_ip, METH_O,
     DOC_VECTOR3REFLECTIP},
    {"distance_to", (PyCFunction)vector_distance_to, METH_O,
     DOC_VECTOR3DISTANCETO},
    {"distance_squared_to", (PyCFunction)vector_distance_squared_to, METH_O,
     DOC_VECTOR3DISTANCESQUAREDTO},
    {"elementwise", (PyCFunction)vector_elementwise, METH_NOARGS,
     DOC_VECTOR3ELEMENTWISE},
    {"as_spherical", (PyCFunction)vector3_as_spherical, METH_NOARGS,
     DOC_VECTOR3ASSPHERICAL},
    {"from_spherical", (PyCFunction)vector3_from_spherical, METH_VARARGS,
     DOC_VECTOR3FROMSPHERICAL},
    {"project", (PyCFunction)vector3_project, METH_O, DOC_VECTOR3PROJECT},
    {"copy", (PyCFunction)vector_copy, METH_NOARGS, DOC_VECTOR3COPY},
    {"__copy__", (PyCFunction)vector_copy, METH_NOARGS, NULL},
    {"clamp_magnitude", (PyCFunction)PG_FASTCALL_NAME(vector_clamp_magnitude),
     PG_FASTCALL, DOC_VECTOR3CLAMPMAGNITUDE},
    {"clamp_magnitude_ip",
     (PyCFunction)PG_FASTCALL_NAME(vector_clamp_magnitude_ip), PG_FASTCALL,
     DOC_VECTOR3CLAMPMAGNITUDEIP},
    {"__safe_for_unpickling__", (PyCFunction)vector_getsafepickle, METH_NOARGS,
     NULL},
    {"__reduce__", (PyCFunction)vector3_reduce, METH_NOARGS, NULL},

    {NULL} /* Sentinel */
};

static PyGetSetDef vector3_getsets[] = {
    {"x", (getter)vector_getx, (setter)vector_setx, NULL, NULL},
    {"y", (getter)vector_gety, (setter)vector_sety, NULL, NULL},
    {"z", (getter)vector_getz, (setter)vector_setz, NULL, NULL},
    {NULL, 0, NULL, NULL, NULL} /* Sentinel */
};

/********************************
 * pgVector3 type definition
 ********************************/

static PyTypeObject pgVector3_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.math.Vector3",
    .tp_basicsize = sizeof(pgVector),
    .tp_dealloc = (destructor)vector_dealloc,
    .tp_repr = (reprfunc)vector_repr,
    .tp_as_number = &vector_as_number,
    .tp_as_sequence = &vector_as_sequence,
    .tp_as_mapping = &vector_as_mapping,
    .tp_str = (reprfunc)vector_str,
    .tp_getattro = (getattrofunc)vector_getAttr_swizzle,
    .tp_setattro = (setattrofunc)vector_setAttr_swizzle,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = DOC_PYGAMEMATHVECTOR3,
    .tp_richcompare = (richcmpfunc)vector_richcompare,
    .tp_iter = vector_iter,
    .tp_methods = vector3_methods,
    .tp_members = vector_members,
    .tp_getset = vector3_getsets,
    .tp_init = (initproc)vector3_init,
    .tp_new = (newfunc)vector3_new,
};

/********************************************
 * pgVectorIterator type definition
 ********************************************/

static void
vectoriter_dealloc(vectoriter *it)
{
    Py_XDECREF(it->vec);
    PyObject_Free(it);
}

static PyObject *
vectoriter_next(vectoriter *it)
{
    assert(it != NULL);
    if (it->vec == NULL)
        return NULL;
    assert(pgVector_Check(it->vec));

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
    return PyLong_FromSsize_t(len);
}

static PyMethodDef vectoriter_methods[] = {
    {
        "__length_hint__",
        (PyCFunction)vectoriter_len,
        METH_NOARGS,
    },
    {NULL, NULL} /* sentinel */
};

static PyTypeObject pgVectorIter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.math.VectorIterator",
    .tp_basicsize = sizeof(vectoriter),
    .tp_dealloc = (destructor)vectoriter_dealloc,
    .tp_getattro = PyObject_GenericGetAttr,
    /* VectorIterator is not subtypable for now, no Py_TPFLAGS_BASETYPE */
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)vectoriter_next,
    .tp_methods = vectoriter_methods,
};

static PyObject *
vector_iter(PyObject *vec)
{
    vectoriter *it;
    if (!pgVector_Check(vec)) {
        PyErr_BadInternalCall();
        return NULL;
    }

    it = PyObject_New(vectoriter, &pgVectorIter_Type);
    if (it == NULL)
        return NULL;
    it->it_index = 0;
    Py_INCREF(vec);
    it->vec = (pgVector *)vec;
    return (PyObject *)it;
}

/*****************************************
 * elementwiseproxy
 *****************************************/

static void
vector_elementwiseproxy_dealloc(vector_elementwiseproxy *it)
{
    Py_XDECREF(it->vec);
    PyObject_Free(it);
}

static PyObject *
vector_elementwiseproxy_richcompare(PyObject *o1, PyObject *o2, int op)
{
    Py_ssize_t i, dim;
    int ret = 1;
    double diff, value;
    pgVector *vec;
    PyObject *other;

    if (vector_elementwiseproxy_Check(o1)) {
        vec = ((vector_elementwiseproxy *)o1)->vec;
        other = o2;
    }
    else {
        vec = ((vector_elementwiseproxy *)o2)->vec;
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
        other = (PyObject *)((vector_elementwiseproxy *)other)->vec;
    dim = vec->dim;

    if (pgVectorCompatible_Check(other, dim)) {
        double *other_coords = PyMem_New(double, dim);

        if (other_coords == NULL) {
            return NULL;
        }
        if (!PySequence_AsVectorCoords(other, other_coords, dim)) {
            PyMem_Free(other_coords);
            return NULL;
        }
        /* use diff == diff to check for NaN */
        /* TODO: how should NaN be handled with LT/LE/GT/GE? */
        switch (op) {
            case Py_EQ:
                for (i = 0; i < dim; i++) {
                    diff = vec->coords[i] - other_coords[i];
                    if ((diff != diff) || (fabs(diff) >= vec->epsilon)) {
                        ret = 0;
                        break;
                    }
                }
                break;
            case Py_NE:
                for (i = 0; i < dim; i++) {
                    diff = vec->coords[i] - other_coords[i];
                    if ((diff == diff) && (fabs(diff) < vec->epsilon)) {
                        ret = 0;
                        break;
                    }
                }
                break;
            case Py_LT:
                for (i = 0; i < dim; i++) {
                    if (vec->coords[i] >= other_coords[i]) {
                        ret = 0;
                        break;
                    }
                }
                break;
            case Py_LE:
                for (i = 0; i < dim; i++) {
                    if (vec->coords[i] > other_coords[i]) {
                        ret = 0;
                        break;
                    }
                }
                break;
            case Py_GT:
                for (i = 0; i < dim; i++) {
                    if (vec->coords[i] <= other_coords[i]) {
                        ret = 0;
                        break;
                    }
                }
                break;
            case Py_GE:
                for (i = 0; i < dim; i++) {
                    if (vec->coords[i] < other_coords[i]) {
                        ret = 0;
                        break;
                    }
                }
                break;
            default:
                PyMem_Free(other_coords);
                PyErr_BadInternalCall();
                return NULL;
        }
        PyMem_Free(other_coords);
    }
    else if (RealNumber_Check(other)) {
        /* the following PyFloat_AsDouble call should never fail because
           then RealNumber_Check should have returned false */
        value = PyFloat_AsDouble(other);
        switch (op) {
            case Py_EQ:
                for (i = 0; i < dim; i++) {
                    diff = vec->coords[i] - value;
                    if (diff != diff || fabs(diff) >= vec->epsilon) {
                        ret = 0;
                        break;
                    }
                }
                break;
            case Py_NE:
                for (i = 0; i < dim; i++) {
                    diff = vec->coords[i] - value;
                    if (diff == diff && fabs(diff) < vec->epsilon) {
                        ret = 0;
                        break;
                    }
                }
                break;
            case Py_LT:
                for (i = 0; i < dim; i++) {
                    if (vec->coords[i] >= value) {
                        ret = 0;
                        break;
                    }
                }
                break;
            case Py_LE:
                for (i = 0; i < dim; i++) {
                    if (vec->coords[i] > value) {
                        ret = 0;
                        break;
                    }
                }
                break;
            case Py_GT:
                for (i = 0; i < dim; i++) {
                    if (vec->coords[i] <= value) {
                        ret = 0;
                        break;
                    }
                }
                break;
            case Py_GE:
                for (i = 0; i < dim; i++) {
                    if (vec->coords[i] < value) {
                        ret = 0;
                        break;
                    }
                }
                break;
            default:
                PyErr_BadInternalCall();
                return NULL;
        }
    }
    else {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    return PyBool_FromLong(ret);
}

/*******************************************************
 * vector_elementwiseproxy PyNumber emulation routines
 *******************************************************/

static PyObject *
vector_elementwiseproxy_generic_math(PyObject *o1, PyObject *o2, int op)
{
    Py_ssize_t i, dim;
    double mod, other_value = 0.0;
    double other_coords[VECTOR_MAX_SIZE] = {0};
    PyObject *other;
    pgVector *vec, *ret;
    if (vector_elementwiseproxy_Check(o1)) {
        vec = ((vector_elementwiseproxy *)o1)->vec;
        other = o2;
    }
    else {
        other = o1;
        vec = ((vector_elementwiseproxy *)o2)->vec;
        op |= OP_ARG_REVERSE;
    }

    dim = vec->dim;

    if (vector_elementwiseproxy_Check(other))
        other = (PyObject *)((vector_elementwiseproxy *)other)->vec;

    if (pgVectorCompatible_Check(other, dim)) {
        op |= OP_ARG_VECTOR;
        if (!PySequence_AsVectorCoords(other, other_coords, dim))
            return NULL;
    }
    else if (RealNumber_Check(other)) {
        op |= OP_ARG_NUMBER;
        other_value = PyFloat_AsDouble(other);
    }
    else
        op |= OP_ARG_UNKNOWN;

    ret = _vector_subtype_new(vec);

    if (ret == NULL) {
        return NULL;
    }

    /* only handle special elementwise cases.
     * all others cases will be handled by the default clause */
    switch (op) {
        case OP_ADD | OP_ARG_NUMBER:
        case OP_ADD | OP_ARG_NUMBER | OP_ARG_REVERSE:
            for (i = 0; i < dim; i++) {
                ret->coords[i] = vec->coords[i] + other_value;
            }
            break;
        case OP_SUB | OP_ARG_NUMBER:
            for (i = 0; i < dim; i++)
                ret->coords[i] = vec->coords[i] - other_value;
            break;
        case OP_SUB | OP_ARG_NUMBER | OP_ARG_REVERSE:
            for (i = 0; i < dim; i++)
                ret->coords[i] = other_value - vec->coords[i];
            break;
        case OP_MUL | OP_ARG_VECTOR:
        case OP_MUL | OP_ARG_VECTOR | OP_ARG_REVERSE:
            for (i = 0; i < vec->dim; i++)
                ret->coords[i] = vec->coords[i] * other_coords[i];
            break;
        case OP_DIV | OP_ARG_VECTOR:
            for (i = 0; i < vec->dim; i++) {
                if (other_coords[i] == 0) {
                    PyErr_SetString(PyExc_ZeroDivisionError,
                                    "division by zero");
                    Py_DECREF(ret);
                    return NULL;
                }
                ret->coords[i] = vec->coords[i] / other_coords[i];
            }
            break;
        case OP_DIV | OP_ARG_VECTOR | OP_ARG_REVERSE:
            for (i = 0; i < vec->dim; i++) {
                if (vec->coords[i] == 0) {
                    PyErr_SetString(PyExc_ZeroDivisionError,
                                    "division by zero");
                    Py_DECREF(ret);
                    return NULL;
                }
                ret->coords[i] = other_coords[i] / vec->coords[i];
            }
            break;
        case OP_DIV | OP_ARG_NUMBER | OP_ARG_REVERSE:
            for (i = 0; i < vec->dim; i++) {
                if (vec->coords[i] == 0) {
                    PyErr_SetString(PyExc_ZeroDivisionError,
                                    "division by zero");
                    Py_DECREF(ret);
                    return NULL;
                }
                ret->coords[i] = other_value / vec->coords[i];
            }
            break;
        case OP_FLOOR_DIV | OP_ARG_VECTOR:
            for (i = 0; i < vec->dim; i++) {
                if (other_coords[i] == 0) {
                    PyErr_SetString(PyExc_ZeroDivisionError,
                                    "division by zero");
                    Py_DECREF(ret);
                    return NULL;
                }
                ret->coords[i] = floor(vec->coords[i] / other_coords[i]);
            }
            break;
        case OP_FLOOR_DIV | OP_ARG_VECTOR | OP_ARG_REVERSE:
            for (i = 0; i < vec->dim; i++) {
                if (vec->coords[i] == 0) {
                    PyErr_SetString(PyExc_ZeroDivisionError,
                                    "division by zero");
                    Py_DECREF(ret);
                    return NULL;
                }
                ret->coords[i] = floor(other_coords[i] / vec->coords[i]);
            }
            break;
        case OP_FLOOR_DIV | OP_ARG_NUMBER | OP_ARG_REVERSE:
            for (i = 0; i < vec->dim; i++) {
                if (vec->coords[i] == 0) {
                    PyErr_SetString(PyExc_ZeroDivisionError,
                                    "division by zero");
                    Py_DECREF(ret);
                    return NULL;
                }
                ret->coords[i] = floor(other_value / vec->coords[i]);
            }
            break;
        case OP_MOD | OP_ARG_VECTOR:
            for (i = 0; i < vec->dim; i++) {
                if (other_coords[i] == 0) {
                    PyErr_SetString(PyExc_ZeroDivisionError,
                                    "division by zero");
                    Py_DECREF(ret);
                    return NULL;
                }
                mod = fmod(vec->coords[i], other_coords[i]);
                /* note: checking mod*value < 0 is incorrect -- underflows
                   to 0 if value < sqrt(smallest nonzero double) */
                if (mod && ((other_coords[i] < 0) != (mod < 0))) {
                    mod += other_coords[i];
                }
                ret->coords[i] = mod;
            }
            break;
        case OP_MOD | OP_ARG_VECTOR | OP_ARG_REVERSE:
            for (i = 0; i < vec->dim; i++) {
                if (vec->coords[i] == 0) {
                    PyErr_SetString(PyExc_ZeroDivisionError,
                                    "division by zero");
                    Py_DECREF(ret);
                    return NULL;
                }
                mod = fmod(other_coords[i], vec->coords[i]);
                /* note: see above */
                if (mod && ((vec->coords[i] < 0) != (mod < 0))) {
                    mod += vec->coords[i];
                }
                ret->coords[i] = mod;
            }
            break;
        case OP_MOD | OP_ARG_NUMBER:
            if (other_value == 0) {
                PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
                Py_DECREF(ret);
                return NULL;
            }
            for (i = 0; i < vec->dim; i++) {
                mod = fmod(vec->coords[i], other_value);
                /* note: see above */
                if (mod && ((other_value < 0) != (mod < 0))) {
                    mod += other_value;
                }
                ret->coords[i] = mod;
            }
            break;
        case OP_MOD | OP_ARG_NUMBER | OP_ARG_REVERSE:
            for (i = 0; i < vec->dim; i++) {
                if (vec->coords[i] == 0) {
                    PyErr_SetString(PyExc_ZeroDivisionError,
                                    "division by zero");
                    Py_DECREF(ret);
                    return NULL;
                }
                mod = fmod(other_value, vec->coords[i]);
                /* note: see above */
                if (mod && ((vec->coords[i] < 0) != (mod < 0))) {
                    mod += vec->coords[i];
                }
                ret->coords[i] = mod;
            }
            break;
        default:
            Py_DECREF(ret);
            return vector_generic_math((PyObject *)vec, other, op);
    }
    return (PyObject *)ret;
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
vector_elementwiseproxy_pow(PyObject *baseObj, PyObject *expoObj,
                            PyObject *mod)
{
    Py_ssize_t i, dim;
    pgVector *vec;
    double *tmp;
    PyObject *bases[VECTOR_MAX_SIZE] = {NULL};
    PyObject *expos[VECTOR_MAX_SIZE] = {NULL};
    PyObject *ret, *result;
    if (mod != Py_None) {
        PyErr_SetString(PyExc_TypeError,
                        "pow() 3rd argument not "
                        "supported for vectors");
        return NULL;
    }

    if (vector_elementwiseproxy_Check(baseObj)) {
        vec = ((vector_elementwiseproxy *)baseObj)->vec;
        dim = vec->dim;
        tmp = vec->coords;
        for (i = 0; i < dim; ++i)
            bases[i] = PyFloat_FromDouble(tmp[i]);
        if (vector_elementwiseproxy_Check(expoObj)) {
            tmp = ((vector_elementwiseproxy *)expoObj)->vec->coords;
            for (i = 0; i < dim; ++i)
                expos[i] = PyFloat_FromDouble(tmp[i]);
        }
        else if (pgVectorCompatible_Check(expoObj, dim)) {
            for (i = 0; i < dim; ++i)
                expos[i] = PySequence_ITEM(expoObj, i);
        }
        else if (RealNumber_Check(expoObj)) {
            /* INCREF so that we can unify the clean up code */
            for (i = 0; i < dim; i++) {
                expos[i] = expoObj;
                Py_INCREF(expoObj);
            }
        }
        else {
            Py_INCREF(Py_NotImplemented);
            ret = Py_NotImplemented;
            goto clean_up;
        }
    }
    else {
        vec = ((vector_elementwiseproxy *)expoObj)->vec;
        dim = vec->dim;
        tmp = vec->coords;
        for (i = 0; i < dim; ++i)
            expos[i] = PyFloat_FromDouble(tmp[i]);
        if (pgVectorCompatible_Check(baseObj, dim)) {
            for (i = 0; i < dim; ++i)
                bases[i] = PySequence_ITEM(baseObj, i);
        }
        else if (RealNumber_Check(baseObj)) {
            /* INCREF so that we can unify the clean up code */
            for (i = 0; i < dim; i++) {
                bases[i] = baseObj;
                Py_INCREF(baseObj);
            }
        }
        else {
            Py_INCREF(Py_NotImplemented);
            ret = Py_NotImplemented;
            goto clean_up;
        }
    }
    if (PyErr_Occurred()) {
        ret = NULL;
        goto clean_up;
    }

    ret = (PyObject *)_vector_subtype_new(vec);
    if (ret == NULL)
        goto clean_up;
    /* there are many special cases so we let python do the work for us */
    for (i = 0; i < dim; i++) {
        result = PyNumber_Power(bases[i], expos[i], Py_None);
        if (result == NULL || !RealNumber_Check(result)) {
            /* raising a negative number to a fractional power returns a
             * complex in python-3.x. we do not allow this. */
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_ValueError,
                                "negative number "
                                "cannot be raised to a fractional power");
            }
            Py_XDECREF(result);
            Py_DECREF(ret);
            ret = NULL;
            goto clean_up;
        }
        ((pgVector *)ret)->coords[i] = PyFloat_AsDouble(result);
        Py_DECREF(result);
    }
clean_up:
    for (i = 0; i < dim; ++i) {
        Py_XDECREF(bases[i]);
        Py_XDECREF(expos[i]);
    }
    return ret;
}

static PyObject *
vector_elementwiseproxy_abs(vector_elementwiseproxy *self)
{
    pgVector *ret = _vector_subtype_new(self->vec);
    if (ret) {
        Py_ssize_t i;

        for (i = 0; i < self->vec->dim; i++) {
            ret->coords[i] = fabs(self->vec->coords[i]);
        }
    }
    return (PyObject *)ret;
}

static PyObject *
vector_elementwiseproxy_neg(vector_elementwiseproxy *self)
{
    return vector_neg(self->vec);
}

static PyObject *
vector_elementwiseproxy_pos(vector_elementwiseproxy *self)
{
    return vector_pos(self->vec);
}

static int
vector_elementwiseproxy_nonzero(vector_elementwiseproxy *self)
{
    return vector_nonzero(self->vec);
}

static PyNumberMethods vector_elementwiseproxy_as_number = {
    .nb_add = (binaryfunc)vector_elementwiseproxy_add,
    .nb_subtract = (binaryfunc)vector_elementwiseproxy_sub,
    .nb_multiply = (binaryfunc)vector_elementwiseproxy_mul,
    .nb_remainder = (binaryfunc)vector_elementwiseproxy_mod,
    .nb_power = (ternaryfunc)vector_elementwiseproxy_pow,
    .nb_negative = (unaryfunc)vector_elementwiseproxy_neg,
    .nb_positive = (unaryfunc)vector_elementwiseproxy_pos,
    .nb_absolute = (unaryfunc)vector_elementwiseproxy_abs,
    .nb_bool = (inquiry)vector_elementwiseproxy_nonzero,
    .nb_floor_divide = (binaryfunc)vector_elementwiseproxy_floor_div,
    .nb_true_divide = (binaryfunc)vector_elementwiseproxy_div,
};

static PyTypeObject pgVectorElementwiseProxy_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name =
        "pygame.math.VectorElementwiseProxy",
    .tp_basicsize = sizeof(vector_elementwiseproxy),
    .tp_dealloc = (destructor)vector_elementwiseproxy_dealloc,
    .tp_as_number = &vector_elementwiseproxy_as_number,
    /* Elementwise Proxy is not subtypable for now, no Py_TPFLAGS_BASETYPE */
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_richcompare = (richcmpfunc)vector_elementwiseproxy_richcompare,
};

static PyObject *
vector_elementwise(pgVector *vec, PyObject *_null)
{
    vector_elementwiseproxy *proxy;
    if (!pgVector_Check(vec)) {
        PyErr_BadInternalCall();
        return NULL;
    }

    proxy =
        PyObject_New(vector_elementwiseproxy, &pgVectorElementwiseProxy_Type);
    if (proxy == NULL)
        return NULL;
    Py_INCREF(vec);
    proxy->vec = (pgVector *)vec;

    return (PyObject *)proxy;
}

static PyObject *
math_clamp(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 3)
        return RAISE(PyExc_TypeError, "clamp requires 3 arguments");

    PyObject *value = args[0];
    PyObject *min = args[1];
    PyObject *max = args[2];

    if (PyNumber_Check(args[0]) != 1 || PyNumber_Check(args[1]) != 1 ||
        PyNumber_Check(args[2]) != 1) {
        return RAISE(PyExc_TypeError, "clamp requires 3 numeric arguments");
    }

    // Using RichCompare instead of converting to C types for performance
    // reasons. This implementation was tested to be faster than using
    // PyFloat_AsDouble and PyErr_Occurred.

    // if value < min: return min
    int result = PyObject_RichCompareBool(value, min, Py_LT);
    if (result == 1) {
        Py_INCREF(min);
        return min;
    }
    else if (result == -1)
        return NULL;

    // if value > max: return max
    result = PyObject_RichCompareBool(value, max, Py_GT);
    if (result == 1) {
        Py_INCREF(max);
        return max;
    }
    else if (result == -1)
        return NULL;

    Py_INCREF(value);
    return value;
}

PG_WRAP_FASTCALL_FUNC(math_clamp, PyObject);

static PyObject *
math_enable_swizzling(pgVector *self, PyObject *_null)
{
    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "pygame.math.enable_swizzling() is deprecated, "
                     "and its functionality is removed. This function will be "
                     "removed in a later version.",
                     1) == -1) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
math_disable_swizzling(pgVector *self, PyObject *_null)
{
    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "pygame.math.disable_swizzling() is deprecated, "
                     "and its functionality is removed. This function will be "
                     "removed in a later version.",
                     1) == -1) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef _math_methods[] = {
    {"clamp", (PyCFunction)PG_FASTCALL_NAME(math_clamp), PG_FASTCALL,
     DOC_PYGAMEMATHCLAMP},
    {"enable_swizzling", (PyCFunction)math_enable_swizzling, METH_NOARGS,
     "Deprecated, will be removed in a future version"},
    {"disable_swizzling", (PyCFunction)math_disable_swizzling, METH_NOARGS,
     "Deprecated, will be removed in a future version."},
    {NULL, NULL, 0, NULL}};

/****************************
 * Module init function
 ****************************/

#if defined(BUILD_STATIC)
// prevent name collision with CPython builtin PyInit_math from math module
MODINIT_DEFINE(pg_math)
#else
MODINIT_DEFINE(math)
#endif
{
    PyObject *module, *apiobj;
    static void *c_api[PYGAMEAPI_MATH_NUMSLOTS];

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "math",
                                         DOC_PYGAMEMATH,
                                         -1,
                                         _math_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* initialize the extension types */
    if ((PyType_Ready(&pgVector2_Type) < 0) ||
        (PyType_Ready(&pgVector3_Type) < 0) ||
        (PyType_Ready(&pgVectorIter_Type) < 0) ||
        (PyType_Ready(&pgVectorElementwiseProxy_Type) < 0) /*||
        (PyType_Ready(&pgVector4_Type) < 0)*/) {
        return NULL;
    }

    /* initialize the module */
    module = PyModule_Create(&_module);

    if (module == NULL) {
        return NULL;
    }

    /* add extension types to module */
    Py_INCREF(&pgVector2_Type);
    Py_INCREF(&pgVector3_Type);
    Py_INCREF(&pgVectorIter_Type);
    Py_INCREF(&pgVectorElementwiseProxy_Type);
    /*
    Py_INCREF(&pgVector4_Type);
    */
    if ((PyModule_AddObject(module, "Vector2", (PyObject *)&pgVector2_Type) !=
         0) ||
        (PyModule_AddObject(module, "Vector3", (PyObject *)&pgVector3_Type) !=
         0) ||
        (PyModule_AddObject(module, "VectorElementwiseProxy",
                            (PyObject *)&pgVectorElementwiseProxy_Type) !=
         0) ||
        (PyModule_AddObject(module, "VectorIterator",
                            (PyObject *)&pgVectorIter_Type) != 0) /*||
(PyModule_AddObject(module, "Vector4", (PyObject *)&pgVector4_Type) !=
0)*/) {
        if (!PyObject_HasAttrString(module, "Vector2"))
            Py_DECREF(&pgVector2_Type);
        if (!PyObject_HasAttrString(module, "Vector3"))
            Py_DECREF(&pgVector3_Type);
        if (!PyObject_HasAttrString(module, "VectorElementwiseProxy"))
            Py_DECREF(&pgVectorElementwiseProxy_Type);
        if (!PyObject_HasAttrString(module, "VectorIterator"))
            Py_DECREF(&pgVectorIter_Type);
        /*
        if (!PyObject_HasAttrString(module, "Vector4"))
            Py_DECREF(&pgVector4_Type);
        */
        Py_DECREF(module);
        return NULL;
    }

    /* export the C api */
    c_api[0] = &pgVector2_Type;
    c_api[1] = &pgVector3_Type;
    /*
    c_api[2] = &pgVector4_Type;
    c_api[3] = pgVector_NEW;
    c_api[4] = pgVectorCompatible_Check;
    */
    apiobj = encapsulate_api(c_api, "math");
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_XDECREF(apiobj);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
