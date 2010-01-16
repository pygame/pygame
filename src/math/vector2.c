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
#define PYGAME_MATHVECTOR2_INTERNAL

#include "pgbase.h"
#include "mathmod.h"
#include "pgmath.h"
#include "mathbase_doc.h"

static int _vector2_init (PyObject *self, PyObject *args, PyObject *kwds);
static PyObject* _vector2_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);

static PyObject* _vector2_get_x (PyObject *self, void *closure);
static int _vector2_set_x (PyObject *self, PyObject *value, void *closure);
static PyObject* _vector2_get_y (PyObject *self, void *closure);
static int _vector2_set_y (PyObject *self, PyObject *value, void *closure);

static void _do_rotate (double *dst_coords, const double *src_coords,
    double angle, double epsilon);

static PyObject* _vector2_rotate (PyObject *self, PyObject *args);
static PyObject* _vector2_rotate_ip (PyObject *self, PyObject *args);
static PyObject* _vector2_cross (PyObject *self, PyObject *args);
static PyObject* _vector2_angleto (PyObject *self, PyObject *args);
static PyObject* _vector2_aspolar (PyVector *self);

/**
 * Methods for the PyVector2.
 */
static PyMethodDef _vector2_methods[] =
{
    { "rotate", _vector2_rotate, METH_VARARGS, DOC_BASE_VECTOR2_ROTATE },
    { "rotate_ip", _vector2_rotate_ip, METH_VARARGS,
      DOC_BASE_VECTOR2_ROTATE_IP },
    { "cross", _vector2_cross, METH_VARARGS, DOC_BASE_VECTOR2_CROSS },
    { "angle_to", _vector2_angleto, METH_VARARGS, DOC_BASE_VECTOR2_ANGLE_TO },
    { "as_polar", (PyCFunction)_vector2_aspolar, METH_NOARGS,
      DOC_BASE_VECTOR2_AS_POLAR },
    { NULL, NULL, 0, NULL },
};

/**
 * Getters and setters for the PyVector2.
 */
static PyGetSetDef _vector2_getsets[] =
{
    { "x", _vector2_get_x, _vector2_set_x, DOC_BASE_VECTOR2_X, NULL },
    { "y", _vector2_get_y, _vector2_set_y, DOC_BASE_VECTOR2_Y, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

PyTypeObject PyVector2_Type =
{
    TYPE_HEAD(NULL,0)
    "base.Vector2",             /* tp_name */
    sizeof (PyVector2),         /* tp_basicsize */
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
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    DOC_BASE_VECTOR2,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _vector2_methods,           /* tp_methods */
    0,                          /* tp_members */
    _vector2_getsets,           /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)_vector2_init,    /* tp_init */
    0,                          /* tp_alloc */
    _vector2_new,               /* tp_new */
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

static PyObject*
_vector2_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyVector2 *vector = (PyVector2 *)type->tp_alloc (type, 0);
    if (!vector)
        return NULL;
    vector->vector.dim = 2;
    vector->vector.coords = PyMem_New (double, 2);
    if (!vector->vector.coords)
    {
        Py_DECREF ((PyObject*)vector);
        return NULL;
    }
    vector->vector.coords[0] = vector->vector.coords[1] = 0.f;
    vector->vector.epsilon = VEC_EPSILON;
    return (PyObject *) vector;
}

static int
_vector2_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyVector2 *vector = (PyVector2 *) self;
    double x = 0.f, y = 0.f;

    if (!PyArg_ParseTuple (args, "|dd", &x, &y))
    {
        PyObject *seq;
        Py_ssize_t dim;
        double *coords;

        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O", &seq))
            return -1;
        coords = VectorCoordsFromObj (seq, &dim);
        if (!coords)
            return -1;
        x = coords[0];
        y = coords[1];
        PyMem_Free (coords);
    }

    vector->vector.coords[0] = x;
    vector->vector.coords[1] = y;
    return 0;
}

/* Vector2 getters/setters */

/**
 * x = Vector2.x
 */
static PyObject*
_vector2_get_x (PyObject *self, void *closure)
{
    return PyFloat_FromDouble (((PyVector2 *)self)->vector.coords[0]);
}

/**
 * Vector2.x = x
 */
static int
_vector2_set_x (PyObject *self, PyObject *value, void *closure)
{
    double tmp;
    if (!DoubleFromObj (value, &tmp))
        return -1;
    ((PyVector2 *)self)->vector.coords[0] = tmp;
    return 0;
}

/**
 * y = Vector2.y
 */
static PyObject*
_vector2_get_y (PyObject *self, void *closure)
{
    return PyFloat_FromDouble (((PyVector2 *)self)->vector.coords[1]);
}

/**
 * Vector2.y = y
 */
static int
_vector2_set_y (PyObject *self, PyObject *value, void *closure)
{
    double tmp;
    if (!DoubleFromObj (value, &tmp))
        return -1;
    ((PyVector2 *)self)->vector.coords[1] = tmp;
    return 0;
}

/* Vector2 methods */

static void
_do_rotate (double *dst_coords, const double *src_coords, double angle,
    double epsilon)
{
    /* make sure angle is in range [0, 360) */
    angle = fmod (angle, 360.);
    if (angle < 0)
        angle += 360.;

    /* special-case rotation by 0, 90, 180 and 270 degrees */
    if (fmod (angle + epsilon, 90.) < 2 * epsilon)
    {
        switch ((int)((angle + epsilon) / 90))
        {
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
            PyErr_SetString (PyExc_RuntimeError,
               "Please report this bug in vector2_do_rotate to the developers");
            break;
        }
    }
    else
    {
        double sinv, cosv;

        angle = DEG2RAD (angle);
        sinv = sin (angle);
        cosv = cos (angle);

        dst_coords[0] = cosv * src_coords[0] - sinv * src_coords[1];
        dst_coords[1] = sinv * src_coords[0] + cosv * src_coords[1];
    }
}

static PyObject*
_vector2_rotate (PyObject *self, PyObject *args)
{
    double angle;
    PyVector *v = (PyVector *) self;
    PyVector *ret;

    if (!PyArg_ParseTuple (args, "d:rotate", &angle))
        return NULL;

    ret = (PyVector*) PyVector2_New (0., 0.);
    if (!ret)
        return NULL;
    _do_rotate (ret->coords, v->coords, angle, v->epsilon);
    return (PyObject*)ret;
}

static PyObject*
_vector2_rotate_ip (PyObject *self, PyObject *args)
{
    double angle, tmp[2];
    PyVector *v = (PyVector *) self;

    if (!PyArg_ParseTuple (args, "d:rotate_ip", &angle))
        return NULL;

    tmp[0] = v->coords[0];
    tmp[1] = v->coords[1];
    _do_rotate (v->coords, tmp, angle, v->epsilon);
    Py_RETURN_NONE;
}

static PyObject*
_vector2_cross (PyObject *self, PyObject *args)
{
    PyObject *other;
    PyVector *v = (PyVector*) self;
    double retval, *othercoords;
    Py_ssize_t otherdim;

    if (!PyArg_ParseTuple (args, "O:cross", &other))
        return NULL;

    if (!IsVectorCompatible (other))
    {
        PyErr_SetString (PyExc_TypeError, "other must be a vector compatible");
        return NULL;
    }

    othercoords = VectorCoordsFromObj (other, &otherdim);
    if (!othercoords)
        return NULL;
    retval = v->coords[0] * othercoords[1] - v->coords[1] * othercoords[0];
    PyMem_Free (othercoords);
    return PyFloat_FromDouble (retval);
}

static PyObject*
_vector2_angleto (PyObject *self, PyObject *args)
{
    double angle;
    PyObject *other;
    PyVector *v = (PyVector*) self;
    double *othercoords;
    Py_ssize_t otherdim;

    if (!PyArg_ParseTuple (args, "O:angleto", &other))
        return NULL;

    if (!IsVectorCompatible (other))
    {
        PyErr_SetString (PyExc_TypeError, "other must be a vector compatible");
        return NULL;
    }

    othercoords = VectorCoordsFromObj (other, &otherdim);
    if (!othercoords)
        return NULL;
    angle = atan2 (othercoords[1], othercoords[0]) -
        atan2 (v->coords[1], v->coords[0]);
    PyMem_Free (othercoords);
    return PyFloat_FromDouble (RAD2DEG (angle));
}


static PyObject *
_vector2_aspolar (PyVector *self)
{
    double r, phi;
    r = sqrt(_ScalarProduct (self->coords, self->coords, self->dim));
    phi = atan2 (self->coords[1], self->coords[0]);
    return Py_BuildValue ("(dd)", r, phi);
}

/* C API */
PyObject*
PyVector2_New (double x, double y)
{
    PyVector2 *v = (PyVector2*) PyVector2_Type.tp_new
        (&PyVector2_Type, NULL, NULL);
    if (!v)
        return NULL;
    v->vector.coords[0] = x;
    v->vector.coords[1] = y;
    return (PyObject *) v;
}

void
vector2_export_capi (void **capi)
{
    capi[PYGAME_MATHVECTOR2_FIRSTSLOT] = &PyVector2_Type;
    capi[PYGAME_MATHVECTOR2_FIRSTSLOT+1] = &PyVector2_New;
}
