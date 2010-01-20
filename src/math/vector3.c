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
#define PYGAME_MATHVECTOR3_INTERNAL

#include "pgbase.h"
#include "mathmod.h"
#include "pgmath.h"
#include "mathbase_doc.h"

static int _vector3_init (PyObject *self, PyObject *args, PyObject *kwds);
static PyObject* _vector3_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);

static PyObject* _vector3_get_x (PyObject *self, void *closure);
static int _vector3_set_x (PyObject *self, PyObject *value, void *closure);
static PyObject* _vector3_get_y (PyObject *self, void *closure);
static int _vector3_set_y (PyObject *self, PyObject *value, void *closure);
static PyObject* _vector3_get_z (PyObject *self, void *closure);
static int _vector3_set_z (PyObject *self, PyObject *value, void *closure);

static void _do_rotate (double *dst_coords, const double *src_coords,
    const double* axis_coords, double angle, double epsilon);

static PyObject* _vector3_rotate (PyObject *self, PyObject *args);
static PyObject* _vector3_rotate_ip (PyObject *self, PyObject *args);
static PyObject* _vector3_rotate_x (PyObject *self, PyObject *args);
static PyObject* _vector3_rotate_x_ip (PyObject *self, PyObject *args);
static PyObject* _vector3_rotate_y (PyObject *self, PyObject *args);
static PyObject* _vector3_rotate_y_ip (PyObject *self, PyObject *args);
static PyObject* _vector3_rotate_z (PyObject *self, PyObject *args);
static PyObject* _vector3_rotate_z_ip (PyObject *self, PyObject *args);
static PyObject* _vector3_cross (PyObject *self, PyObject *args);
static PyObject* _vector3_angleto (PyObject *self, PyObject *args);
static PyObject* _vector3_asspherical (PyObject *self);

/**
 * Methods for the PyVector3.
 */
static PyMethodDef _vector3_methods[] =
{
    { "rotate_x", _vector3_rotate_x, METH_O, DOC_BASE_VECTOR3_ROTATE_X },
    { "rotate_x_ip", _vector3_rotate_x_ip, METH_O,
      DOC_BASE_VECTOR3_ROTATE_X_IP },
    { "rotate_y", _vector3_rotate_y, METH_O, DOC_BASE_VECTOR3_ROTATE_Y },
    { "rotate_y_ip", _vector3_rotate_y_ip, METH_O,
      DOC_BASE_VECTOR3_ROTATE_Y_IP },
    { "rotate_z", _vector3_rotate_z, METH_O, DOC_BASE_VECTOR3_ROTATE_Z },
    { "rotate_z_ip", _vector3_rotate_z_ip, METH_O,
      DOC_BASE_VECTOR3_ROTATE_Z_IP },
    { "rotate", _vector3_rotate, METH_VARARGS, DOC_BASE_VECTOR3_ROTATE },
    { "rotate_ip", _vector3_rotate_ip, METH_VARARGS,
      DOC_BASE_VECTOR3_ROTATE_IP },
    { "cross", _vector3_cross, METH_O, DOC_BASE_VECTOR3_CROSS },
    { "angle_to", _vector3_angleto, METH_O, DOC_BASE_VECTOR3_ANGLE_TO },
    { "as_spherical", (PyCFunction)_vector3_asspherical, METH_NOARGS,
      DOC_BASE_VECTOR3_AS_SPHERICAL },
    { NULL, NULL, 0, NULL },
};

/**
 * Getters and setters for the PyVector3.
 */
static PyGetSetDef _vector3_getsets[] =
{
    { "x", _vector3_get_x, _vector3_set_x, DOC_BASE_VECTOR3_X, NULL },
    { "y", _vector3_get_y, _vector3_set_y, DOC_BASE_VECTOR3_Y, NULL },
    { "z", _vector3_get_z, _vector3_set_z, DOC_BASE_VECTOR3_Z, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

PyTypeObject PyVector3_Type =
{
    TYPE_HEAD(NULL,0)
    "base.Vector3",             /* tp_name */
    sizeof (PyVector3),         /* tp_basicsize */
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
    DOC_BASE_VECTOR3,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _vector3_methods,           /* tp_methods */
    0,                          /* tp_members */
    _vector3_getsets,           /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)_vector3_init,    /* tp_init */
    0,                          /* tp_alloc */
    _vector3_new,               /* tp_new */
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
_vector3_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyVector3 *vector = (PyVector3 *)type->tp_alloc (type, 0);
    if (!vector)
        return NULL;
    vector->vector.dim = 3;
    vector->vector.coords = PyMem_New (double, 3);
    if (!vector->vector.coords)
    {
        Py_DECREF ((PyObject*)vector);
        return NULL;
    }
    vector->vector.coords[0] = vector->vector.coords[1] =
        vector->vector.coords[2] = 0.f;
    vector->vector.epsilon = VEC_EPSILON;
    return (PyObject *) vector;
}

static int
_vector3_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyVector3 *vector = (PyVector3 *) self;
    double x = 0.f, y = 0.f, z = 0.f;

    if (!PyArg_ParseTuple (args, "|ddd", &x, &y, &z))
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
        if (dim < 3)
        {
            PyErr_SetString (PyExc_ValueError,
                "sequence or vector must have at least three dimensions");
            PyMem_Free (coords);
            return -1;
        }
        x = coords[0];
        y = coords[1];
        z = coords[2];
        PyMem_Free (coords);
    }
    vector->vector.coords[0] = x;
    vector->vector.coords[1] = y;
    vector->vector.coords[2] = z;
    return 0;
}

/* Vector getters/setters */

/**
 * x = Vector3.x
 */
static PyObject*
_vector3_get_x (PyObject *self, void *closure)
{
    return PyFloat_FromDouble (((PyVector3 *)self)->vector.coords[0]);
}

/**
 * Vector3.x = x
 */
static int
_vector3_set_x (PyObject *self, PyObject *value, void *closure)
{
    double tmp;
    if (!DoubleFromObj (value, &tmp))
        return -1;
    ((PyVector3 *)self)->vector.coords[0] = tmp;
    return 0;
}

/**
 * y = Vector3.y
 */
static PyObject*
_vector3_get_y (PyObject *self, void *closure)
{
    return PyFloat_FromDouble (((PyVector3 *)self)->vector.coords[1]);
}

/**
 * Vector3.y = y
 */
static int
_vector3_set_y (PyObject *self, PyObject *value, void *closure)
{
    double tmp;
    if (!DoubleFromObj (value, &tmp))
        return -1;
    ((PyVector3 *)self)->vector.coords[1] = tmp;
    return 0;
}

/**
 * z = Vector3.z
 */
static PyObject*
_vector3_get_z (PyObject *self, void *closure)
{
    return PyFloat_FromDouble (((PyVector3 *)self)->vector.coords[2]);
}

/**
 * Vector3.z = z
 */
static int
_vector3_set_z (PyObject *self, PyObject *value, void *closure)
{
    double tmp;
    if (!DoubleFromObj (value, &tmp))
        return -1;
    ((PyVector3 *)self)->vector.coords[2] = tmp;
    return 0;
}

/* Vector3 methods */
static void
_do_rotate (double *dst_coords, const double *src_coords,
    const double* axis_coords, double angle, double epsilon)
{
    double sinv, cosv, coscompl;
    double nfactor;
    double axislen = 0;
    double axis[3];
    Py_ssize_t i;

    /* make sure angle is in range [0, 360) */
    angle = fmod (angle, 360.);
    if (angle < 0)
        angle += 360.;

    for (i = 0; i < 3; ++i)
    {
        axislen += axis_coords[i] * axis_coords[i];
        axis[i] = axis_coords[i];
    }

    /* normalize the axis */
    if (axislen - 1 > epsilon)
    {
        nfactor = 1. / sqrt (axislen);
        for (i = 0; i < 3; ++i)
            axis[i] *= nfactor;
    }

    /* special-case rotation by 0, 90, 180 and 270 degrees */
    if (fmod (angle + epsilon, 90.) < 2 * epsilon)
    {
        switch ((int)((angle + epsilon) / 90))
        {
        case 0: /* 0 degrees */
            memcpy (dst_coords, src_coords, 3 * sizeof(double));
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
    else
    {
        angle = DEG2RAD (angle);
        sinv = sin (angle);
        cosv = cos (angle);
        coscompl = 1 - cosv;

        dst_coords[0] =
            (src_coords[0] * (cosv + axis[0] * axis[0] * coscompl) +
             src_coords[1] * (axis[0] * axis[1] * coscompl - axis[2] * sinv) +
             src_coords[2] * (axis[0] * axis[2] * coscompl + axis[1] * sinv));
        dst_coords[1] =
            (src_coords[0] * (axis[0] * axis[1] * coscompl + axis[2] * sinv) +
             src_coords[1] * (cosv + axis[1] * axis[1] * coscompl) +
             src_coords[2] * (axis[1] * axis[2] * coscompl - axis[0] * sinv));
        dst_coords[2] =
            (src_coords[0] * (axis[0] * axis[2] * coscompl - axis[1] * sinv) +
             src_coords[1] * (axis[1] * axis[2] * coscompl + axis[0] * sinv) +
             src_coords[2] * (cosv + axis[2] * axis[2] * coscompl));
    }
}

static PyObject*
_vector3_rotate (PyObject *self, PyObject *args)
{
    PyVector *v = (PyVector *) self;
    PyVector *ret;
    PyObject *axis;
    Py_ssize_t axisdim;
    double *axiscoords, angle;
    
    if (!PyArg_ParseTuple (args, "dO:rotate", &angle, &axis))
        return NULL;

    if (!IsVectorCompatible (axis))
    {
        PyErr_SetString (PyExc_TypeError, "axis must be a vector compatible");
        return NULL;
    }

    axiscoords = VectorCoordsFromObj (axis, &axisdim);
    if (!axiscoords)
        return NULL;
    if (axisdim < v->dim)
    {
        PyErr_SetString (PyExc_ValueError,
            "axis must have at least three dimensions");
        PyMem_Free (axiscoords);
        return NULL;
    }
    ret = (PyVector*) PyVector3_New (0.f, 0.f, 0.f);
    _do_rotate (ret->coords, v->coords, axiscoords, angle, v->epsilon);
    PyMem_Free (axiscoords);
    return (PyObject*) ret;
}

static PyObject*
_vector3_rotate_ip (PyObject *self, PyObject *args)
{
    PyVector *v = (PyVector *) self;
    PyObject *axis;
    Py_ssize_t axisdim;
    double *axiscoords, angle, tmp[3];
    
    if (!PyArg_ParseTuple (args, "dO:rotate_ip", &angle, &axis))
        return NULL;

    if (!IsVectorCompatible (axis))
    {
        PyErr_SetString (PyExc_TypeError, "axis must be a vector compatible");
        return NULL;
    }

    axiscoords = VectorCoordsFromObj (axis, &axisdim);
    if (!axiscoords)
        return NULL;
    if (axisdim < v->dim)
    {
        PyErr_SetString (PyExc_ValueError,
            "axis must have at least three dimensions");
        PyMem_Free (axiscoords);
        return NULL;
    }

    memcpy (tmp, v->coords, sizeof (double) * 3);
    _do_rotate (v->coords, tmp, axiscoords, angle, v->epsilon);
    PyMem_Free (axiscoords);
    Py_RETURN_NONE;
}

static PyObject*
_vector3_rotate_x (PyObject *self, PyObject *args)
{
    PyVector *v = (PyVector *) self;
    PyVector *ret;
    double sinvalue, cosvalue;
    double angle;

    if (!DoubleFromObj (args, &angle))
        return NULL;
    
    ret = (PyVector*) PyVector3_New (0.f, 0.f, 0.f);
    if (!ret)
        return NULL;

    angle = DEG2RAD (angle);
    sinvalue = sin (angle);
    cosvalue = cos (angle);

    ret->coords[0] = v->coords[0];
    ret->coords[1] = v->coords[1] * cosvalue - v->coords[2] * sinvalue;
    ret->coords[2] = v->coords[1] * sinvalue + v->coords[2] * cosvalue;
    return (PyObject*) ret;
}

static PyObject*
_vector3_rotate_x_ip (PyObject *self, PyObject *args)
{
    PyVector *v = (PyVector *) self;
    double tmpcoords[3];
    double sinvalue, cosvalue;
    double angle;

    if (!DoubleFromObj (args, &angle))
        return NULL;

    angle = DEG2RAD (angle);
    sinvalue = sin (angle);
    cosvalue = cos (angle);
    memcpy(tmpcoords, v->coords, sizeof(double) * 3);

    v->coords[1] = tmpcoords[1] * cosvalue - tmpcoords[2] * sinvalue;
    v->coords[2] = tmpcoords[1] * sinvalue + tmpcoords[2] * cosvalue;
    Py_RETURN_NONE;
}

static PyObject*
_vector3_rotate_y (PyObject *self, PyObject *args)
{
    PyVector *v = (PyVector *) self;
    PyVector *ret;
    double sinvalue, cosvalue;
    double angle;

    if (!DoubleFromObj (args, &angle))
        return NULL;

    ret = (PyVector*) PyVector3_New (0.f, 0.f, 0.f);
    if (!ret)
        return NULL;

    angle = DEG2RAD (angle);
    sinvalue = sin (angle);
    cosvalue = cos (angle);

    ret->coords[0] = v->coords[0] * cosvalue + v->coords[2] * sinvalue;
    ret->coords[1] = v->coords[1];
    ret->coords[2] = - v->coords[0] * sinvalue + v->coords[2] * cosvalue;
    return (PyObject*) ret;
}

static PyObject*
_vector3_rotate_y_ip (PyObject *self, PyObject *args)
{
    PyVector *v = (PyVector *) self;
    double tmpcoords[3];
    double sinvalue, cosvalue;
    double angle;

    if (!DoubleFromObj (args, &angle))
        return NULL;

    angle = DEG2RAD (angle);
    sinvalue = sin (angle);
    cosvalue = cos (angle);
    memcpy(tmpcoords, v->coords, sizeof(double) * 3);

    v->coords[0] = tmpcoords[0] * cosvalue + tmpcoords[2] * sinvalue;
    v->coords[2] = -tmpcoords[0] * sinvalue + tmpcoords[2] * cosvalue;
    Py_RETURN_NONE;
}

static PyObject*
_vector3_rotate_z (PyObject *self, PyObject *args)
{
    PyVector *v = (PyVector *) self;
    PyVector *ret;
    double sinvalue, cosvalue;
    double angle;

    if (!DoubleFromObj (args, &angle))
        return NULL;

    ret = (PyVector*) PyVector3_New (0.f, 0.f, 0.f);
    if (!ret)
        return NULL;

    angle = DEG2RAD (angle);
    sinvalue = sin (angle);
    cosvalue = cos (angle);

    ret->coords[0] = v->coords[0] * cosvalue - v->coords[1] * sinvalue;
    ret->coords[1] = v->coords[0] * sinvalue + v->coords[1] * cosvalue;
    ret->coords[2] = v->coords[2];
    return (PyObject*) ret;
}

static PyObject*
_vector3_rotate_z_ip (PyObject *self, PyObject *args)
{
    PyVector *v = (PyVector *) self;
    double tmpcoords[3];
    double sinvalue, cosvalue;
    double angle;

    if (!DoubleFromObj (args, &angle))
        return NULL;

    angle = DEG2RAD (angle);
    sinvalue = sin (angle);
    cosvalue = cos (angle);
    memcpy(tmpcoords, v->coords, sizeof(double) * 3);

    v->coords[0] = tmpcoords[0] * cosvalue - tmpcoords[1] * sinvalue;
    v->coords[1] = tmpcoords[0] * sinvalue + tmpcoords[1] * cosvalue;
    Py_RETURN_NONE;
}

static PyObject*
_vector3_cross (PyObject *self, PyObject *args)
{
    Py_ssize_t otherdim;
    double *othercoords;
    PyVector *v = (PyVector *) self;
    PyVector *ret;

    if (!IsVectorCompatible (args))
    {
        PyErr_SetString (PyExc_TypeError, "other must be a vector compatible");
        return NULL;
    }

    othercoords = VectorCoordsFromObj (args, &otherdim);
    if (!othercoords)
        return NULL;
    if (otherdim < v->dim)
    {
        PyErr_SetString (PyExc_ValueError,
            "other must have at least three dimensions");
        PyMem_Free (othercoords);
        return NULL;
    }

    ret = (PyVector*) PyVector3_New (0.f, 0.f, 0.f);
    if (!ret)
    {
        PyMem_Free (othercoords);
        return NULL;
    }

    ret->coords[0] = (v->coords[1] * othercoords[2]) -
        (v->coords[2] * othercoords[1]);
    ret->coords[1] = (v->coords[2] * othercoords[0]) -
        (v->coords[0] * othercoords[2]);
    ret->coords[2] = (v->coords[0] * othercoords[1]) -
        (v->coords[1] * othercoords[0]);
    return (PyObject*) ret;
}

static PyObject*
_vector3_angleto (PyObject *self, PyObject *args)
{
    double angle, tmp1, tmp2;
    double *othercoords;
    Py_ssize_t otherdim;
    PyVector* v = (PyVector*)self;

    if (!IsVectorCompatible (args))
    {
        PyErr_SetString (PyExc_TypeError, "other must be a vector compatible");
        return NULL;
    }

    othercoords = VectorCoordsFromObj (args, &otherdim);
    if (!othercoords)
        return NULL;
    if (otherdim < v->dim)
    {
        PyErr_SetString (PyExc_ValueError,
            "other must have at least three dimensions");
        PyMem_Free (othercoords);
        return NULL;
    }
    
    tmp1 = _ScalarProduct (v->coords, v->coords, v->dim);
    tmp2 = _ScalarProduct (othercoords, othercoords, otherdim);
    angle = acos (_ScalarProduct (v->coords, othercoords, v->dim) /
        sqrt (tmp1 * tmp2));
    PyMem_Free (othercoords);

    return PyFloat_FromDouble (RAD2DEG (angle));
}

static PyObject*
_vector3_asspherical (PyObject *self)
{
    PyVector* v = (PyVector*)self;
    double r, theta, phi;
    r = sqrt (_ScalarProduct (v->coords, v->coords, v->dim));

    theta = acos (v->coords[2] / r);
    phi = atan2 (v->coords[1], v->coords[0]);
    return Py_BuildValue ("(ddd)", r, theta, phi);
}


/* C API */
PyObject*
PyVector3_New (double x, double y, double z)
{
    PyVector3 *v = (PyVector3*) PyVector3_Type.tp_new
        (&PyVector3_Type, NULL, NULL);
    if (!v)
        return NULL;
    v->vector.coords[0] = x;
    v->vector.coords[1] = y;
    v->vector.coords[2] = z;
    return (PyObject *) v;
}

void
vector3_export_capi (void **capi)
{
    capi[PYGAME_MATHVECTOR3_FIRSTSLOT] = &PyVector3_Type;
    capi[PYGAME_MATHVECTOR3_FIRSTSLOT+1] = &PyVector3_New;
}
