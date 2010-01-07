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

/**
 * Methods for the PyVector3.
 */
static PyMethodDef _vector3_methods[] = {
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
    vector->vector.epsilon = DBL_EPSILON;
    return (PyObject *) vector;
}

static int
_vector3_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyVector3 *vector = (PyVector3 *) self;
    double x, y, z;

    if (!PyArg_ParseTuple (args, "ddd", &x, &y, &z))
        return -1;
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
