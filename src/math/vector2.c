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

/**
 * Methods for the PyVector2.
 */
static PyMethodDef _vector2_methods[] = {
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
    vector->vector.epsilon = DBL_EPSILON;
    return (PyObject *) vector;
}

static int
_vector2_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyVector2 *vector = (PyVector2 *) self;
    double x, y;

    if (!PyArg_ParseTuple (args, "dd", &x, &y))
        return -1;
    vector->vector.coords[0] = x;
    vector->vector.coords[1] = y;
    return 0;
}

/* Vector getters/setters */

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
