/*
  pygame physics - Pygame physics module

  Copyright (C) 2008 Zhang Fan

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

#define PHYSICS_BODY_INTERNAL

#include "physicsmod.h"
#include "pgphysics.h"

static void _body_init_values (PyBody *body);
static void _body_dealloc (PyBody *body);
static PyObject* _body_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _body_init (PyBody *body, PyObject *args, PyObject *kwds);

static PyObject* _body_getdict (PyBody *body, void *closure);
static PyObject* _body_getmass (PyBody *body, void *closure);
static int _body_setmass (PyBody *body, PyObject *value, void *closure);
static PyObject* _body_getshape (PyBody *body, void *closure);
static PyObject* _body_getrotation (PyBody *body, void *closure);
static int _body_setrotation (PyBody *body, PyObject *value, void *closure);
static PyObject* _body_gettorque (PyBody *body, void *closure);
static int _body_settorque (PyBody *body, PyObject *value, void *closure);
static PyObject* _body_getrestitution (PyBody *body, void *closure);
static int _body_setrestitution (PyBody *body, PyObject *value, void *closure);
static PyObject* _body_getfriction (PyBody *body, void *closure);
static int _body_setfriction (PyBody *body, PyObject *value, void *closure);
static PyObject* _body_getlinveldamping (PyBody *body, void *closure);
static int _body_setlinveldamping (PyBody *body, PyObject *value,
    void *closure);
static PyObject* _body_getangleveldamping (PyBody *body, void *closure);
static int _body_setangleveldamping (PyBody *body, PyObject *value,
    void *closure);
static PyObject* _body_getvelocity (PyBody *body, void *closure);
static int _body_setvelocity (PyBody *body, PyObject *value, void *closure);
static PyObject* _body_getanglevelocity (PyBody *body, void *closure);
static int _body_setanglevelocity (PyBody *body, PyObject *value,
    void *closure);
static PyObject* _body_getposition (PyBody *body, void *closure);
static int _body_setposition (PyBody *body, PyObject *value, void *closure);
static PyObject* _body_getforce (PyBody *body, void *closure);
static int _body_setforce (PyBody *body, PyObject *value, void *closure);
static PyObject* _body_getstatic (PyBody *body, void *closure);
static int _body_setstatic (PyBody *body, PyObject *value, void *closure);

static PyObject *_body_getpoints (PyObject *self, PyObject *args);

/**
 * Methods, which are bound to the PyBody type.
 */
static PyMethodDef _body_methods[] =
{
    { "get_points", _body_getpoints, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
};

/**
 * Getters/Setters
 */
static PyGetSetDef _body_getsets[] =
{
    { "__dict__", (getter) _body_getdict, NULL, NULL, NULL },
    { "mass", (getter) _body_getmass, (setter) _body_setmass, NULL, NULL },
    { "shape", (getter) _body_getshape, NULL, NULL, NULL },
    { "rotation", (getter) _body_getrotation, (setter) _body_setrotation,
      NULL, NULL },
    { "torque", (getter) _body_gettorque, (setter) _body_settorque, NULL,
      NULL },
    { "restitution", (getter) _body_getrestitution,
      (setter) _body_setrestitution, NULL, NULL },
    { "friction", (getter) _body_getfriction, (setter) _body_setfriction, NULL,
      NULL },
    { "lin_vel_damping", (getter) _body_getlinveldamping,
      (setter) _body_setlinveldamping, NULL, NULL },
    { "angle_vel_damping", (getter) _body_getangleveldamping,
      (setter) _body_setangleveldamping, NULL, NULL },
    { "velocity", (getter) _body_getvelocity, (setter) _body_setvelocity, NULL
      , NULL },
    { "angular_velocity", (getter) _body_getanglevelocity,
      (setter) _body_setanglevelocity, NULL, NULL },
    { "position", (getter) _body_getposition, (setter) _body_setposition, NULL,
      NULL },
    { "force", (getter) _body_getforce, (setter) _body_setforce, NULL, NULL },
    { "static", (getter) _body_getstatic, (setter) _body_setstatic, NULL,
      NULL },
    { NULL, NULL, NULL, NULL, NULL }
};


PyTypeObject PyBody_Type =
{
    TYPE_HEAD(NULL, 0)
    "physics.Body",             /* tp_name */
    sizeof (PyBody),            /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _body_dealloc, /* tp_dealloc */
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
    "",
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _body_methods,              /* tp_methods */
    0,                          /* tp_members */
    _body_getsets,              /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    offsetof (PyBody, dict),    /* tp_dictoffset */
    (initproc)_body_init,       /* tp_init */
    0,                          /* tp_alloc */
    _body_new,                  /* tp_new */
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

static void
_body_init_values (PyBody *body)
{
    body->angle_velocity = 0.0;
    body->friction = 0.0;
    body->mass = 1.0;
    body->restitution = 1.0;
    body->rotation = 0.0;
    body->torque = 0.0;
    body->shape = NULL;
    body->isstatic = 0;
    body->linear_vel_damping = 0.0;
    body->angle_vel_damping = 0.06;
    PyVector2_Set (body->force, 0.0, 0.0);
    PyVector2_Set (body->impulse, 0.0, 0.0);
    PyVector2_Set (body->linear_velocity, 0.0, 0.0);
    PyVector2_Set (body->position, 0.0, 0.0);
}

static void
_body_dealloc (PyBody *body)
{
    Py_XDECREF (body->shape);
    ((PyObject*)body)->ob_type->tp_free ((PyObject*)body);
}

static PyObject*
_body_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyBody *body = (PyBody*) type->tp_alloc (type, 0);
    if (!body)
        return NULL;
    _body_init_values (body);
    return (PyObject*) body;
}

static int
_body_init (PyBody *body, PyObject *args, PyObject *kwds)
{
    PyObject *shape;

    if (!PyArg_ParseTuple (args, "O", &shape))
        return -1;

    if (!PyShape_Check (shape))
    {
        PyErr_SetString (PyExc_TypeError, "shape must be a Shape");
        return -1;
    }

    Py_INCREF (shape);
    body->shape = shape;
    if (!PyShape_Update_FAST ((PyShape*) body->shape, body))
    {
        /* An error occured. */
        Py_DECREF (shape);
        return -1;
    }
    return 0;
}

/* Getters/Setters */
static PyObject*
_body_getdict (PyBody *body, void *closure)
{
    if (!body->dict)
    {
        body->dict = PyDict_New ();
        if (!body->dict)
            return NULL;
    }
    Py_INCREF (body->dict);
    return body->dict;
}

static PyObject*
_body_getmass (PyBody *body, void *closure)
{
    return PyFloat_FromDouble (body->mass);
}

static int
_body_setmass (PyBody *body, PyObject *value, void *closure)
{
    double mass;

    if (!DoubleFromObj (value, &mass))
        return -1;
    if (mass <= 0)
    {
        PyErr_SetString (PyExc_ValueError, "mass must not be smaller than 0");
        return -1;
    }
    body->mass = mass;

    if (body->shape)
    {
        if (!PyShape_Update_FAST ((PyShape*) body->shape, body))
            return -1;
    }
    return 0;
}

static PyObject*
_body_getshape (PyBody *body, void *closure)
{
    if (!body->shape)
        Py_RETURN_NONE;
    Py_INCREF (body->shape);
    return body->shape;
}

static PyObject*
_body_getrotation (PyBody *body, void *closure)
{
    return PyFloat_FromDouble (RAD2DEG (body->rotation));
}

static int
_body_setrotation (PyBody *body, PyObject *value, void *closure)
{
    double rotation;
    if (!DoubleFromObj (value, &rotation))
        return -1;
    body->rotation = DEG2RAD (rotation);
    return 0;
}

static PyObject*
_body_gettorque (PyBody *body, void *closure)
{
    return PyFloat_FromDouble (body->torque);
}

static int
_body_settorque (PyBody *body, PyObject *value, void *closure)
{
    double torque;
    if (!DoubleFromObj (value, &torque))
        return -1;
    body->torque = torque;
    return 0;
}

static PyObject*
_body_getrestitution (PyBody *body, void *closure)
{
    return PyFloat_FromDouble (body->restitution);
}

static int
_body_setrestitution (PyBody *body, PyObject *value, void *closure)
{
    double restitution;
    if (!DoubleFromObj (value, &restitution))
        return -1;
    if (restitution < 0 || restitution > 1)
    {
        PyErr_SetString(PyExc_ValueError,
            "restitution must be in the range [0,1]");
        return -1;
    }
    body->restitution = restitution;
    return 0;
}

static PyObject*
_body_getfriction (PyBody *body, void *closure)
{
    return PyFloat_FromDouble (body->friction);
}

static int
_body_setfriction (PyBody *body, PyObject *value, void *closure)
{
    double friction;
    if (!DoubleFromObj (value, &friction))
        return -1;
    if (friction < 0)
    {
        PyErr_SetString (PyExc_ValueError, "friction must not be negative");
        return -1;
    }
    body->friction = friction;
    return 0;
}

static PyObject*
_body_getlinveldamping (PyBody *body, void *closure)
{
    return PyFloat_FromDouble (body->linear_vel_damping);
}

static int
_body_setlinveldamping (PyBody *body, PyObject *value, void *closure)
{
    double damping;
    if (!DoubleFromObj (value, &damping))
        return -1;
    if (damping < 0 || damping > 1)
    {
        PyErr_SetString(PyExc_ValueError,
            "linear_vel_damping must be in the range [0,1]");
        return -1;
    }
    body->linear_vel_damping = damping;
    return 0;
}

static PyObject*
_body_getangleveldamping (PyBody *body, void *closure)
{
    return PyFloat_FromDouble (body->angle_vel_damping);
}

static int
_body_setangleveldamping (PyBody *body, PyObject *value, void *closure)
{
    double damping;
    if (!DoubleFromObj (value, &damping))
        return -1;
    if (damping < 0 || damping > 1)
    {
        PyErr_SetString(PyExc_ValueError,
            "angle_vel_damping must be in the range [0,1]");
        return -1;
    }
    body->angle_vel_damping = damping;
    return 0;
}

static PyObject*
_body_getvelocity (PyBody *body, void *closure)
{
    return Py_BuildValue ("(ff)", body->linear_velocity.real,
        body->linear_velocity.imag);
}

static int
_body_setvelocity (PyBody *body, PyObject *value, void *closure)
{
    double real, imag;

    if (!PySequence_Check (value) || PySequence_Size (value) != 2)
    {
        PyErr_SetString (PyExc_TypeError, "velocity must be a x, y sequence");
        return -1;
    }
    if (!DoubleFromSeqIndex (value, 0, &real))
        return -1;
    if (!DoubleFromSeqIndex (value, 1, &imag))
        return -1;
    body->linear_velocity.real = real;
    body->linear_velocity.imag = imag;
    return 0;
}

static PyObject*
_body_getanglevelocity (PyBody *body, void *closure)
{
    return PyFloat_FromDouble (RAD2DEG(body->angle_velocity));
}

static int
_body_setanglevelocity (PyBody *body, PyObject *value, void *closure)
{
    double velocity;
    if (!DoubleFromObj (value, &velocity))
        return -1;
    body->angle_velocity = DEG2RAD (velocity);
    return 0;
}

static PyObject*
_body_getposition (PyBody *body, void *closure)
{
    return Py_BuildValue ("(ff)", body->position.real, body->position.imag);
}

static int
_body_setposition (PyBody *body, PyObject *value, void *closure)
{
    double real, imag;

    if (!PySequence_Check (value) || PySequence_Size (value) != 2)
    {
        PyErr_SetString (PyExc_TypeError, "position must be a x, y sequence");
        return -1;
    }
    if (!DoubleFromSeqIndex (value, 0, &real))
        return -1;
    if (!DoubleFromSeqIndex (value, 1, &imag))
        return -1;
    body->position.real = real;
    body->position.imag = imag;
    return 0;
}

static PyObject*
_body_getforce (PyBody *body, void *closure)
{
    return Py_BuildValue ("(ff)", body->force.real, body->force.imag);
}

static int
_body_setforce (PyBody *body, PyObject *value, void *closure)
{
    double real, imag;

    if (!PySequence_Check(value) || PySequence_Size (value) != 2)
    {
        PyErr_SetString (PyExc_TypeError, "force must be a x, y sequence");
        return -1;
    }
    if (!DoubleFromSeqIndex (value, 0, &real))
        return -1;
    if (!DoubleFromSeqIndex (value, 1, &imag))
        return -1;
    body->force.real = real;
    body->force.imag = imag;
    return 0;
}

static PyObject*
_body_getstatic (PyBody *body, void *closure)
{
    if (body->isstatic)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static int
_body_setstatic (PyBody *body, PyObject *value, void *closure)
{
    int isstatic = PyObject_IsTrue (value);
    if (isstatic == -1)
        return -1;
    body->isstatic = isstatic;
    return 0;
}

/* Methods */
static PyObject*
_body_getpoints (PyObject *self, PyObject *args)
{
    PyBody* body = (PyBody*)self;
    Py_ssize_t i, count;
    PyObject* list;
    PyVector2 *vertices;
    PyVector2 golVertex;
    PyObject* tuple;

    if (!body->shape)
    {
        Py_RETURN_NONE;
    }

    vertices = PyShape_GetVertices_FAST ((PyShape*)body->shape, &count);
    if (!vertices)
    {
        /* TODO: does get_vertices() set the error? */
        return NULL;
    }

    list = PyList_New (count);
    if (!list)
    {
        PyMem_Free (vertices);
        return NULL;
    }

    for (i = 0; i < count; i++)
    {
        PyBody_GetGlobalPos (body, vertices[i], golVertex);
        tuple = PyVector2_AsTuple (golVertex);
        if (!tuple)
        {
            Py_DECREF (list);
            PyMem_Free (vertices);
            return NULL;
        }
        PyList_SET_ITEM (list, i, tuple);
    }
    PyMem_Free (vertices);
    return list;
}

/* C API */
PyObject*
PyBody_New (void)
{
    PyBody* body = (PyBody*) PyObject_New (PyBody, &PyBody_Type);
    if (!body)
        return NULL;
    _body_init_values (body);
    return (PyObject*) body;
}

PyObject*
PyBody_CheckCollision (PyObject *body1, PyObject *body2)
{
    if (!PyBody_Check (body1) || !PyBody_Check (body2))
    {
        PyErr_SetString (PyExc_TypeError,
            "body arguments must be Body objects");
        return NULL;
    }
    if (!((PyBody*)body1)->shape || !((PyBody*)body2)->shape)
    {
        PyErr_SetString (PyExc_ValueError,
            "body arguments must have shapes assigned");
        return NULL;
    }
    return PyBody_CheckCollision_FAST ((PyBody*)body1, (PyBody*)body2);
}

PyObject*
PyBody_CheckCollision_FAST (PyBody *body1, PyBody *body2)
{
    PyObject *retval = NULL;
    PyContact *contact;
    Py_ssize_t i;
    
    PyErr_Clear ();
    
    /* Assume, the objects are consistent! */
    retval = PyShape_Collide_FAST ((PyShape*)body1->shape, body1->position,
        body1->rotation, (PyShape*)body2->shape, body2->position,
        body2->rotation);
    
    if (!retval) /* Error */
        return NULL;
    if (retval == Py_None)
        return retval;

    for (i = 0; i < PyList_Size (retval); i++)
    {
        /* Update the empty contacts. */
        contact = (PyContact*) PyList_GET_ITEM (retval, i);
        contact->joint.body1 = (PyObject*) body1;
        contact->joint.body2 = (PyObject*) body2;
    }
    return retval;
}

void
body_export_capi (void **capi)
{
    capi[PHYSICS_BODY_FIRSTSLOT] = &PyBody_Type;
    capi[PHYSICS_BODY_FIRSTSLOT + 1] = &PyBody_New;
}
