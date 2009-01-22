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

#define PHYSICS_JOINT_INTERNAL

#include "physicsmod.h"
#include "pgphysics.h"

static void _joint_dealloc (PyJoint *joint);
static PyObject* _joint_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _joint_init (PyJoint *joint, PyObject *args, PyObject *kwds);

static PyObject* _joint_getdict (PyJoint *joint, void *closure);
static PyObject* _joint_getbody1 (PyJoint *joint, void *closure);
static PyObject* _joint_getbody2 (PyJoint *joint, void *closure);
static PyObject* _joint_getcollideconnect (PyJoint *joint, void *closure);

static PyObject* _joint_solveconstraints (PyJoint *joint, PyObject *args);

/**
 * Methods, which are bound to the PyJoint type.
 */
static PyMethodDef _joint_methods[] =
{
    { "solve_constraints", (PyCFunction)_joint_solveconstraints, METH_VARARGS,
      NULL },
    { NULL, NULL, 0, NULL }
};

/**
 * Getters/Setters
 */
static PyGetSetDef _joint_getsets[] =
{
    { "__dict__", (getter) _joint_getdict, NULL, NULL, NULL },
    { "body1", (getter) _joint_getbody1, NULL, NULL, NULL },
    { "body2", (getter) _joint_getbody2, NULL, NULL, NULL },
    { "is_collide_connect", (getter) _joint_getcollideconnect, NULL, NULL,
      NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

PyTypeObject PyJoint_Type =
{
    TYPE_HEAD(NULL, 0)
    "physics.Joint",            /* tp_name */
    sizeof (PyJoint),           /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _joint_dealloc,/* tp_dealloc */
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
    _joint_methods,             /* tp_methods */
    0,                          /* tp_members */
    _joint_getsets,             /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    offsetof (PyJoint, dict),   /* tp_dictoffset */
    (initproc)_joint_init,      /* tp_init */
    0,                          /* tp_alloc */
    _joint_new,                 /* tp_new */
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
_joint_dealloc (PyJoint *joint)
{
    Py_XDECREF (joint->dict);
    ((PyObject*)joint)->ob_type->tp_free ((PyObject*)joint);
}

static PyObject*
_joint_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyJoint* joint = (PyJoint*) type->tp_alloc (type, 0);
    if (!joint)
        return NULL;
    
    joint->dict = NULL;
    joint->body1 = NULL;
    joint->body2 = NULL;
    joint->iscollideconnect = 0;
    joint->solve_constraints = NULL;
    return (PyObject *) joint;
}

static int
_joint_init (PyJoint *joint, PyObject *args, PyObject *kwds)
{
    PyObject *body1, *body2;
    int collide;
    
    if (!PyArg_ParseTuple (args, "O!O!i", &PyBody_Type, &body1,
        &PyBody_Type, &body2, &collide))
        return -1;
    if (collide)
        joint->iscollideconnect = 1;
    Py_INCREF (body1);
    Py_INCREF (body2);
    joint->body1 = body1;
    joint->body1 = body2;
    return 0;
}

/* Getters/Setters */
static PyObject*
_joint_getdict (PyJoint *joint, void *closure)
{
    if (!joint->dict)
    {
        joint->dict = PyDict_New ();
        if (!joint->dict)
            return NULL;
    }
    Py_INCREF (joint->dict);
    return joint->dict;
}

static PyObject*
_joint_getbody1 (PyJoint *joint, void *closure)
{
    Py_INCREF (joint->body1);
    return joint->body1;
}

static PyObject*
_joint_getbody2 (PyJoint *joint, void *closure)
{
    Py_INCREF (joint->body2);
    return joint->body2;
}

static PyObject*
_joint_getcollideconnect (PyJoint *joint, void *closure)
{
    return PyBool_FromLong (joint->iscollideconnect);
}

/* Methods */
static PyObject*
_joint_solveconstraints (PyJoint *joint, PyObject *args)
{
    if (joint->solve_constraints)
    {
        double steptime;
        if (!PyArg_ParseTuple (args, "d", &steptime))
            return NULL;
        joint->solve_constraints (joint, steptime);
        Py_RETURN_NONE;
    }
    PyErr_SetString (PyExc_NotImplementedError, "method not implemented");
    return NULL;
}

/* C API */
int
PyJoint_SolveConstraints (PyObject *joint, double steptime)
{
    if (!PyJoint_Check (joint))
    {
        PyErr_SetString (PyExc_TypeError, "joint must be a Joint");
        return 0;
    }
    return PyJoint_SolveConstraints_FAST ((PyJoint*)joint, steptime);
}

int
PyJoint_SolveConstraints_FAST (PyJoint *joint, double steptime)
{
    PyObject *result;
    
    if (joint->solve_constraints)
    {
        joint->solve_constraints (joint, steptime);
        return 1;
    }
    result = PyObject_CallMethod ((PyObject*)joint, "solve_constraints", "d",
        steptime);
    if (!result)
        return 0;

    Py_DECREF (result);
    return 1;
}

void
joint_export_capi (void **capi)
{
    capi[PHYSICS_JOINT_FIRSTSLOT + 0] = &PyJoint_Type;
}
