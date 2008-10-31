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

#define PHYSICS_CONTACT_INTERNAL

#include "physicsmod.h"
#include "pgphysics.h"

static void  _solve_constraints (PyJoint *joint, double steptime);

static void _contact_dealloc (PyContact *contact);
static int _contact_init (PyContact *self, PyObject *args, PyObject *kwds);

/**
 * Methods, which are bound to the PyContact type.
 */
static PyMethodDef _contact_methods[] =
{
    { NULL, NULL, 0, NULL }
};

/**
 * Getters/Setters
 */
static PyGetSetDef _contact_getsets[] =
{
    { NULL, NULL, NULL, NULL, NULL }
};

PyTypeObject PyContact_Type =
{
    TYPE_HEAD(NULL, 0)
    "physics.Contact",          /* tp_name */
    sizeof (PyContact),         /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _contact_dealloc,/* tp_dealloc */
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
    _contact_methods,           /* tp_methods */
    0,                          /* tp_members */
    _contact_getsets,           /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)_contact_init,    /* tp_init */
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

static void
_solve_constraints (PyJoint *joint, double steptime)
{
    PyContact *contact = (PyContact*) joint;
    PyBody *refbody, *incbody;
    PyVector2 moment, bm, refr, incidr;
    PyShape *refshape, *incshape;

    if (!contact)
        return;
    
    refbody = (PyBody*)joint->body1;
    incbody = (PyBody*)joint->body2;
    refshape = (PyShape*) refbody->shape;
    incshape = (PyShape*) incbody->shape;

    moment = contact->acc_moment;
    bm = contact->split_acc_moment;

    refr = c_diff (contact->position, refbody->position);
    incidr = c_diff (contact->position, incbody->position);

    if (PyVector2_Dot (contact->dv, contact->normal) > 0)
        return;

    if (!refbody->isstatic)
    {
        refbody->linear_velocity = c_diff (refbody->linear_velocity, 
            PyVector2_DivideWithReal (moment, refbody->mass));
        refbody->angle_velocity -=
            PyVector2_Cross (refr, moment) / refshape->inertia;

        refbody->bias_lv = c_diff (refbody->bias_lv,
            PyVector2_DivideWithReal (bm, refbody->mass));
        refbody->bias_w -= PyVector2_Cross (refr, bm) / refshape->inertia;
    }

    if (!incbody->isstatic)
    {
        incbody->linear_velocity = c_sum(incbody->linear_velocity, 
            PyVector2_DivideWithReal (moment, incbody->mass));
        incbody->angle_velocity +=
            PyVector2_Cross (incidr, moment) / incshape->inertia;

        incbody->bias_lv = c_sum (incbody->bias_lv,
            PyVector2_DivideWithReal (bm, incbody->mass));
        incbody->bias_w += PyVector2_Cross (incidr, bm) / incshape->inertia;
    }
}

static void
_contact_dealloc (PyContact *contact)
{
    PyObject *obj = (PyObject*) &(contact->joint);
    obj->ob_type->tp_free ((PyObject*)contact);
}

static int
_contact_init (PyContact *self, PyObject *args, PyObject *kwds)
{
    if (PyJoint_Type.tp_init ((PyObject *)self, args, kwds) < 0)
        return -1;
    /* TODO */
    return 0;
}

/* Getters/Setters */

/* C API */
PyObject*
PyContact_New (void)
{
    /* TODO */
    return (PyObject*) PyObject_New (PyContact, &PyContact_Type);
}

int
PyContact_Collision (PyObject *contact, double steptime)
{
    if (!PyContact_Check (contact))
    {
        PyErr_SetString (PyExc_TypeError, "contact must be a Contact");
        return 0;
    }
    return PyContact_Collision_FAST ((PyContact*)contact, steptime);
}

int
PyContact_Collision_FAST (PyContact *contact, double steptime)
{
    return 1;
}

void
contact_export_capi (void **capi)
{
    capi[PHYSICS_CONTACT_FIRSTSLOT + 0] = &PyContact_Type;
}

