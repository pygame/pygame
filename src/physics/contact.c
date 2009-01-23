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
static PyObject* _contact_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
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
    _contact_new,               /* tp_new */
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

static PyObject*
_contact_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyContact *contact = (PyContact*) PyJoint_Type.tp_new (type, args, kwds);
    if (!contact)
        return NULL;
    PyVector2_Set (contact->position, 0, 0);
    PyVector2_Set (contact->normal, 0, 0);
    PyVector2_Set (contact->dv, 0, 0);
    contact->depth = 0;
    contact->weight = 0;
    contact->resist = 0;
    contact->kfactor = 0;
    contact->tfactor = 0;
    PyVector2_Set (contact->acc_moment, 0, 0);
    PyVector2_Set (contact->split_acc_moment, 0, 0);
    contact->joint.solve_constraints = _solve_constraints;
    return (PyObject*) contact;
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
    return (PyObject*) PyContact_Type.tp_new (&PyContact_Type, NULL, NULL);
}

int
PyContact_Collision (PyObject *contact, double steptime)
{
    if (!contact || !PyContact_Check (contact))
    {
        PyErr_SetString (PyExc_TypeError, "contact must be a Contact");
        return 0;
    }
    return PyContact_Collision_FAST ((PyContact*)contact, steptime);
}

int
PyContact_Collision_FAST (PyContact *contact, double steptime)
{
/**
 * MAX_C_DEP is the maximal permitted penetrating depth after collision
 * reaction.
 * BIAS_FACTOR is a empirical factor. The two constants are used for
 * position collision after collision reaction is done.
 * for further learning you can read Erin Canto's GDC 2006 Slides, page 23.
 * (You can download it from www.gphysics.com)
 */
#define MAX_C_DEP 0.01
#define BIAS_FACTOR 0.2

    PyVector2 neg_dV, refV, incidV;
    PyVector2 refR, incidR;
    PyBody *refBody, *incidBody;
    double moment_len;
    PyVector2 moment;

    double vbias;
    PyVector2 brefV, bincidV, bneg_dV;
    double bm_len;
    PyVector2 bm;

    refBody = (PyBody *) contact->joint.body1;
    incidBody = (PyBody *)contact->joint.body2;

    contact->resist = sqrt(refBody->restitution*incidBody->restitution);

    /*
     * The algorithm below is an implementation of the empirical formula for
     * impulse-based collision reaction. You can learn the formula thoroughly
     * from Helmut Garstenauer's thesis, Page 60.
     */
    refR = c_diff(contact->position, refBody->position);
    incidR = c_diff(contact->position, incidBody->position);
    //dV = v2 + w2xr2 - (v1 + w1xr1)
    incidV = c_sum(incidBody->linear_velocity,
        PyVector2_fCross(incidBody->angle_velocity, incidR));
    refV = c_sum(refBody->linear_velocity,
        PyVector2_fCross(refBody->angle_velocity, refR));

    contact->dv = c_diff(incidV, refV);
    neg_dV = c_diff(refV, incidV);
	
    moment_len = PyVector2_Dot (PyVector2_MultiplyWithReal(neg_dV,
            (1 + contact->resist)), contact->normal)/contact->kfactor;
    moment_len = MAX(0, moment_len);
    
    /* finally we get the momentum(oh...) */
    moment = PyVector2_MultiplyWithReal(contact->normal, moment_len);
    contact->acc_moment.real += moment.real/contact->weight;
    contact->acc_moment.imag += moment.imag/contact->weight; 

    /* split impulse */
    vbias = BIAS_FACTOR*MAX(0, contact->depth - MAX_C_DEP)/steptime;
    /* biased dv */
    bincidV = c_sum(incidBody->bias_lv,
        PyVector2_fCross(incidBody->bias_w, incidR));
    brefV = c_sum(refBody->bias_lv, PyVector2_fCross(refBody->bias_w, refR));
    bneg_dV = c_diff(brefV, bincidV); 
    /* biased moment */
    bm_len = PyVector2_Dot(PyVector2_MultiplyWithReal(bneg_dV, 1.),
        contact->normal)/contact->kfactor;
    bm_len = MAX(0, bm_len + vbias/contact->kfactor);
    bm = PyVector2_MultiplyWithReal(contact->normal, bm_len);
    contact->split_acc_moment.real += bm.real/contact->weight;
    contact->split_acc_moment.imag += bm.imag/contact->weight;

#undef MAX_C_DEP
#undef BIAS_FACTOR
    return 1;
}

void
contact_export_capi (void **capi)
{
    capi[PHYSICS_CONTACT_FIRSTSLOT + 0] = &PyContact_Type;
    capi[PHYSICS_CONTACT_FIRSTSLOT + 1] = PyContact_New;
}

