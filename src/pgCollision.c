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

#include <assert.h>
#include "pgDeclare.h"
#include "pgVector2.h"
#include "pgCollision.h"

static int _LiangBarskey_Internal(double p, double q, double* u1, double* u2);
static void _UpdateV(PyJointObject* joint, double step);
static void _UpdateP(PyJointObject* joint, double step);
static void _ContactDestroy(PyJointObject* contact);
static PyObject* _ContactNewInternal(PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static PyJointObject* _ContactNew(PyBodyObject* refBody,
    PyBodyObject* incidBody);

// We borrow this graph from Box2DLite
// Box vertex and edge numbering:
//
//        ^ y
//        |
//        e3
//   v3 ----- v2
//    |        |
// e0 |        | e2  --> x
//    |        |
//   v0 ----- v1
//        e1


//TODO: add rest contact


/**
 * Apply Liang-Barskey clip on a AABB box
 * (p1, p2) is the input line segment to be clipped (note: it's a 2d vector)
 * (ans_p1, ans_p2) is the output line segment
 * TEST: Liang-Barskey clip has been tested.
 *
 * @param p
 * @param q
 * @param u1
 * @param u2
 * @return
 */
static int _LiangBarskey_Internal(double p, double q, double* u1, double* u2)
{
    double val;

    if(IS_NEAR_ZERO(p))
    {
        if(q < 0)
            return 0;
        return 1;
    }
    
    val = q/p;
    
    if(p < 0)
        *u1 = MAX(*u1, val);
    else
        *u2 = MIN(*u2, val);

    return 1; 
}

/**
 * TODO
 *
 * @param joint
 * @param step
 */
static void _UpdateV(PyJointObject* joint, double step)
{
    PyContact *contact;
    PyBodyObject *refBody, *incidBody;
    PyVector2 moment, bm;
    PyVector2 refR, incidR;
    PyShapeObject *refShape, *incidShape;

    contact = (PyContact*)joint;
    refBody = (PyBodyObject *)joint->body1;
    incidBody = (PyBodyObject *)joint->body2;
    refShape = (PyShapeObject*) refBody->shape;
    incidShape = (PyShapeObject*) incidBody->shape;

    moment = **(contact->ppAccMoment);
    bm = **(contact->ppSplitAccMoment);

    refR = c_diff(contact->pos, refBody->vecPosition);
    incidR = c_diff(contact->pos, incidBody->vecPosition);

    if(PyVector2_Dot(contact->dv, contact->normal) > 0) return;

    if(!refBody->bStatic)
    {
        refBody->vecLinearVelocity = c_diff(refBody->vecLinearVelocity, 
            PyVector2_DivideWithReal(moment, refBody->fMass));
        refBody->fAngleVelocity -= PyVector2_Cross(refR, moment)/refShape->rInertia;

        refBody->cBiasLV = c_diff(refBody->cBiasLV,
            PyVector2_DivideWithReal(bm, refBody->fMass));
        refBody->cBiasW -= PyVector2_Cross(refR, bm)/refShape->rInertia;
    }

    if(!incidBody->bStatic)
    {
        incidBody->vecLinearVelocity = c_sum(incidBody->vecLinearVelocity, 
            PyVector2_DivideWithReal(moment, incidBody->fMass));
        incidBody->fAngleVelocity += PyVector2_Cross(incidR, moment)/incidShape->rInertia;

        incidBody->cBiasLV = c_sum(incidBody->cBiasLV,
            PyVector2_DivideWithReal(bm, incidBody->fMass));
        incidBody->cBiasW += PyVector2_Cross(incidR, bm)/incidShape->rInertia;
    }
}

/**
 * TODO
 *
 * @param joint
 * @param step
 */
static void _UpdateP(PyJointObject* joint, double step)
{
    //isolated function
}

int Collision_LiangBarskey(AABBBox* box, PyVector2* p1, PyVector2* p2, 
    PyVector2* ans_p1, PyVector2* ans_p2)
{
    PyVector2 dp;
    double u1, u2;
	

    u1 = 0.f;
    u2 = 1.f;
    dp = c_diff(*p2, *p1);	//dp = p2 - p1

    if(!_LiangBarskey_Internal(-dp.real, p1->real - box->left, &u1, &u2)) return 0;
    if(!_LiangBarskey_Internal(dp.real, box->right - p1->real, &u1, &u2)) return 0;
    if(!_LiangBarskey_Internal(-dp.imag, p1->imag - box->bottom, &u1, &u2)) return 0;
    if(!_LiangBarskey_Internal(dp.imag, box->top - p1->imag, &u1, &u2)) return 0;

    if(u1 > u2) return 0;

    if(u1 == 0.f)
        *ans_p1 = *p1;
    else
        *ans_p1 = c_sum(*p1, PyVector2_MultiplyWithReal(dp, u1)); //ans_p1 = p1 + u1*dp
    if(u2 == 1.f)
        *ans_p2 = *p2;
    else
        *ans_p2 = c_sum(*p1, PyVector2_MultiplyWithReal(dp, u2)); //ans_p2 = p2 + u2*dp;

    return 1;
}

int Collision_PartlyLB(AABBBox* box, PyVector2* p1, PyVector2* p2, 
    CollisionAxis axis, PyVector2* ans_p1, PyVector2* ans_p2, 
    int* valid_p1, int* valid_p2)
{
    PyVector2 dp;
    double u1, u2;
	
    u1 = 0.f;
    u2 = 1.f;
    dp = c_diff(*p2, *p1);

    switch(axis)
    {
    case CA_X:
        if(!_LiangBarskey_Internal(-dp.imag, p1->imag - box->bottom, &u1, &u2)) return 0;
        if(!_LiangBarskey_Internal(dp.imag, box->top - p1->imag, &u1, &u2)) return 0;
        break;
    case CA_Y:
        if(!_LiangBarskey_Internal(-dp.real, p1->real - box->left, &u1, &u2)) return 0;
        if(!_LiangBarskey_Internal(dp.real, box->right - p1->real, &u1, &u2)) return 0;
        break;
    default:
        assert(0);
        break;
    }

    if(u1 > u2) return 0;

    if(u1 == 0.f)
        *ans_p1 = *p1;
    else
        *ans_p1 = c_sum(*p1, PyVector2_MultiplyWithReal(dp, u1)); //ans_p1 = p1 + u1*dp
    if(u2 == 1.f)
        *ans_p2 = *p2;
    else
        *ans_p2 = c_sum(*p1, PyVector2_MultiplyWithReal(dp, u2)); //ans_p2 = p2 + u2*dp;

    switch(axis)
    {
    case CA_X:
        *valid_p1 = PyMath_LessEqual(box->left, ans_p1->real) && 
            PyMath_LessEqual(ans_p1->real, box->right);
        *valid_p2 = PyMath_LessEqual(box->left, ans_p2->real) && 
            PyMath_LessEqual(ans_p2->real, box->right);
        break;
    case CA_Y:
        *valid_p1 = PyMath_LessEqual(box->bottom, ans_p1->imag) && 
            PyMath_LessEqual(ans_p1->imag, box->top);
        *valid_p2 = PyMath_LessEqual(box->bottom, ans_p2->imag) && 
            PyMath_LessEqual(ans_p2->imag, box->top);
        break;
    default:
        assert(0);
        break;
    }

    return *valid_p1 || *valid_p2;
}

void Collision_ApplyContact(PyObject* contactObject, double step)
{
/**
 * TODO
 */
#define MAX_C_DEP 0.01

/**
 * TODO
 */
#define BIAS_FACTOR 0.2

    PyVector2 neg_dV, refV, incidV;
    PyVector2 refR, incidR;
    PyContact *contact;
    PyBodyObject *refBody, *incidBody;
    double moment_len;
    PyVector2 moment;
    PyVector2* p;

    double vbias;
    PyVector2 brefV, bincidV, bneg_dV;
    double bm_len;
    PyVector2 bm;

    contact = (PyContact*)contactObject;
    refBody = (PyBodyObject*)contact->joint.body1;
    incidBody = (PyBodyObject*)contact->joint.body2;

    contact->resist = sqrt(refBody->fRestitution*incidBody->fRestitution);

    /*
     * TODO: explain the magic happening here.
     */

    refR = c_diff(contact->pos, refBody->vecPosition);
    incidR = c_diff(contact->pos, incidBody->vecPosition);
    //dV = v2 + w2xr2 - (v1 + w1xr1)
    incidV = c_sum(incidBody->vecLinearVelocity, PyVector2_fCross(incidBody->fAngleVelocity,
            incidR));
    refV = c_sum(refBody->vecLinearVelocity, PyVector2_fCross(refBody->fAngleVelocity,
            refR));

    contact->dv = c_diff(incidV, refV);
    neg_dV = c_diff(refV, incidV);
	
    moment_len = PyVector2_Dot(PyVector2_MultiplyWithReal(neg_dV, (1 + contact->resist)), 
        contact->normal)/contact->kFactor;
    moment_len = MAX(0, moment_len);
    
    //finally we get the momentum(oh...)
    moment = PyVector2_MultiplyWithReal(contact->normal, moment_len);
    p = *(contact->ppAccMoment);
    //TODO: test weight
    p->real += moment.real/contact->weight;
    p->imag += moment.imag/contact->weight; 

    //split impulse
    vbias = BIAS_FACTOR*MAX(0, contact->depth - MAX_C_DEP)/step;
    //biasdv
    bincidV = c_sum(incidBody->cBiasLV, PyVector2_fCross(incidBody->cBiasW, incidR));
    brefV = c_sum(refBody->cBiasLV, PyVector2_fCross(refBody->cBiasW, refR));
    bneg_dV = c_diff(brefV, bincidV); 
    //bias_moment
    bm_len = PyVector2_Dot(PyVector2_MultiplyWithReal(bneg_dV, 1.),
        contact->normal)/contact->kFactor;
    bm_len = MAX(0, bm_len + vbias/contact->kFactor);
    bm = PyVector2_MultiplyWithReal(contact->normal, bm_len);
    p = *(contact->ppSplitAccMoment);
    p->real += bm.real/contact->weight;
    p->imag += bm.imag/contact->weight;


#undef MAX_C_DEP
#undef BIAS_FACTOR
}


PyTypeObject PyContact_Type =
{
    PyObject_HEAD_INIT(NULL)
    0,
    "physics.Contact",		/* tp_name */
    sizeof(PyContact),		/* tp_basicsize */
    0,                          /* tp_itemsize */
    0,				/* tp_dealloc */
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "",                         /* tp_doc */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    0,				/* tp_methods */
    0,				/* tp_members */
    0,				/* tp_getset */
    0,				/* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    0,				/* tp_init */
    0,				/* tp_alloc */
    _ContactNewInternal,	/* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0                           /* tp_del */
};

/**
 * Deallocates the passed PyContact.
 *
 * @param contact The PyContact to deallocate
 */
static void _ContactDestroy(PyJointObject* contact)
{
    PyVector2 **p = ((PyContact*)contact)->ppAccMoment;
    if(p)
    {
        if(*p)
        {
            PyObject_Free(*p);
            *p = NULL;
        }
        PyObject_Free(p);
        p = NULL;
    }

    p = ((PyContact*)contact)->ppSplitAccMoment;
    if(p)
    {
        if(*p)
        {
            PyObject_Free(*p);
            *p = NULL;
        }
        PyObject_Free(p);
        p = NULL;
    }
}

/**
 * Creates a new PyContact object.
 */
static PyObject* _ContactNewInternal(PyTypeObject *type, PyObject *args,
    PyObject *kwds)
{
    PyContact* op;
    type->tp_base = &PyJoint_Type;
    op = (PyContact*)type->tp_alloc(type, 0);
    return (PyObject*)op;
}

/**
 * Creates a new PyContact object and initializes its internals.
 *
 * @param refBody
 * @param incidBody
 * @return A new PyContact.
 */
static PyJointObject* _ContactNew(PyBodyObject* refBody,PyBodyObject* incidBody)
{
    PyContact* contact;
    //TODO: this function would be replaced.
    contact = (PyContact*)_ContactNewInternal(&PyContact_Type, NULL, NULL);
    contact->joint.body1 = (PyObject*)refBody;
    contact->joint.body2 = (PyObject*)incidBody;
    contact->joint.SolveConstraintPosition = _UpdateP;
    contact->joint.SolveConstraintVelocity = _UpdateV;
    contact->joint.Destroy = _ContactDestroy;

    contact->ppAccMoment = NULL;
    contact->ppSplitAccMoment = NULL;

    return (PyJointObject*)contact;
}

/* Internally used PyContact functions */

PyObject* PyContact_New(PyBodyObject* refBody, PyBodyObject* incidBody)
{
    return (PyObject*)_ContactNew (refBody, incidBody);
}
