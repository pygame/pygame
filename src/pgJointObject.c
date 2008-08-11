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
#include "pgDeclare.h"
#include "pgVector2.h"
#include "pgHelpFunctions.h"
#include "pgBodyObject.h"
#include "pgJointObject.h"

static void _JointBase_InitInternal(PyJointObject* joint,PyObject* b1,
    PyObject* b2,int bCollideConnect);
static void _JointDestroy(PyJointObject* joint);
static PyObject* _JointBaseNew(PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _JointBase_init(PyJointObject* joint,PyObject *args, PyObject *kwds);

static int _Joint_setBody1(PyJointObject* joint,PyObject* value,void* closure);
static PyObject* _Joint_getBody1(PyJointObject* joint,void* closure);
static int _Joint_setBody2(PyJointObject* joint,PyObject* value,void* closure);
static PyObject* _Joint_getBody2(PyJointObject* joint,void* closure);

static int _DistanceJoint_init(PyDistanceJointObject* joint,PyObject *args,
    PyObject *kwds);
static PyObject* _DistanceJointNew(PyTypeObject *type, PyObject *args,
    PyObject *kwds);

static void _SolveDistanceJointPosition(PyJointObject* joint,double stepTime);
static void _DistanceJoint_ComputeOneDynamic (PyBodyObject* body,
    PyVector2* staticAnchor, PyVector2* localAnchor,double distance,
    double stepTime);
static void _DistanceJoint_ComputeTwoDynamic(PyDistanceJointObject* joint,
    double stepTime);
static void _SolveDistanceJointVelocity(PyJointObject* joint,double stepTime);
static void _ReComputeDistance(PyDistanceJointObject* joint);

static PyObject* _DistanceJoint_getDistance(PyDistanceJointObject* joint,
    void* closure);
/*static int _DistanceJoint_setDistance(PyDistanceJointObject* joint,
  PyObject* value,void* closure);*/
static PyObject* _DistanceJoint_getAnchor1(PyDistanceJointObject* joint,
    void* closure);
static int _DistanceJoint_setAnchor1(PyDistanceJointObject* joint,
    PyObject* value,void* closure);
static PyObject* _DistanceJoint_getAnchor2(PyDistanceJointObject* joint,
    void* closure);
static int _DistanceJoint_setAnchor2(PyDistanceJointObject* joint,
    PyObject* value,void* closure);
static PyObject* _DistanceJoint_getPointList(PyObject *self, PyObject *args);

static PyObject* _RevoluteJoint_getPointList(PyRevoluteJointObject* joint,void* closure);
static int _RevoluteJoint_setAnchor(PyRevoluteJointObject* joint,
									 PyObject* value,void* closure);
static PyObject* _RevoluteJoint_getAnchor(PyRevoluteJointObject* joint,
										   void* closure);

static int _RevoluteJoint_init(PyRevoluteJointObject* joint,PyObject *args,
							   PyObject *kwds);
static PyObject* _RevoluteJointNew(PyTypeObject *type, PyObject *args,
								   PyObject *kwds);

static void _SolveRevoluteJointVelocity(PyJointObject* joint,double stepTime);
/* C API */
static PyObject* PyJoint_New(PyObject *body1, PyObject *body2, int collideConnect);
static PyObject* PyDistanceJoint_New(PyObject *body1, PyObject *body2, int collideConnect);
static int PyDistanceJoint_SetAnchors(PyObject *joint,PyVector2 anchor1,PyVector2 anchor2);
static PyObject* PyRevoluteJoint_New(PyObject *body1,PyObject *body2,int collideConnect);
static int PyRevoluteJoint_SetAnchorsFromConnectWorldAnchor(PyObject* joint,PyVector2 initWorldAnchor);


static PyGetSetDef _JointBase_getseters[] = {
    { "body1",(getter)_Joint_getBody1,(setter)_Joint_setBody1,"",NULL },
    { "body2",(getter)_Joint_getBody2,(setter)_Joint_setBody2,"",NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

PyTypeObject PyJoint_Type =
{
    PyObject_HEAD_INIT(NULL)
    0,
    "physics.Joint",            /* tp_name */
    sizeof(PyJointObject),      /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _JointDestroy, /* tp_dealloc */
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
    0,                          /* tp_methods */
    0,				/* tp_members */
    _JointBase_getseters,     /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)_JointBase_init, /* tp_init */
    0,                          /* tp_alloc */
    _JointBaseNew,           /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0                           /* tp_del */
};

static void _JointBase_InitInternal(PyJointObject* joint,PyObject* b1,
    PyObject* b2,int bCollideConnect)
{
    Py_INCREF (b1);
    Py_INCREF (b2);
    joint->body1 = b1;
    joint->body2 = b2;
    joint->isCollideConnect = bCollideConnect;
}

static int _JointBase_init(PyJointObject* joint,PyObject *args, PyObject *kwds)
{
    PyObject* body1, *body2;
    int bCollide;
    static char *kwlist[] = {"body1", "body2", "collide_connect", NULL};

    if (!PyArg_ParseTupleAndKeywords(args,kwds,"OOi",kwlist,&body1,&body2,
            &bCollide))
    {
        return -1;
    }
 
    if (!PyBody_Check (body1))
    {
        PyErr_SetString (PyExc_TypeError, "body1 must be a Body");
        return -1;
    }
    if (!PyBody_Check (body2))
    {
        PyErr_SetString (PyExc_TypeError, "body2 must be a Body");
        return -1;
    }
   
    _JointBase_InitInternal(joint, body1, body2, bCollide);
    return 0;
}

//TODO: this function would get err when inherited level > 2
static void _JointDestroy(PyJointObject* joint)
{
    if (joint->Destroy)
    {
        joint->Destroy(joint);
    }
    Py_XDECREF (joint->body1);
    Py_XDECREF (joint->body2);
    joint->ob_type->tp_free((PyObject*)joint);
}

static PyObject* _JointBaseNew(PyTypeObject *type, PyObject *args,
    PyObject *kwds)
{
    /* In case we have arguments in the python code, parse them later
     * on.
     */
    PyJointObject* joint = (PyJointObject*)type->tp_alloc(type, 0);
    if (!joint)
        return NULL;

    joint->body1 = NULL;
    joint->body2 = NULL;
    joint->isCollideConnect = 0;
    joint->SolveConstraintVelocity = NULL;
    joint->SolveConstraintPosition = NULL;
    joint->Destroy = NULL;

    return (PyObject*) joint;
}

static int _Joint_setBody1(PyJointObject* joint,PyObject* value,void* closure)
{
    if(!PyBody_Check (value))
    {
        PyErr_SetString(PyExc_TypeError, "argument must be a Body");
        return -1;
    }
    Py_XDECREF (joint->body1);
    Py_INCREF (value);
    joint->body1 = value;
    return 0;
}

static PyObject* _Joint_getBody1(PyJointObject* joint,void* closure)
{
    Py_INCREF (joint->body1);
    return joint->body1;
}

static int _Joint_setBody2(PyJointObject* joint,PyObject* value,void* closure)
{
    if(!PyBody_Check (value))
    {
        PyErr_SetString(PyExc_TypeError, "argument must be a Body");
        return -1;
    }
    Py_XDECREF (joint->body2);
    Py_INCREF (value);
    joint->body2 = value;
    return 0;
}

static PyObject* _Joint_getBody2(PyJointObject* joint,void* closure)
{
    Py_INCREF (joint->body2);
    return joint->body2;
}

/* Distance joint */
static PyMethodDef _DistanceJoint_methods[] = {
    {"get_points",_DistanceJoint_getPointList,METH_VARARGS,""	},
    {NULL, NULL, 0, NULL}   /* Sentinel */
};

static PyGetSetDef _DistanceJoint_getseters[] = {
    { "distance",(getter)_DistanceJoint_getDistance,
      /*(setter)_DistanceJoint_setDistance*/NULL,"",NULL, },
    { "anchor1",(getter)_DistanceJoint_getAnchor1,
      (setter)_DistanceJoint_setAnchor1,"",NULL, },
    { "anchor2",(getter)_DistanceJoint_getAnchor2,
      (setter)_DistanceJoint_setAnchor2,"",NULL, },
    { NULL, NULL, NULL, NULL, NULL }
};


PyTypeObject PyDistanceJoint_Type =
{
    PyObject_HEAD_INIT(NULL)
    0,
    "physics.DistanceJoint",            /* tp_name */
    sizeof(PyDistanceJointObject),      /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) 0,		/* tp_dealloc */
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
    _DistanceJoint_methods,    /* tp_methods */
    0,                          /* tp_members */
    _DistanceJoint_getseters,	/* tp_getset */
    0,				/* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)_DistanceJoint_init,  /* tp_init */
    0,                          /* tp_alloc */
    _DistanceJointNew,          /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0                           /* tp_del */
};

static int _DistanceJoint_init(PyDistanceJointObject* joint,PyObject *args,
    PyObject *kwds)
{
    if(PyJoint_Type.tp_init((PyObject*)joint, args, kwds) < 0)
        return -1;
    return 0;
}

static PyObject* _DistanceJointNew(PyTypeObject *type, PyObject *args,
    PyObject *kwds)
{
    PyDistanceJointObject *joint = (PyDistanceJointObject*)
        _JointBaseNew (type, args, kwds);
    if (!joint)
        return NULL;
    
    joint->joint.SolveConstraintVelocity = _SolveDistanceJointVelocity;
    //joint->joint.SolveConstraintPosition = _SolveDistanceJointPosition;
    joint->distance = 0.0;
    PyVector2_Set(joint->anchor1,0,0);
    PyVector2_Set(joint->anchor2,0,0);

    return (PyObject*)joint;
}

static void _SolveDistanceJointPosition(PyJointObject* joint,double stepTime)
{
    PyVector2 vecL,vecP;
    PyDistanceJointObject* pJoint = (PyDistanceJointObject*)joint;
    PyBodyObject *b1, *b2;
    if (joint->body1 && (!joint->body2))
    {
        b1 = (PyBodyObject*)joint->body1;
        vecL = c_diff(b1->vecPosition, pJoint->anchor2);
        PyVector2_Normalize(&vecL);
        vecL = PyVector2_MultiplyWithReal(vecL,pJoint->distance);
        b1->vecPosition = c_sum(pJoint->anchor2,vecL);
        return;
    } 

    if(joint->body1 && joint->body2)
    {
        b1 = (PyBodyObject*)joint->body1;
        b2 = (PyBodyObject*)joint->body2;
        vecL = c_diff(b1->vecPosition, b2->vecPosition);
        PyVector2_Normalize(&vecL);
        vecP = c_sum(b1->vecPosition, b2->vecPosition);
        vecL = PyVector2_MultiplyWithReal(vecL,pJoint->distance * 0.5);
        b1->vecPosition = c_sum(b1->vecPosition,vecL);
        b2->vecPosition = c_diff(b2->vecPosition,vecL);
        return;
    }
}

static void _DistanceJoint_ComputeOneDynamic (PyBodyObject* body,
    PyVector2* staticAnchor, PyVector2* localAnchor,double distance,
    double stepTime)
{
    double a,b,bb,k,temp; //for solve equation
    PyVector2 localP = PyBodyObject_GetGlobalPos(body,localAnchor);
    PyVector2 L = c_diff(*staticAnchor,localP);
    PyVector2 vP = PyBodyObject_GetLocalPointVelocity(body,*localAnchor);
    PyVector2 vPL = PyVector2_Project(L,vP);
    PyVector2 dvBody;
    double dAngleV;

    localP = c_diff(localP,body->vecPosition);
    k = ((PyShapeObject*)body->shape)->rInertia / body->fMass;
    k += PyVector2_GetLengthSquare(localP);
    bb = (distance - PyVector2_GetLength(L));
    PyVector2_Normalize(&L);
    temp = PyVector2_Cross(localP,L);
    a = 1 + PyVector2_Dot(PyVector2_fCross(temp,localP),L) / k;
    b = -PyVector2_Dot(vPL,L);

    temp = b /a;
    dvBody = PyVector2_MultiplyWithReal(L,temp);

    body->vecLinearVelocity = c_sum(body->vecLinearVelocity,dvBody);
    dAngleV = PyVector2_Cross(localP,dvBody);
    dAngleV /= k;
    body->fAngleVelocity += dAngleV;

    // Position correction.
    temp = -bb /a;
    dvBody = PyVector2_MultiplyWithReal(L,temp);
    dAngleV = PyVector2_Cross(localP,dvBody);
    dAngleV /= k;
    body->vecPosition = c_sum(body->vecPosition,dvBody);
    body->fRotation += dAngleV;
    
    return;
}

static void _DistanceJoint_ComputeTwoDynamic(PyDistanceJointObject* joint,
    double stepTime)
{
    PyBodyObject *body1 = (PyBodyObject*)joint->joint.body1;
    PyBodyObject *body2 = (PyBodyObject*)joint->joint.body2;
    double a1,a2,b1,b2,bb,k1,k2,temp,temp1,temp2; //for solve equation
    PyVector2 localP1 = PyBodyObject_GetGlobalPos(body1,&joint->anchor1);
    PyVector2 localP2 = PyBodyObject_GetGlobalPos(body2,&joint->anchor2);
    PyVector2 L = c_diff(localP1,localP2);
	PyVector2 eL = L;
    PyVector2 vP1 = PyBodyObject_GetLocalPointVelocity(body1,joint->anchor1);
    PyVector2 vP2 = PyBodyObject_GetLocalPointVelocity(body2,joint->anchor2);
    PyVector2 vPL1,vPL2;
    PyVector2 dvBody1,dvBody2;
    double dAngleV1,dAngleV2;
	PyVector2_Normalize(&eL)
	vPL1 = PyVector2_Project(eL,vP1);
	vPL2 = PyVector2_Project(eL,vP2);
    k1 = ((PyShapeObject*)body1->shape)->rInertia / body1->fMass;
    k2 = ((PyShapeObject*)body2->shape)->rInertia / body2->fMass;
    
    localP1 = c_diff(localP1,body1->vecPosition);
    localP2 = c_diff(localP2,body2->vecPosition);
    k1 += PyVector2_GetLengthSquare(localP1);
    k2 += PyVector2_GetLengthSquare(localP2);

    bb = (joint->distance - PyVector2_GetLength(L)) ;
    PyVector2_Normalize(&L);
    temp = PyVector2_Cross(localP1,L);
    a1 = 1 + PyVector2_Dot(PyVector2_fCross(temp,localP1),L) / k1;

    a1 /= body1->fMass;

    temp = PyVector2_Cross(localP2,L);
    a2 = 1 + PyVector2_Dot(PyVector2_fCross(temp,localP2),L) / k2;

    a2 /= body2->fMass;

    b1 = PyVector2_Dot(vPL1,L);
    b2 = PyVector2_Dot(vPL2,L);

    temp = (b2 - b1) /(a1 + a2);
    temp1 = temp / body1->fMass;
    temp2 = -temp / body2->fMass;
    dvBody1 = PyVector2_MultiplyWithReal(L,temp1);
    dvBody2 = PyVector2_MultiplyWithReal(L,temp2);

    body1->vecLinearVelocity = c_sum(body1->vecLinearVelocity,dvBody1);
    dAngleV1 = PyVector2_Cross(localP1,dvBody1);
    dAngleV1 /= k1;
    body1->fAngleVelocity += dAngleV1;

    body2->vecLinearVelocity = c_sum(body2->vecLinearVelocity,dvBody2);
    dAngleV2 = PyVector2_Cross(localP2,dvBody2);
    dAngleV2 /= k2;
    body2->fAngleVelocity += dAngleV2;

    //for position correction

    temp = bb /(a1 + a2);
    temp1 = temp / body1->fMass;
    temp2 = -temp / body2->fMass;
    dvBody1 = PyVector2_MultiplyWithReal(L,temp1);
    dvBody2 = PyVector2_MultiplyWithReal(L,temp2);

    body1->vecPosition = c_sum(body1->vecPosition,dvBody1);
    body2->vecPosition = c_sum(body2->vecPosition,dvBody2);
    body1->vecLinearVelocity = c_sum(body1->vecLinearVelocity,PyVector2_MultiplyWithReal(dvBody1,1/stepTime));
    body2->vecLinearVelocity = c_sum(body2->vecLinearVelocity,PyVector2_MultiplyWithReal(dvBody2,1/stepTime));
    dAngleV1 = PyVector2_Cross(localP1,dvBody1);
    dAngleV1 /= k1;
    dAngleV2 = PyVector2_Cross(localP2,dvBody2);
    dAngleV2 /= k2;
    body1->fRotation += dAngleV1;
    body2->fRotation += dAngleV2;
    body1->fAngleVelocity += (dAngleV1 / stepTime);
    body2->fAngleVelocity += (dAngleV2 / stepTime);
}

static void _SolveDistanceJointVelocity(PyJointObject* joint,double stepTime)
{
    PyDistanceJointObject* pJoint = (PyDistanceJointObject*)joint;
    PyBodyObject* body1 = (PyBodyObject*) joint->body1;
    PyBodyObject* body2 = (PyBodyObject*) joint->body2;
/*     if (body1 && (!body2)) */
/*     { */
/*         if (body1->bStatic) */
/*         { */
/*             return; */
/*         } */
/*         else */
/*         { */
			
/*             _PG_DistanceJoint_ComputeOneDynamic(body1,&pJoint->anchor2,&pJoint->anchor1,pJoint->distance,stepTime); */

/*             /\*double a,b,c,d,e,f,k; */
/*               pgVector2 localP = PG_GetGlobalPos(body1,&pJoint->anchor1); */
/*               pgVector2 L = c_diff(localP,pJoint->anchor2); */
/*               pgVector2 vP = PG_GetLocalPointVelocity(body1,pJoint->anchor1); */
/*               pgVector2 vPL = c_project(L,vP); */
/*               pgVector2 dvBody; */
/*               double dAngleV; */
/*               vPL = c_neg(vPL); */

/*               localP = c_diff(localP,body1->vecPosition); */
/*               k = body1->shape->rInertia / body1->fMass; */


			
/*               a = (1 - localP.real * localP.imag / k); */
/*               b = (localP.imag * localP.imag / k); */
/*               c = (localP.real * localP.real /k); */
/*               d = a; */
/*               e = vPL.real; */
/*               f = vPL.imag; */
			
/*               dvBody.imag = (e*c - a*f) / (b*c - a*d); */
/*               dvBody.real = (e*d - f*b) / (a*d - b*c); */

/*               body1->vecLinearVelocity = c_sum(body1->vecLinearVelocity,dvBody); */
/*               dAngleV = c_cross(localP,dvBody); */
/*               dAngleV /= k; */
/*               body1->fRotation += dAngleV;*\/ */


/*             return; */
/*         } */
/*     } */

    if(body1 && body2)
    {
        if (body1->bStatic && body2->bStatic)
        {
            return;
        }
        if (body1->bStatic)
        {
            PyVector2 staticAnchor = PyBodyObject_GetGlobalPos(body1,&pJoint->anchor1);
            _DistanceJoint_ComputeOneDynamic(body2,&staticAnchor,&pJoint->anchor2,pJoint->distance,stepTime);
            return;
        }
        if (body2->bStatic)
        {
            PyVector2 staticAnchor = PyBodyObject_GetGlobalPos(body2,&pJoint->anchor2);
            _DistanceJoint_ComputeOneDynamic(body1,&staticAnchor,&pJoint->anchor1,pJoint->distance,stepTime);
            return;
        }
        _DistanceJoint_ComputeTwoDynamic(pJoint,stepTime);
        
    }
}

static void _ReComputeDistance(PyDistanceJointObject* joint)
{
    PyBodyObject *b1 = (PyBodyObject*) joint->joint.body1;
    PyBodyObject *b2 = (PyBodyObject*) joint->joint.body2;
    if (b1 && b2)
    {
        PyVector2 s1 = PyBodyObject_GetGlobalPos(b1,&joint->anchor1);
        PyVector2 s2 = PyBodyObject_GetGlobalPos(b2,&joint->anchor2);
        joint->distance = PyVector2_GetLength(c_diff(s1,s2));
    }
    else
    {
        PyVector2 s1 = PyBodyObject_GetGlobalPos(b1,&joint->anchor1);
        joint->distance = PyVector2_GetLength(c_diff(s1,joint->anchor2));
    }
}

static PyObject* _DistanceJoint_getDistance(PyDistanceJointObject* joint,
    void* closure)
{
    return PyFloat_FromDouble(joint->distance);
}

/*
static int _DistanceJoint_setDistance(PyDistanceJointObject* joint,
    PyObject* value,void* closure)
{
    if (PyNumber_Check (value))
    {
        PyObject *tmp = PyNumber_Float (value);

        if (tmp)
        {
            double distance = PyFloat_AsDouble (tmp);
            Py_DECREF (tmp);
            if (PyErr_Occurred ())
                return -1;
            joint->distance = distance;
            return 0;
        }
    }
    PyErr_SetString (PyExc_TypeError, "distance must be a float");
    return -1;
    }
*/

static PyObject* _DistanceJoint_getAnchor1(PyDistanceJointObject* joint,
    void* closure)
{
    return Py_BuildValue ("(ff)", joint->anchor1.real, joint->anchor1.imag);
}

static int _DistanceJoint_setAnchor1(PyDistanceJointObject* joint,
    PyObject* value,void* closure)
{
    PyObject *item;
    double real, imag;

    if (!PySequence_Check(value) || PySequence_Size (value) != 2)
    {
        PyErr_SetString (PyExc_TypeError, "anchor must be a x, y sequence");
        return -1;
    }

    item = PySequence_GetItem (value, 0);
    if (!DoubleFromObj (item, &real))
        return -1;
    item = PySequence_GetItem (value, 1);
    if (!DoubleFromObj (item, &imag))
        return -1;
    
    joint->anchor1.real = real;
    joint->anchor1.imag = imag;
    _ReComputeDistance(joint);
    return 0;
}

static PyObject* _DistanceJoint_getAnchor2(PyDistanceJointObject* joint,
    void* closure)
{
    return Py_BuildValue ("(ff)", joint->anchor2.real, joint->anchor2.imag);
}

static int _DistanceJoint_setAnchor2(PyDistanceJointObject* joint,
    PyObject* value,void* closure)
{
    PyObject *item;
    double real, imag;

    if (!PySequence_Check(value) || PySequence_Size (value) != 2)
    {
        PyErr_SetString (PyExc_TypeError, "anchor must be a x, y sequence");
        return -1;
    }

    item = PySequence_GetItem (value, 0);
    if (!DoubleFromObj (item, &real))
        return -1;
    item = PySequence_GetItem (value, 1);
    if (!DoubleFromObj (item, &imag))
        return -1;
    
    joint->anchor2.real = real;
    joint->anchor2.imag = imag;
    _ReComputeDistance(joint);
    return 0;
}

static PyObject* _DistanceJoint_getPointList(PyObject *self, PyObject *args)
{
    PyDistanceJointObject* joint = (PyDistanceJointObject*)self;
    PyObject* list  = PyList_New(2);

    PyVector2 p = PyBodyObject_GetGlobalPos(((PyBodyObject*)joint->joint.body1),&joint->anchor1);
    PyObject* tuple = FromPhysicsVector2ToPoint(p);
    PyList_SetItem(list,0,tuple);

    if(joint->joint.body2)
    {
        p = PyBodyObject_GetGlobalPos(((PyBodyObject*)joint->joint.body2),&joint->anchor2);
		
    }
    else
    {
        p = joint->anchor2;
    }

    tuple = FromPhysicsVector2ToPoint(p);
    PyList_SetItem(list,1,tuple);
    return list;
}

/* C API */
static PyObject* PyJoint_New(PyObject *body1, PyObject *body2, int collideConnect)
{
    PyObject *joint;

    if (!PyBody_Check (body1))
    {
        PyErr_SetString (PyExc_TypeError, "body1 must be a Body");
        return 0;
    }
    if (!PyBody_Check (body2))
    {
        PyErr_SetString (PyExc_TypeError, "body2 must be a Body");
        return 0;
    }

    joint = _JointBaseNew (&PyJoint_Type, NULL, NULL);
    if (!joint)
        return NULL;
    _JointBase_InitInternal ((PyJointObject*)joint, body1, body2, collideConnect);
    return joint;
}

static PyObject* PyDistanceJoint_New(PyObject *body1, PyObject *body2, int collideConnect)
{
    PyObject* joint;

    if (!PyBody_Check (body1))
    {
        PyErr_SetString (PyExc_TypeError, "body1 must be a Body");
        return 0;
    }
    if (!PyBody_Check (body2))
    {
        PyErr_SetString (PyExc_TypeError, "body2 must be a Body");
        return 0;
    }

    joint = _DistanceJointNew (&PyDistanceJoint_Type, NULL, NULL);
    if (!joint)
        return NULL;

    _JointBase_InitInternal ((PyJointObject*)joint, body1, body2, collideConnect);
    return joint;
}

static int PyDistanceJoint_SetAnchors(PyObject *joint,PyVector2 anchor1,PyVector2 anchor2)
{
    if (!PyDistanceJoint_Check (joint))
    {
        PyErr_SetString (PyExc_TypeError, "joint must be a DistanceJoint");
        return 0;
    }

    ((PyDistanceJointObject*)joint)->anchor1.real = anchor1.real;
    ((PyDistanceJointObject*)joint)->anchor1.imag = anchor1.imag;
    ((PyDistanceJointObject*)joint)->anchor2.real = anchor2.real;
    ((PyDistanceJointObject*)joint)->anchor2.imag = anchor2.imag;
    _ReComputeDistance((PyDistanceJointObject*)joint);
    return 1;
}


/*RevoluteJoint*/
static PyMethodDef _RevoluteJoint_methods[] = {
	{"get_points",_RevoluteJoint_getPointList,METH_VARARGS,""	},
	{NULL, NULL, 0, NULL}   /* Sentinel */
};

static PyGetSetDef _RevoluteJoint_getseters[] = {
	{ "anchor",(getter)_RevoluteJoint_getAnchor,
	(setter)_RevoluteJoint_setAnchor,"",NULL, },
	{ NULL, NULL, NULL, NULL, NULL }
};


PyTypeObject PyRevoluteJoint_Type =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"physics.RevoluteJoint",            /* tp_name */
	sizeof(PyRevoluteJointObject),      /* tp_basicsize */
	0,                          /* tp_itemsize */
	(destructor) 0,		/* tp_dealloc */
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
	_RevoluteJoint_methods,    /* tp_methods */
	0,                          /* tp_members */
	_RevoluteJoint_getseters,	/* tp_getset */
	0,				/* tp_base */
	0,                          /* tp_dict */
	0,                          /* tp_descr_get */
	0,                          /* tp_descr_set */
	0,                          /* tp_dictoffset */
	(initproc)_RevoluteJoint_init,  /* tp_init */
	0,                          /* tp_alloc */
	_RevoluteJointNew,          /* tp_new */
	0,                          /* tp_free */
	0,                          /* tp_is_gc */
	0,                          /* tp_bases */
	0,                          /* tp_mro */
	0,                          /* tp_cache */
	0,                          /* tp_subclasses */
	0,                          /* tp_weaklist */
	0                           /* tp_del */
};

static int _RevoluteJoint_init(PyRevoluteJointObject* joint,PyObject *args,
							   PyObject *kwds)
{
	if(PyJoint_Type.tp_init((PyObject*)joint, args, kwds) < 0)
		return -1;
	return 0;
}

static PyObject* _RevoluteJointNew(PyTypeObject *type, PyObject *args,
								   PyObject *kwds)
{
	PyRevoluteJointObject *joint = (PyRevoluteJointObject*)
		_JointBaseNew (type, args, kwds);
	if (!joint)
		return NULL;

	joint->joint.SolveConstraintVelocity = _SolveRevoluteJointVelocity;
	PyVector2_Set(joint->anchor1,0,0);
	PyVector2_Set(joint->anchor2,0,0);

	return (PyObject*)joint;
}


static void _RevoluteJoint_ComputeOneDynamic (PyBodyObject* body,
											  PyVector2* staticAnchor, PyVector2* localAnchor,
											  double stepTime)
{
	double a,b,k,temp; //for solve equation
	PyVector2 localP = PyBodyObject_GetGlobalPos(body,localAnchor);
	PyVector2 vP = PyBodyObject_GetLocalPointVelocity(body,*localAnchor);
	PyVector2 L = c_diff(*staticAnchor,localP);
	PyVector2 dvBody;
	double dAngleV;

	localP = c_diff(localP,body->vecPosition);
	k = ((PyShapeObject*)body->shape)->rInertia / body->fMass;
	k += PyVector2_GetLengthSquare(localP);
	b = -PyVector2_GetLength(vP);
	/*bb = PyVector2_GetLength(L);*/
	PyVector2_Normalize(&vP);
	temp = PyVector2_Cross(localP,vP);
	a = 1 + PyVector2_Dot(PyVector2_fCross(temp,localP),vP) / k;


	temp = b /a;
	dvBody = PyVector2_MultiplyWithReal(vP,temp);

	body->vecLinearVelocity = c_sum(body->vecLinearVelocity,dvBody);
	dAngleV = PyVector2_Cross(localP,dvBody);
	dAngleV /= k;
	body->fAngleVelocity += dAngleV;

	// Position correction.
	/*temp = -bb /a;
	dvBody = PyVector2_MultiplyWithReal(L,temp);
	dAngleV = PyVector2_Cross(localP,dvBody);
	dAngleV /= k;
	body->vecPosition = c_sum(body->vecPosition,dvBody);
	body->fRotation += dAngleV;*/

	body->vecPosition = c_sum(body->vecPosition,L);

	return;
}

static void _RevoluteJoint_ComputeTwoDynamic(PyDistanceJointObject* joint,
											 double stepTime)
{
	PyBodyObject *body1 = (PyBodyObject*)joint->joint.body1;
	PyBodyObject *body2 = (PyBodyObject*)joint->joint.body2;
	double a1,a2,b1,b2,bb,k1,k2,temp,temp1,temp2; //for solve equation
	PyVector2 localP1 = PyBodyObject_GetGlobalPos(body1,&joint->anchor1);
	PyVector2 localP2 = PyBodyObject_GetGlobalPos(body2,&joint->anchor2);
	PyVector2 L = c_diff(localP1,localP2);
	PyVector2 eL = L;
	PyVector2 vP1 = PyBodyObject_GetLocalPointVelocity(body1,joint->anchor1);
	PyVector2 vP2 = PyBodyObject_GetLocalPointVelocity(body2,joint->anchor2);
	PyVector2 vPL1,vPL2;
	PyVector2 dvBody1,dvBody2;
	double dAngleV1,dAngleV2;
	PyVector2_Normalize(&eL)
		vPL1 = PyVector2_Project(eL,vP1);
	vPL2 = PyVector2_Project(eL,vP2);
	k1 = ((PyShapeObject*)body1->shape)->rInertia / body1->fMass;
	k2 = ((PyShapeObject*)body2->shape)->rInertia / body2->fMass;

	localP1 = c_diff(localP1,body1->vecPosition);
	localP2 = c_diff(localP2,body2->vecPosition);
	k1 += PyVector2_GetLengthSquare(localP1);
	k2 += PyVector2_GetLengthSquare(localP2);

	bb = (joint->distance - PyVector2_GetLength(L)) ;
	PyVector2_Normalize(&L);
	temp = PyVector2_Cross(localP1,L);
	a1 = 1 + PyVector2_Dot(PyVector2_fCross(temp,localP1),L) / k1;

	a1 /= body1->fMass;

	temp = PyVector2_Cross(localP2,L);
	a2 = 1 + PyVector2_Dot(PyVector2_fCross(temp,localP2),L) / k2;

	a2 /= body2->fMass;

	b1 = PyVector2_Dot(vPL1,L);
	b2 = PyVector2_Dot(vPL2,L);

	temp = (b2 - b1) /(a1 + a2);
	temp1 = temp / body1->fMass;
	temp2 = -temp / body2->fMass;
	dvBody1 = PyVector2_MultiplyWithReal(L,temp1);
	dvBody2 = PyVector2_MultiplyWithReal(L,temp2);

	body1->vecLinearVelocity = c_sum(body1->vecLinearVelocity,dvBody1);
	dAngleV1 = PyVector2_Cross(localP1,dvBody1);
	dAngleV1 /= k1;
	body1->fAngleVelocity += dAngleV1;

	body2->vecLinearVelocity = c_sum(body2->vecLinearVelocity,dvBody2);
	dAngleV2 = PyVector2_Cross(localP2,dvBody2);
	dAngleV2 /= k2;
	body2->fAngleVelocity += dAngleV2;

	//for position correction

	temp = bb /(a1 + a2);
	temp1 = temp / body1->fMass;
	temp2 = -temp / body2->fMass;
	dvBody1 = PyVector2_MultiplyWithReal(L,temp1);
	dvBody2 = PyVector2_MultiplyWithReal(L,temp2);

	body1->vecPosition = c_sum(body1->vecPosition,dvBody1);
	body2->vecPosition = c_sum(body2->vecPosition,dvBody2);
	body1->vecLinearVelocity = c_sum(body1->vecLinearVelocity,PyVector2_MultiplyWithReal(dvBody1,1/stepTime));
	body2->vecLinearVelocity = c_sum(body2->vecLinearVelocity,PyVector2_MultiplyWithReal(dvBody2,1/stepTime));
	dAngleV1 = PyVector2_Cross(localP1,dvBody1);
	dAngleV1 /= k1;
	dAngleV2 = PyVector2_Cross(localP2,dvBody2);
	dAngleV2 /= k2;
	body1->fRotation += dAngleV1;
	body2->fRotation += dAngleV2;
	body1->fAngleVelocity += (dAngleV1 / stepTime);
	body2->fAngleVelocity += (dAngleV2 / stepTime);
}

static void _SolveRevoluteJointVelocity(PyJointObject* joint,double stepTime)
{
	PyDistanceJointObject* pJoint = (PyDistanceJointObject*)joint;
	PyBodyObject* body1 = (PyBodyObject*) joint->body1;
	PyBodyObject* body2 = (PyBodyObject*) joint->body2;
	

	if(body1 && body2)
	{
		if (body1->bStatic && body2->bStatic)
		{
			return;
		}
		if (body1->bStatic)
		{
			PyVector2 staticAnchor = PyBodyObject_GetGlobalPos(body1,&pJoint->anchor1);
			_RevoluteJoint_ComputeOneDynamic(body2,&staticAnchor,&pJoint->anchor2,stepTime);
			return;
		}
		if (body2->bStatic)
		{
			PyVector2 staticAnchor = PyBodyObject_GetGlobalPos(body2,&pJoint->anchor2);
			_RevoluteJoint_ComputeOneDynamic(body1,&staticAnchor,&pJoint->anchor1,stepTime);
			return;
		}
		_RevoluteJoint_ComputeTwoDynamic(pJoint,stepTime);

	}
}

static int _RevoluteJoint_setAnchor(PyRevoluteJointObject* joint,
									PyObject* value,void* closure)
{
	PyObject *item;
	PyVector2 global_p;

	if (!PySequence_Check(value) || PySequence_Size (value) != 2)
	{
		PyErr_SetString (PyExc_TypeError, "anchor must be a x, y sequence");
		return -1;
	}

	item = PySequence_GetItem (value, 0);
	if (!DoubleFromObj (item, &global_p.real))
		return -1;
	item = PySequence_GetItem (value, 1);
	if (!DoubleFromObj (item, &global_p.imag))
		return -1;

	if (joint->joint.body1)
	{
		joint->anchor1 = PyBodyObject_GetRelativePosFromGlobal(joint->joint.body1,&global_p);
	}
	else
	{
		joint->anchor1 = global_p;
	}

	if (joint->joint.body2)
	{
		joint->anchor2 = PyBodyObject_GetRelativePosFromGlobal(joint->joint.body2,&global_p);
	}
	else
	{
		joint->anchor2 = global_p;
	}

	return 0;
}

static PyObject* _RevoluteJoint_getAnchor(PyRevoluteJointObject* joint,
										  void* closure)
{
	PyVector2 global_p;
	if (joint->joint.body1)
	{
		global_p = PyBodyObject_GetGlobalPos(joint->joint.body1,&joint->anchor1);
	}
	else if(joint->joint.body2)
	{
		global_p = PyBodyObject_GetGlobalPos(joint->joint.body2,&joint->anchor2);
	}
	else
	{
		assert(false);
	}
	return Py_BuildValue ("(ff)", global_p.real, global_p.imag);
}

static PyObject* _RevoluteJoint_getPointList(PyRevoluteJointObject* joint,void* closure)
{
	PyObject* list  = PyList_New(2);

	PyVector2 p = PyBodyObject_GetGlobalPos(((PyBodyObject*)joint->joint.body1),&joint->anchor1);
	PyObject* tuple = FromPhysicsVector2ToPoint(p);
	PyList_SetItem(list,0,tuple);

	if(joint->joint.body2)
	{
		p = PyBodyObject_GetGlobalPos(((PyBodyObject*)joint->joint.body2),&joint->anchor2);

	}
	else
	{
		p = joint->anchor2;
	}

	tuple = FromPhysicsVector2ToPoint(p);
	PyList_SetItem(list,1,tuple);
	return list;
}

void PyJointObject_ExportCAPI (void **c_api)
{
    c_api[PHYSICS_JOINT_FIRSTSLOT] = &PyJoint_Type;
    c_api[PHYSICS_JOINT_FIRSTSLOT + 1] = &PyJoint_New;
    c_api[PHYSICS_JOINT_FIRSTSLOT + 2] = &PyDistanceJoint_Type;
    c_api[PHYSICS_JOINT_FIRSTSLOT + 3] = &PyDistanceJoint_New;
    c_api[PHYSICS_JOINT_FIRSTSLOT + 4] = &PyDistanceJoint_SetAnchors;
}
