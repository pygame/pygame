#include "pgJointObject.h"
#include <structmember.h>

extern PyTypeObject pgDistanceJointType;

void PG_InitJointBase(pgJointObject* joint,pgBodyObject* b1,pgBodyObject* b2,int bCollideConnect)
{
	joint->body1 = b1;
	joint->body2 = b2;
	joint->isCollideConnect = bCollideConnect;
	joint->SolveConstraintVelocity = NULL;
	joint->SolveConstraintPosition = NULL;
	joint->Destroy = NULL;
}

pgJointObject* _PG_JointNewInternal(PyTypeObject *type)
{
	pgJointObject* op = (pgJointObject*)type->tp_alloc(type, 0);
	//PG_InitJointBase(op);
	return op;
}

//TODO: this function would get err when inherited level > 2
void PG_JointDestroy(pgJointObject* joint)
{
	if (joint->Destroy)
	{
		joint->Destroy(joint);
	}

	joint->ob_type->tp_free((PyObject*)joint);
}

PyObject* _PG_JointBaseNew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	/* In case we have arguments in the python code, parse them later
	* on.
	*/
	PyObject* joint;
	if(PyType_Ready(type)==-1) return NULL;
	
	joint = (PyObject*) _PG_JointNewInternal(type);
	
	return joint;
}

static int _pgJointBase_init(pgJointObject* joint,PyObject *args, PyObject *kwds)
{
	PyObject* body1, *body2;
	int bCollide;
	static char *kwlist[] = {"body1", "body2", "isCollideConnect", NULL};
	if(!PyArg_ParseTupleAndKeywords(args,kwds,"|OOi",kwlist,&body1,&body2,&bCollide))
	{
		return -1;
	}
	PG_InitJointBase(joint,(pgBodyObject*)body1,(pgBodyObject*)body2,bCollide);
	return 0;
}

static int _pgJoint_setBody1(pgJointObject* joint,PyObject* value,void* closure)
{
	if(value == NULL)
	{
		PyErr_SetString(PyExc_TypeError, "Cannot set the body1 attribute");
		return -1;
	}
	else
	{
		joint->body1 = (pgBodyObject*)value;
		return 0;
	}
}

static PyObject* _pgJoint_getBody1(pgJointObject* joint,void* closure)
{
	return (PyObject*)joint->body1;
}

static int _pgJoint_setBody2(pgJointObject* joint,PyObject* value,void* closure)
{
	if(value == NULL)
	{
		PyErr_SetString(PyExc_TypeError, "Cannot set the body2 attribute");
		return -1;
	}
	else
	{
		joint->body2 = (pgBodyObject*)value;
		return 0;
	}
}

static PyObject* _pgJoint_getBody2(pgJointObject* joint,void* closure)
{
	return (PyObject*)joint->body2;
}

static PyGetSetDef _pgJointBase_getseters[] = {
	{
		"body1",(getter)_pgJoint_getBody1,(setter)_pgJoint_setBody1,"",NULL,
	},
	{
		"body2",(getter)_pgJoint_getBody2,(setter)_pgJoint_setBody2,"",NULL,
	},
	{
		NULL
	}
};



PyTypeObject pgJointType =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"physics.Joint",            /* tp_name */
	sizeof(pgJointObject),      /* tp_basicsize */
	0,                          /* tp_itemsize */
	(destructor) PG_JointDestroy,/* tp_dealloc */
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
	0,							/* tp_methods */
	0,							/* tp_members */
	_pgJointBase_getseters,     /* tp_getset */
	0,                          /* tp_base */
	0,                          /* tp_dict */
	0,                          /* tp_descr_get */
	0,                          /* tp_descr_set */
	0,                          /* tp_dictoffset */
	(initproc)_pgJointBase_init, /* tp_init */
	0,                          /* tp_alloc */
	_PG_JointBaseNew,           /* tp_new */
	0,                          /* tp_free */
	0,                          /* tp_is_gc */
	0,                          /* tp_bases */
	0,                          /* tp_mro */
	0,                          /* tp_cache */
	0,                          /* tp_subclasses */
	0,                          /* tp_weaklist */
	0                           /* tp_del */
};

void PG_DistanceJointInit(pgDistanceJointObject* joint)
{
	joint->distance = 10.0;
	PG_Set_Vector2(joint->anchor1,0,0);
	PG_Set_Vector2(joint->anchor2,0,0);
}


static int _pgDistanceJoint_init(pgDistanceJointObject* joint,PyObject *args, PyObject *kwds)
{
	if(pgJointType.tp_init((PyObject*)joint, args, kwds) < 0)
	{
		return -1;
	}
	PG_DistanceJointInit(joint);
	return 0;
}



void PG_SolveDistanceJointPosition(pgJointObject* joint,double stepTime)
{
	pgVector2 vecL,vecP;
	pgDistanceJointObject* pJoint = (pgDistanceJointObject*)joint;
	
	if (joint->body1 && (!joint->body2))
	{
		vecL = c_diff(joint->body1->vecPosition,pJoint->anchor2);
		c_normalize(&vecL);
		vecL = c_mul_complex_with_real(vecL,pJoint->distance);
		joint->body1->vecPosition = c_sum(pJoint->anchor2,vecL);
		return;
	} 

	if(joint->body1 && joint->body2)
	{
		vecL = c_diff(joint->body1->vecPosition,joint->body2->vecPosition);
		vecP = c_sum(joint->body1->vecPosition,joint->body2->vecPosition);
		c_normalize(&vecL);
		vecL = c_mul_complex_with_real(vecL,pJoint->distance * 0.5);
		joint->body1->vecPosition = c_sum(joint->body1->vecPosition,vecL);
		joint->body2->vecPosition = c_diff(joint->body2->vecPosition,vecL);
		return;
	}
}

void PG_SolveDistanceJointVelocity(pgJointObject* joint,double stepTime)
{
	pgVector2 vecL;
	double lamda,cosTheta1V,cosTheta2V,mk;
	pgVector2 impuseAdd,v1Add,v2Add;
	pgDistanceJointObject* pJoint = (pgDistanceJointObject*)joint;
	if (joint->body1 && (!joint->body2))
	{
		vecL = c_diff(joint->body1->vecPosition, pJoint->anchor2);
		c_normalize(&vecL);
		lamda = -c_dot(vecL, joint->body1->vecLinearVelocity);
		vecL = c_mul_complex_with_real(vecL, lamda);
		joint->body1->vecLinearVelocity = c_sum(joint->body1->vecLinearVelocity, vecL);
		return;
	}

	if(joint->body1 && joint->body2)
	{
		vecL = c_diff(joint->body1->vecPosition, joint->body2->vecPosition);
		c_normalize(&vecL);
		cosTheta1V = c_dot(vecL, joint->body1->vecLinearVelocity);
		cosTheta2V = c_dot(vecL, joint->body2->vecLinearVelocity);
		lamda = cosTheta1V - cosTheta2V;
		mk = joint->body1->fMass * joint->body2->fMass / (joint->body1->fMass + joint->body2->fMass);
		lamda *= mk;
		impuseAdd = c_mul_complex_with_real(vecL, lamda);
		v1Add = c_div_complex_with_real(impuseAdd, joint->body1->fMass);
		v2Add = c_div_complex_with_real(impuseAdd, joint->body2->fMass);
		/*joint->body1->vecLinearVelocity = c_sum(joint->body1->vecLinearVelocity,v1Add);
		joint->body2->vecLinearVelocity = c_diff(joint->body2->vecLinearVelocity,v2Add);*/
		joint->body1->vecLinearVelocity = c_diff(joint->body1->vecLinearVelocity, v1Add);
		joint->body2->vecLinearVelocity = c_sum(joint->body2->vecLinearVelocity, v2Add);
		return;
	}
}

//just for C test usage, not for python???
pgJointObject* PG_DistanceJointNew(pgBodyObject* b1,pgBodyObject* b2,int bCollideConnect,double dist,pgVector2 a1,pgVector2 a2)
{
	pgDistanceJointObject* pjoint = (pgDistanceJointObject*)PyObject_MALLOC(sizeof(pgDistanceJointObject));	
	//pgDistanceJointObject* pjoint = (pgDistanceJointObject*)PyObject_MALLOC(sizeof(pgDistanceJointObject));
	PG_InitJointBase(&(pjoint->joint), b1, b2, bCollideConnect);
	pjoint->distance = dist;
	pjoint->anchor1 = a1;
	pjoint->anchor2 = a2;
	pjoint->joint.SolveConstraintVelocity = PG_SolveDistanceJointVelocity;
	return (pgJointObject*)pjoint;
}

static PyMemberDef _pgDistanceJoint_members[] = 
{
	{"distance",T_DOUBLE,offsetof(pgDistanceJointObject,distance),0,""},
	{
		NULL
	}
}; 



PyTypeObject pgDistanceJointType =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"physics.DistanceJoint",            /* tp_name */
	sizeof(pgDistanceJointObject),      /* tp_basicsize */
	0,                          /* tp_itemsize */
	(destructor) 0,				/* tp_dealloc */
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
	0,							/* tp_methods */
	_pgDistanceJoint_members,	/* tp_members */
	0,							/* tp_getset */
	0,							/* tp_base */
	0,                          /* tp_dict */
	0,                          /* tp_descr_get */
	0,                          /* tp_descr_set */
	0,                          /* tp_dictoffset */
	(initproc)_pgDistanceJoint_init,  /* tp_init */
	0,                          /* tp_alloc */
	0,							/* tp_new */
	0,                          /* tp_free */
	0,                          /* tp_is_gc */
	0,                          /* tp_bases */
	0,                          /* tp_mro */
	0,                          /* tp_cache */
	0,                          /* tp_subclasses */
	0,                          /* tp_weaklist */
	0                           /* tp_del */
};

