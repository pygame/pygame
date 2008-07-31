#include "pgJointObject.h"
#include "pgShapeObject.h"
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
	Py_INCREF(op);
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
	static char *kwlist2[] = {"body1", "isCollideConnect", NULL};
	if(!PyArg_ParseTupleAndKeywords(args,kwds,"OOi",kwlist,&body1,&body2,&bCollide))
	{
		if(!PyArg_ParseTupleAndKeywords(args,kwds,"Oi",kwlist2,&body1,&bCollide))
			return -1;
		else
		{
			body2 = NULL;
		}
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
	if (joint->body1)
	{
		return (PyObject*)joint->body1;
	}
	else
	{
		PyErr_SetString(PyExc_ValueError,"body1 is NULL!");
		return NULL;
	}
	
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
	if (joint->body2)
	{
		return (PyObject*)joint->body2;
	}
	else
	{
		PyErr_SetString(PyExc_ValueError,"body2 is NULL!");
		return NULL;
	}
}

static PyGetSetDef _pgJointBase_getseters[] = {
	{
		"body1",(getter)_pgJoint_getBody1,(setter)_pgJoint_setBody1,"",NULL,
	},
	{
		"body2",(getter)_pgJoint_getBody2,(setter)_pgJoint_setBody2,"",NULL,
	},
	{
            NULL, NULL, NULL, NULL, NULL
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

void _PG_DistanceJoint_ComputeOneDynamic(pgBodyObject* body,pgVector2* staticAnchor,pgVector2* localAnchor,double distance,double stepTime)
{
	/*double a,b,c,d,e,f,k;
	pgVector2 localP = PG_GetGlobalPos(body,localAnchor);
	pgVector2 L = c_diff(localP,*staticAnchor);
	pgVector2 vP = PG_GetLocalPointVelocity(body,*localAnchor);
	pgVector2 vPL = c_project(L,vP);
	pgVector2 dvBody;
	double dAngleV;
	vPL = c_neg(vPL);

	localP = c_diff(localP,body->vecPosition);
	k = body->shape->rInertia / body->fMass;



	a = (1 - localP.real * localP.imag / k);
	b = (localP.imag * localP.imag / k);
	c = (localP.real * localP.real /k);
	d = a;
	e = vPL.real;
	f = vPL.imag;

	dvBody.imag = (e*c - a*f) / (b*c - a*d);
	dvBody.real = (e*d - f*b) / (a*d - b*c);

	body->vecLinearVelocity = c_sum(body->vecLinearVelocity,dvBody);
	dAngleV = c_cross(localP,dvBody);
	dAngleV /= k;
	body->fRotation += dAngleV;*/

	double a,b,bb,k,temp; //for solve equation
	pgVector2 localP = PG_GetGlobalPos(body,localAnchor);
	pgVector2 L = c_diff(*staticAnchor,localP);
	pgVector2 vP = PG_GetLocalPointVelocity(body,*localAnchor);
	pgVector2 vPL = c_project(L,vP);
	pgVector2 dvBody;
	double dAngleV;
	//vPL = c_neg(vPL);

	localP = c_diff(localP,body->vecPosition);
	k = body->shape->rInertia / body->fMass;
	k *= 2;
	k += c_get_length_square(localP);
	bb = (distance - c_get_length(L));
	c_normalize(&L);
	temp = c_cross(localP,L);
	a = 1 + c_dot(c_fcross(temp,localP),L) / k;
	/*lengthP = c_get_length(localP);
	if (lengthP < 1e-5)
	{
		a = 1;
	}
	else
	{
		a = (1 + temp * temp / (k * lengthP));
	}*/
	
	b = -c_dot(vPL,L);
	

	temp = b /a;
	dvBody = c_mul_complex_with_real(L,temp);

	body->vecLinearVelocity = c_sum(body->vecLinearVelocity,dvBody);
	dAngleV = c_cross(localP,dvBody);
	dAngleV /= k;
	body->fAngleVelocity += dAngleV;


	//for position correction
	
	temp = -bb /a;
	dvBody = c_mul_complex_with_real(L,temp);
	dAngleV = c_cross(localP,dvBody);
	dAngleV /= k;
	body->vecPosition = c_sum(body->vecPosition,dvBody);
	body->fRotation += dAngleV;

	/*temp = c_get_length(c_diff(*staticAnchor,PG_GetGlobalPos(body,localAnchor)));
	temp -= distance; 
	printf("%f\n",temp);*/

	return;
}

void _PG_DistanceJoint_ComputeTwoDynamic(pgDistanceJointObject* joint,double stepTime)
{
	pgBodyObject *body1 = joint->joint.body1,*body2 = joint->joint.body2;
	double a1,a2,b1,b2,bb,k1,k2,temp,temp1,temp2; //for solve equation
	pgVector2 localP1 = PG_GetGlobalPos(body1,&joint->anchor1);
	pgVector2 localP2 = PG_GetGlobalPos(body2,&joint->anchor2);
	pgVector2 L = c_diff(localP1,localP2);
	pgVector2 vP1 = PG_GetLocalPointVelocity(body1,joint->anchor1);
	pgVector2 vP2 = PG_GetLocalPointVelocity(body2,joint->anchor2);
	pgVector2 vPL1 = c_project(L,vP1);
	pgVector2 vPL2 = c_project(L,vP2);
	pgVector2 dvBody1,dvBody2;
	double dAngleV1,dAngleV2;
	k1 = body1->shape->rInertia / body1->fMass;
	k1 *= 2;
	k2 = body2->shape->rInertia / body2->fMass;
	k2 *= 2;

	localP1 = c_diff(localP1,body1->vecPosition);
	localP2 = c_diff(localP2,body2->vecPosition);
	k1 += c_get_length_square(localP1);
	k2 += c_get_length_square(localP2);

	bb = (joint->distance - c_get_length(L)) ;
	c_normalize(&L);
	temp = c_cross(localP1,L);
	a1 = 1 + c_dot(c_fcross(temp,localP1),L) / k1;

	a1 /= body1->fMass;

	temp = c_cross(localP2,L);
	a2 = 1 + c_dot(c_fcross(temp,localP2),L) / k2;

	a2 /= body2->fMass;

	b1 = c_dot(vPL1,L);
	b2 = c_dot(vPL2,L);

	temp = (b2 - b1) /(a1 + a2);
	temp1 = temp / body1->fMass;
	temp2 = -temp / body2->fMass;
	dvBody1 = c_mul_complex_with_real(L,temp1);
	dvBody2 = c_mul_complex_with_real(L,temp2);

	body1->vecLinearVelocity = c_sum(body1->vecLinearVelocity,dvBody1);
	dAngleV1 = c_cross(localP1,dvBody1);
	dAngleV1 /= k1;
	body1->fAngleVelocity += dAngleV1;

	body2->vecLinearVelocity = c_sum(body2->vecLinearVelocity,dvBody2);
	dAngleV2 = c_cross(localP2,dvBody2);
	dAngleV2 /= k2;
	body2->fAngleVelocity += dAngleV2;

	//for position correction
	temp = bb /(a1 + a2);
	temp1 = temp / body1->fMass;
	temp2 = -temp / body2->fMass;
	dvBody1 = c_mul_complex_with_real(L,temp1);
	dvBody2 = c_mul_complex_with_real(L,temp2);

	body1->vecPosition = c_sum(body1->vecPosition,dvBody1);
	body2->vecPosition = c_sum(body2->vecPosition,dvBody2);
	body1->vecLinearVelocity = c_sum(body1->vecLinearVelocity,c_mul_complex_with_real(dvBody1,1/stepTime));
	body2->vecLinearVelocity = c_sum(body2->vecLinearVelocity,c_mul_complex_with_real(dvBody2,1/stepTime));
	dAngleV1 = c_cross(localP1,dvBody1);
	dAngleV1 /= k1;
	dAngleV2 = c_cross(localP2,dvBody2);
	dAngleV2 /= k2;
	body1->fRotation += dAngleV1;
	body2->fRotation += dAngleV2;
	body1->fAngleVelocity += (dAngleV1 / stepTime);
	body2->fAngleVelocity += (dAngleV2 / stepTime);
	temp = c_get_length(c_diff(PG_GetGlobalPos(body1,&joint->anchor1),PG_GetGlobalPos(body2,&joint->anchor2)));
	temp -= joint->distance; 
	printf("%f\n",temp);


	////body1->cBiasLV = c_sum(body1->cBiasLV,dvBody1);
	//body1->cBiasLV = dvBody1;
	//dAngleV1 = c_cross(localP1,dvBody1);
	//dAngleV1 /= k1;
	//body1->cBiasW = dAngleV1;

	////body2->cBiasLV = c_sum(body2->cBiasLV,dvBody2);
	//body2->cBiasLV = dvBody2;
	//dAngleV2 = c_cross(localP2,dvBody2);
	//dAngleV2 /= k2;
	//body2->cBiasW = dAngleV2;

	//PG_CorrectBodyPos(body1,stepTime);
	//PG_CorrectBodyPos(body2,stepTime);
}

void PG_SolveDistanceJointVelocity(pgJointObject* joint,double stepTime)
{
	/*pgVector2 vecL;
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
		joint->body1->vecLinearVelocity = c_diff(joint->body1->vecLinearVelocity, v1Add);
		joint->body2->vecLinearVelocity = c_sum(joint->body2->vecLinearVelocity, v2Add);
		return;
	}*/

	pgDistanceJointObject* pJoint = (pgDistanceJointObject*)joint;
	pgBodyObject* body1 = joint->body1;
	pgBodyObject* body2 = joint->body2;
	if (body1 && (!body2))
	{
		if (body1->bStatic)
		{
			return;
		}
		else
		{
			_PG_DistanceJoint_ComputeOneDynamic(body1,&pJoint->anchor2,&pJoint->anchor1,pJoint->distance,stepTime);

			/*double a,b,c,d,e,f,k;
			pgVector2 localP = PG_GetGlobalPos(body1,&pJoint->anchor1);
			pgVector2 L = c_diff(localP,pJoint->anchor2);
			pgVector2 vP = PG_GetLocalPointVelocity(body1,pJoint->anchor1);
			pgVector2 vPL = c_project(L,vP);
			pgVector2 dvBody;
			double dAngleV;
			vPL = c_neg(vPL);

			localP = c_diff(localP,body1->vecPosition);
			k = body1->shape->rInertia / body1->fMass;


			
			a = (1 - localP.real * localP.imag / k);
			b = (localP.imag * localP.imag / k);
			c = (localP.real * localP.real /k);
			d = a;
			e = vPL.real;
			f = vPL.imag;
			
			dvBody.imag = (e*c - a*f) / (b*c - a*d);
			dvBody.real = (e*d - f*b) / (a*d - b*c);

			body1->vecLinearVelocity = c_sum(body1->vecLinearVelocity,dvBody);
			dAngleV = c_cross(localP,dvBody);
			dAngleV /= k;
			body1->fRotation += dAngleV;*/


			return;
		}
	}

	if(body1 && body2)
	{
		if (body1->bStatic && body2->bStatic)
		{
			return;
		}
		if (body1->bStatic)
		{
			pgVector2 staticAnchor = PG_GetGlobalPos(body1,&pJoint->anchor1);
			_PG_DistanceJoint_ComputeOneDynamic(body2,&staticAnchor,&pJoint->anchor2,pJoint->distance,stepTime);
			return;
		}
		if (body2->bStatic)
		{
			pgVector2 staticAnchor = PG_GetGlobalPos(body2,&pJoint->anchor2);
			_PG_DistanceJoint_ComputeOneDynamic(body1,&staticAnchor,&pJoint->anchor1,pJoint->distance,stepTime);
			return;
		}
		_PG_DistanceJoint_ComputeTwoDynamic(pJoint,stepTime);
				
	}
}

//just for C test usage, not for python???
pgJointObject* PG_DistanceJointNew(pgBodyObject* b1,pgBodyObject* b2,int bCollideConnect,double dist,pgVector2 a1,pgVector2 a2)
{
	//pgDistanceJointObject* pjoint = (pgDistanceJointObject*)PyObject_MALLOC(sizeof(pgDistanceJointObject));	
	pgDistanceJointObject* pjoint = (pgDistanceJointObject*)PyObject_MALLOC(sizeof(pgDistanceJointObject));
	PG_InitJointBase(&(pjoint->joint), b1, b2, bCollideConnect);
	//pjoint->distance = dist;
	if (b1 && b2)
	{
		pgVector2 s1 = PG_GetGlobalPos(b1,&a1);
		pgVector2 s2 = PG_GetGlobalPos(b2,&a2);
		pjoint->distance = c_get_length(c_diff(s1,s2));
	}
	else
	{
		pgVector2 s1 = PG_GetGlobalPos(b1,&a1);
		pjoint->distance = c_get_length(c_diff(s1,a2));
	}
	pjoint->anchor1 = a1;
	pjoint->anchor2 = a2;
	pjoint->joint.SolveConstraintVelocity = PG_SolveDistanceJointVelocity;
	return (pgJointObject*)pjoint;
}

static PyObject* _pgDistanceJoint_getDistance(pgDistanceJointObject* joint,void* closure)
{
	return PyFloat_FromDouble(joint->distance);
}

static int _pgDistanceJoint_setDistance(pgDistanceJointObject* joint,PyObject* value,void* closure)
{
	if(PyFloat_Check(value))
	{
		joint->distance = PyFloat_AsDouble(value);
		return 0;
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "value must be float number");
		return -1;
	}
}

static PyObject* _pgDistanceJoint_getAnchor1(pgDistanceJointObject* joint,void* closure)
{
	return PyComplex_FromCComplex(joint->anchor1);
}

static int _pgDistanceJoint_setAnchor1(pgDistanceJointObject* joint,PyObject* value,void* closure)
{
	if (PyComplex_Check(value))
	{
		joint->anchor1 = PyComplex_AsCComplex(value);
		return 0;
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "value must be complex number");
		return -1;
	}
}

static PyObject* _pgDistanceJoint_getAnchor2(pgDistanceJointObject* joint,void* closure)
{
	return PyComplex_FromCComplex(joint->anchor2);
}

static int _pgDistanceJoint_setAnchor2(pgDistanceJointObject* joint,PyObject* value,void* closure)
{
	if (PyComplex_Check(value))
	{
		joint->anchor2 = PyComplex_AsCComplex(value);
		return 0;
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "value must be complex number");
		return -1;
	}
}

static PyObject* _pgDistanceJoint_getPointList(PyObject *self, PyObject *args)
{
	pgDistanceJointObject* joint = (pgDistanceJointObject*)self;
	PyObject* list  = PyList_New(2);

	pgVector2 p = PG_GetGlobalPos(joint->joint.body1,&joint->anchor1);
	PyObject* tuple = FromPhysicsVector2ToPygamePoint(p);
	PyList_SetItem(list,0,tuple);

	if(joint->joint.body2)
	{
		p = PG_GetGlobalPos(joint->joint.body2,&joint->anchor2);
		
	}
	else
	{
		p = joint->anchor2;
	}

	tuple = FromPhysicsVector2ToPygamePoint(p);
	PyList_SetItem(list,1,tuple);
	return list;
}

static PyMethodDef _pgDistanceJoint_methods[] = {
	{"get_point_list",_pgDistanceJoint_getPointList,METH_VARARGS,""	},
	{NULL, NULL, 0, NULL}   /* Sentinel */
};

static PyMemberDef _pgDistanceJoint_members[] = 
{
	
    {	NULL, 0, 0, 0, NULL}
}; 

static PyGetSetDef _pgDistanceJoint_getseters[] = {
	{
		"distance",(getter)_pgDistanceJoint_getDistance,(setter)_pgDistanceJoint_setDistance,"",NULL,
	},
	{
		"anchor1",(getter)_pgDistanceJoint_getAnchor1,(setter)_pgDistanceJoint_setAnchor1,"",NULL,
	},
	{
		"anchor2",(getter)_pgDistanceJoint_getAnchor2,(setter)_pgDistanceJoint_setAnchor2,"",NULL,
	},
	{
            NULL, NULL, NULL, NULL, NULL
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
	_pgDistanceJoint_methods,							/* tp_methods */
	_pgDistanceJoint_members,	/* tp_members */
	_pgDistanceJoint_getseters,	/* tp_getset */
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

