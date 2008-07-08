#include "pgBodyObject.h"
#include "pgWorldObject.h"
#include "pgVector2.h"
#include "pgShapeObject.h"
#include <structmember.h>

extern PyTypeObject pgBodyType;

void PG_Bind_RectShape(pgBodyObject* body, double width, double height, double seta)
{
	if(body->shape == NULL)
		body->shape = PG_RectShapeNew(body, width, height, seta);
	else
	{
		Py_DECREF(body->shape);
		body->shape = PG_RectShapeNew(body, width, height, seta);
	}
}

void PG_FreeUpdateBodyVel(pgWorldObject* world,pgBodyObject* body, double dt)
{
	pgVector2 totalVelAdd;
	pgVector2 totalF;
	double k;
	if(body->bStatic) return;

	totalF = c_sum(body->vecForce, world->vecGravity);
	k = dt / body->fMass;
	totalVelAdd = c_mul_complex_with_real(totalF, k);
	body->vecLinearVelocity = c_sum(body->vecLinearVelocity, totalVelAdd);
}

void PG_FreeUpdateBodyPos(pgWorldObject* world,pgBodyObject* body,double dt)
{
	pgVector2 totalPosAdd;

	if(body->bStatic) return;

	totalPosAdd = c_mul_complex_with_real(body->vecLinearVelocity, dt);
	body->vecPosition = c_sum(body->vecPosition,totalPosAdd);
	body->fRotation += body->fAngleVelocity*dt;
}

void PG_BodyInit(pgBodyObject* body)
{
	body->fAngleVelocity = 0.0;
	body->fFriction = 0.0;
	body->fMass = 1.0;
	body->fRestitution = 1.0;
	body->fRotation = 0.0;
	body->fTorque = 0.0;
	body->shape = NULL;
	PG_Set_Vector2(body->vecForce,0.0,0.0);
	PG_Set_Vector2(body->vecImpulse,0.0,0.0);
	PG_Set_Vector2(body->vecLinearVelocity,0.0,0.0);
	PG_Set_Vector2(body->vecPosition,0.0,0.0);
}

PyObject* _PG_BodyNew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	//TODO: parse args later on
	pgBodyObject* op;
	if(PyType_Ready(type)==-1) return NULL;
	op = (pgBodyObject*)type->tp_alloc(type, 0);
	PG_BodyInit(op);
	return (PyObject*)op;
}

void PG_BodyDestroy(pgBodyObject* body)
{
	/*
	* DECREF anything related to the Body, such as the lists and
	* release any other memory hold by it.
	*/

	//delete shape
	Py_DECREF(body->shape);
	body->ob_type->tp_free((PyObject*)body);
}



pgBodyObject* PG_BodyNew()
{
	return (pgBodyObject*) _PG_BodyNew(&pgBodyType, NULL, NULL);
}

pgVector2 PG_GetGlobalPos(pgBodyObject* body, pgVector2* local_p)
{
	pgVector2 ans;
	
	ans = *local_p;
	c_rotate(&ans, body->fRotation);
	ans = c_sum(ans, body->vecPosition);

	return ans;
}


pgVector2 PG_GetRelativePos(pgBodyObject* bodyA, pgBodyObject* bodyB, pgVector2* p_in_B)
{
	pgVector2 trans, p_in_A;
	double rotate;
	
	trans = c_diff(bodyB->vecPosition, bodyA->vecPosition);
	c_rotate(&trans, -bodyA->fRotation);
	rotate = bodyA->fRotation - bodyB->fRotation;
	p_in_A = *p_in_B;
	c_rotate(&p_in_A, -rotate);
	p_in_A = c_sum(p_in_A, trans);
	
	return p_in_A;
}

//============================================================
//getter and setter functions

//velocity
static PyObject* _pgBody_getVelocity(pgBodyObject* body,void* closure)
{
	return PyComplex_FromCComplex(body->vecLinearVelocity);
}

static int _pgBody_setVelocity(pgBodyObject* body,PyObject* value,void* closure)
{
	if (value == NULL || (!PyComplex_Check(value))) {
		PyErr_SetString(PyExc_TypeError, "Cannot set the velocity attribute");
		return -1;
	}
	else
	{
		body->vecLinearVelocity = PyComplex_AsCComplex(value);
		return 0;
	}
}

//position
static PyObject* _pgBody_getPosition(pgBodyObject* body,void* closure)
{
	return PyComplex_FromCComplex(body->vecPosition);
}

static int _pgBody_setPosition(pgBodyObject* body,PyObject* value,void* closure)
{
	if (value == NULL || (!PyComplex_Check(value))) {
		PyErr_SetString(PyExc_TypeError, "Cannot set the position attribute");
		return -1;
	}
	else
	{
		body->vecPosition = PyComplex_AsCComplex(value);
		return 0;
	}
}

//force
static PyObject* _pgBody_getForce(pgBodyObject* body,void* closure)
{
	return PyComplex_FromCComplex(body->vecForce);
}

static int _pgBody_setForce(pgBodyObject* body,PyObject* value,void* closure)
{
	if (value == NULL || (!PyComplex_Check(value))) {
		PyErr_SetString(PyExc_TypeError, "Cannot set the force attribute");
		return -1;
	}
	else
	{
		body->vecForce = PyComplex_AsCComplex(value);
		return 0;
	}
}

static PyObject* _pgBody_bindRectShape(PyObject* body,PyObject* args)
{
	double width,height,seta;
	if (!PyArg_ParseTuple(args,"ddd",&width,&height,&seta))
	{
		//printf("fail\n"); //for test
		PyErr_SetString(PyExc_ValueError,"parameters are wrong");
		return NULL;
	}
	else
	{
		PG_Bind_RectShape((pgBodyObject*)body,width,height,seta);
		if (((pgBodyObject*)body)->shape == NULL)
		{
			PyErr_SetString(PyExc_ValueError,"shape binding is failed");
			return NULL;
		}
		else
		{
			//printf("%d\n",((pgBodyObject*)body)->shape);
			Py_RETURN_NONE;
			//return ((pgBodyObject*)body)->shape;
		}
	}
}

//===============================================================


static PyMemberDef _pgBody_members[] = {
	{"mass", T_DOUBLE, offsetof(pgBodyObject,fMass), 0,"Mass"},
	{"rotation", T_DOUBLE, offsetof(pgBodyObject,fRotation), 0,"Rotation"},
	{"torque",T_DOUBLE,offsetof(pgBodyObject,fTorque),0,""},
	{"restitution",T_DOUBLE,offsetof(pgBodyObject,fRestitution),0,""},
	{"friction",T_DOUBLE,offsetof(pgBodyObject,fFriction),0,""},
    {NULL}  /* Sentinel */
};

static PyMethodDef _pgBody_methods[] = {
	{"bind_rect_shape",_pgBody_bindRectShape,METH_VARARGS,""},
    {NULL, NULL, 0, NULL}   /* Sentinel */
};

static PyGetSetDef _pgBody_getseters[] = {
	{"velocity",(getter)_pgBody_getVelocity,(setter)_pgBody_setVelocity,"velocity",NULL},
	{"position",(getter)_pgBody_getPosition,(setter)_pgBody_setPosition,"position",NULL},
	{"force",(getter)_pgBody_getForce,(setter)_pgBody_setForce,"force",NULL},
	{ NULL}
};

PyTypeObject pgBodyType =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"physics.Body",            /* tp_name */
	sizeof(pgBodyObject),      /* tp_basicsize */
	0,                          /* tp_itemsize */
	(destructor)PG_BodyDestroy,/* tp_dealloc */
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
	_pgBody_methods,		   	/* tp_methods */
	_pgBody_members,            /* tp_members */
	_pgBody_getseters,          /* tp_getset */
	0,                          /* tp_base */
	0,                          /* tp_dict */
	0,                          /* tp_descr_get */
	0,                          /* tp_descr_set */
	0,                          /* tp_dictoffset */
	0,                          /* tp_init */
	0,                          /* tp_alloc */
	_PG_BodyNew,                /* tp_new */
	0,                          /* tp_free */
	0,                          /* tp_is_gc */
	0,                          /* tp_bases */
	0,                          /* tp_mro */
	0,                          /* tp_cache */
	0,                          /* tp_subclasses */
	0,                          /* tp_weaklist */
	0                           /* tp_del */
};


