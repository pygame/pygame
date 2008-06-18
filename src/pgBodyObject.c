#include "pgBodyObject.h"
#include "pgWorldObject.h"
#include "pgVector2.h"
#include "pgShapeObject.h"
#include <structmember.h>



void PG_FreeUpdateBodyVel(pgWorldObject* world,pgBodyObject* body, double dt)
{
	pgVector2 totalVelAdd;
	double k;
	totalVelAdd = c_sum(body->vecForce,world->vecGravity);
	k = dt / body->fMass;
	totalVelAdd = c_mul_complex_with_real(totalVelAdd,k);
	body->vecLinearVelocity = c_sum(body->vecLinearVelocity,totalVelAdd);
}

void PG_FreeUpdateBodyPos(pgWorldObject* world,pgBodyObject* body,double dt)
{
	pgVector2 totalPosAdd;

	//totalVelAdd = c_div_complex_with_real(body->vecImpulse,body->fMass);
	//body->vecLinearVelocity = c_sum(body->vecLinearVelocity,totalVelAdd);

	totalPosAdd = c_mul_complex_with_real(body->vecLinearVelocity,dt);
	body->vecPosition = c_sum(body->vecPosition,totalPosAdd);
	body->shape->UpdateAABB(body->shape);
}

void PG_BodyInit(pgBodyObject* body)
{
	body->fAngleVelocity = 0.0;
	body->fFriction = 0.0;
	body->fMass = 1.0;
	body->fRestitution = 1.0;
	body->fRotation = 0.0;
	body->fTorque = 0.0;
	PG_Set_Vector2(body->vecForce,0.0,0.0);
	PG_Set_Vector2(body->vecImpulse,0.0,0.0);
	PG_Set_Vector2(body->vecLinearVelocity,0.0,0.0);
	PG_Set_Vector2(body->vecPosition,0.0,0.0);

	//TODO: here just for testing, would be replaced by generic function
	body->shape = PG_RectShapeNew(body, 20, 20, 0);
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
	PG_ShapeObjectDestroy(body->shape);
	body->ob_type->tp_free((PyObject*)body);
}

static PyTypeObject pgBodyType =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"physics.body",            /* tp_name */
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
	0,		   		            /* tp_methods */
	0,                          /* tp_members */
	0,                          /* tp_getset */
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
	rotate = bodyA->fRotation - bodyB->fRotation;
	p_in_A = *p_in_B;
	c_rotate(&p_in_A, rotate);
	p_in_A = c_sum(p_in_A, trans);
	
	return p_in_A;
}

pgVector2 PG_GetVelocity1(pgVector2 r, double w)
{
	pgVector2 v;
	double r_len, v_len;

	r_len = c_get_length(r);
	if(is_zero(r_len))
	{
		v.imag = v.real = 0;
	}
	else
	{
		r.real /= r_len;
		r.imag /= r_len;
		v_len = fabs(r_len*w);
		r.real *= v_len;
		r.imag *= v_len;
		if(w > 0) //counter-clock wise
		{	
			v.real = -r.imag;
			v.imag = r.real;
		}
		else
		{
			v.real = r.imag;
			v.imag = -r.real;
		}
	}

	return v;
}

pgVector2 PG_GetVelocity(pgBodyObject* body, pgVector2* global_p)
{	
	//get rotate radius vector r
	pgVector2 r = c_diff(*global_p, body->vecPosition);
	return PG_GetVelocity1(r, body->fRotation);
}


//static PyMemberDef Body_members[] = {
//	{"mass", T_FLOAT, offsetof(pgBodyObject,fMass), 0,"Mass"},
//	{"linear_velocity",T_DOUBLE,offsetof(pyBodyObject,vecLinearVelocity),"Linear Velocity"},
//	{"angle_velocity", T_FLOAT, offsetof(pgBodyObject,fAngleVelocity), 0,"Angle Velocity"},
//	{"position", T_DOUBLE, offsetof(pgBodyObject,vecPosition), 0,"position"},
//	{"rotation", T_FLOAT, offsetof(pgBodyObject,fRotation), 0,"Rotation"},
//    {NULL}  /* Sentinel */
//};
//
//static PyMethodDef Body_methods[] = {
//    {NULL}  /* Sentinel */
//};
//


