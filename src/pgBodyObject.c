#include "pgBodyObject.h"
#include "pgWorldObject.h"
#include "pgComplexExtensions.h"
#include <structmember.h>


void PG_FreeUpdateBody(pgWorldObject* world,pgBodyObject* body, double dt)
{
	Py_complex totalVelAdd,totalPosAdd;
	double k;
	totalVelAdd = c_sum(body->vecForce,world->vecGravity);
	k = dt / body->fMass;
	totalVelAdd = c_mul_complex_with_real(totalVelAdd,k);
	body->vecLinearVelocity = c_sum(body->vecLinearVelocity,totalVelAdd);

	totalPosAdd = c_mul_complex_with_real(body->vecLinearVelocity,dt);
	body->vecPosition = c_sum(body->vecPosition,totalPosAdd);
}

void PG_BodyInit(pgBodyObject* body)
{
	body->fAngleVelocity = 0.0;
	body->fFriction = 0.0;
	body->fMass = 1.0;
	body->fRestitution = 1.0;
	body->fRotation = 0.0;
	body->fTorque = 0.0;
	body->vecForce.real = 0.0;
	body->vecForce.imag = 0.0;
	body->vecImpulse.real = 0.0;
	body->vecImpulse.imag = 0.0;
	body->vecLinearVelocity.real = 0.0;
	body->vecLinearVelocity.imag = 0.0;
	body->vecPosition.real = 0.0;
	body->vecPosition.imag = 0.0;
}

pgBodyObject* PG_BodyNew()
{
	pgBodyObject* op;
	op = (pgBodyObject*)PyObject_MALLOC(sizeof(pgBodyObject));
	PG_BodyInit(op);
	return op;
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
//static PyTypeObject BodyType = {
//    PyObject_HEAD_INIT(NULL)
//    0,                         /*ob_size*/
//    "physics.body",			/*tp_name*/
//    sizeof(pgBodyObject),	/*tp_basicsize*/
//    0,                         /*tp_itemsize*/
//    (destructor)Body_dealloc, /*tp_dealloc*/
//    0,                         /*tp_print*/
//    0,                         /*tp_getattr*/
//    0,                         /*tp_setattr*/
//    0,                         /*tp_compare*/
//    0,                         /*tp_repr*/
//    0,                         /*tp_as_number*/
//    0,                         /*tp_as_sequence*/
//    0,                         /*tp_as_mapping*/
//    0,                         /*tp_hash */
//    0,                         /*tp_call*/
//    0,                         /*tp_str*/
//    0,                         /*tp_getattro*/
//    0,                         /*tp_setattro*/
//    0,                         /*tp_as_buffer*/
//    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
//    "Body objects",           /* tp_doc */
//    0,		               /* tp_traverse */
//    0,		               /* tp_clear */
//    0,		               /* tp_richcompare */
//    0,		               /* tp_weaklistoffset */
//    0,		               /* tp_iter */
//    0,		               /* tp_iternext */
//    Body_methods,             /* tp_methods */
//    Body_members,             /* tp_members */
//    0,                         /* tp_getset */
//    0,                         /* tp_base */
//    0,                         /* tp_dict */
//    0,                         /* tp_descr_get */
//    0,                         /* tp_descr_set */
//    0,                         /* tp_dictoffset */
//    (initproc)Body_init,      /* tp_init */
//    0,                         /* tp_alloc */
//    Body_new,                 /* tp_new */
//};


