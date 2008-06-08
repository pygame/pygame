#ifndef _PYGAME_PHYSICS_BODY_
#define _PYGAME_PHYSICS_BODY_


#include <Python.h>

typedef struct _pgWorldObject pgWorldObject;

typedef struct _pgBodyObject{
	PyObject_HEAD

	double		fMass;
	Py_complex	vecLinearVelocity;
	double		fAngleVelocity;

	Py_complex	vecPosition;
	double		fRotation;
	Py_complex	vecImpulse;
	Py_complex	vecForce;
	double		fTorque;

	double		fRestitution;
	double		fFriction;
} pgBodyObject;

pgBodyObject* PG_BodyNew();
void	PG_BodyDestroy(pgBodyObject* body);

void PG_FreeUpdateBody(pgWorldObject* world,pgBodyObject* body,double dt);


#endif //_PYGAME_PHYSICS_BODY_