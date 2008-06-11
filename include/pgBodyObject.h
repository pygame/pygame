#ifndef _PYGAME_PHYSICS_BODY_
#define _PYGAME_PHYSICS_BODY_


#include "pgVector2.h"

typedef struct _pgWorldObject pgWorldObject;
typedef struct _pgShapeObject pgShapeObject;

typedef struct _pgBodyObject{
	PyObject_HEAD

	double		fMass;
	pgVector2	vecLinearVelocity;
	double		fAngleVelocity;

	pgVector2	vecPosition;
	double		fRotation;
	pgVector2	vecImpulse;
	pgVector2	vecForce;
	double		fTorque;

	double		fRestitution;
	double		fFriction;

	pgShapeObject* shape;

} pgBodyObject;

pgBodyObject* PG_BodyNew();
void	PG_BodyDestroy(pgBodyObject* body);

void PG_FreeUpdateBodyVel(pgWorldObject* world, pgBodyObject* body, double dt);
void PG_FreeUpdateBodyPos(pgWorldObject* world, pgBodyObject* body, double dt);

pgVector2 PG_GetGlobalCor(pgBodyObject* body, pgVector2* local);

#endif //_PYGAME_PHYSICS_BODY_