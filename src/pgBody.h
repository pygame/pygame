#ifndef _PYGAME_PHYSICS_BODY_
#define _PYGAME_PHYSICS_BODY_

#include "pgMathTypeDef.h"
#include "pgVector2.h"

typedef struct _pgShape pgShape;

typedef struct _pgBody{
	
	pgReal		fMass;
	pgVector2	vecLinearVelocity;
	pgReal		fAngleVelocity;

	pgVector2	vecPosition;
	pgReal		fRotation;
	pgVector2	vecImpulse;
	pgVector2	vecForce;
	pgReal		fTorque;

	pgReal		fRestitution;
	pgReal		fFriction;

	pgShape*	shapeBody;
} pgBody;



#endif //_PYGAME_PHYSICS_BODY_
