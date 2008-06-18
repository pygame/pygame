#ifndef _PYGAME_PHYSICS_COLLISION_
#define _PYGAME_PHYSICS_COLLISION_

#include "pgBodyObject.h"

typedef struct _pgContact
{
	pgVector2 pos;
	pgVector2 normal;
	double depth;
	pgBodyObject* incBody; //incident rigid body
	pgBodyObject* refBody; //reference rigid body
}pgContact;

enum pgCollisionType
{
	MOVING_AWAY,
	RESTING,
	MOVING_TOWARD
};

#endif
