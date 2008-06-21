#ifndef _PYGAME_PHYSICS_COLLISION_
#define _PYGAME_PHYSICS_COLLISION_

#include "pgBodyObject.h"
#include "pgAABBBox.h"

typedef struct _pgContact
{
	pgVector2 pos;
	pgVector2 normal;
	double depth;
	pgBodyObject* incBody; //incident rigid body
	pgBodyObject* refBody; //reference rigid body
}pgContact;

typedef enum _pgCollisionType
{
	MOVING_AWAY,
	RESTING,
	MOVING_TOWARD
}pgCollisionType;

int PG_LiangBarskey(pgAABBBox* box, pgVector2* p1, pgVector2* p2, 
					 pgVector2* ans_p1, pgVector2* ans_p2);

#endif
