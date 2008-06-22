#ifndef _PYGAME_PHYSICS_COLLISION_
#define _PYGAME_PHYSICS_COLLISION_

#include "pgBodyObject.h"
#include "pgJointObject.h"
#include "pgAABBBox.h"

typedef struct _pgContact
{
	//assert body2 is the incident rigid body
	//and body1 is the reference rigid body
	pgJointObject joint;

	pgVector2 pos;
	pgVector2 normal;
	double depth;
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
