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
	pgVector2 dv;
	double depth;
	double weight;
	double resist;
	pgVector2** ppAccMoment;
}pgContact;

typedef enum _pgCollisionType
{
	MOVING_AWAY,
	RESTING,
	MOVING_TOWARD
}pgCollisionType;

typedef enum _pgCollisionAxis
{
	CA_X = 0,
	CA_Y = 1
}pgCollisionAxis;

typedef enum _pgCollisionFace
{
	CF_LEFT,
	CF_BOTTOM,
	CF_RIGHT,
	CF_TOP
}pgCollisionFace;

int PG_LiangBarskey(pgAABBBox* box, pgVector2* p1, pgVector2* p2, 
					 pgVector2* ans_p1, pgVector2* ans_p2);

int PG_PartlyLB(pgAABBBox* box, pgVector2* p1, pgVector2* p2, 
				pgCollisionAxis axis, pgVector2* ans_p1, pgVector2* ans_p2,
				int* valid_p1, int* valid_p2);

pgJointObject* PG_ContactNew(pgBodyObject* refBody, pgBodyObject* incidBody);

void PG_AppendContact(pgBodyObject* refBody, pgBodyObject* incidBody, PyObject* contactList);
void PG_ApplyContact(PyObject* contactObject);

#endif
