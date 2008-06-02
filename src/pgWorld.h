#ifndef _PYGAME_PHYSICS_WORLD_
#define _PYGAME_PHYSICS_WORLD_

#include <ds.h>

#include "pgMathTypeDef.h"
#include "pgBody.h"
#include "pgJoint.h"


typedef struct _pgWorld{
	PARRAY	bodyArray;
	PARRAY	jointArray;

	pgReal	fGravity;
	pgReal	fDamping;
} pgWorld;

static void pgAddBodyToWorld(pgWorld* world,pgBody* body);
static void pgRemoveBodyFromWorld(pgWorld* world,pgBody* body);
static void pgAddJointToWorld(pgWorld* world,pgJoint* joint);
static void pgRemoveJointFromWorld(pgWorld* world,pgJoint* joint);
static void pgWorldStepSimulation(pgWorld* world,pgReal stepTime);

#endif