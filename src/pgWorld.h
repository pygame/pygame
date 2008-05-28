#ifndef _PYGAME_PHYSICS_WORLD_
#define _PYGAME_PHYSICS_WORLD_

typedef _pgWorld{
	
} pgWorld;

void pgAddBodyToWorld(pgWorld* world,pgBody* body);
void pgRemoveBodyFromWorld(pgWorld* world,pgBody* body);
void pgAddJointToWorld(pgWorld* world,pgJoint* joint);
void pgRemoveJointFromWorld(pgWorld* world,pgJoint* joint);
void pgWorldStepSimulation(pgWorld* world,pgReal stepTime);

#endif