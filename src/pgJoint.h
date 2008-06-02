#ifndef _PYGAME_PHYSICS_JOINT_
#define _PYGAME_PHYSICS_JOINT_

#include "pgMathTypeDef.h"

typedef struct _pgJoint pgJoint;

typedef struct _pgJoint{
	pgBody*	body1;
	pgBody*	body2;
	bool	isCollideConnect;
	void	(*SolveConstraint)(pgJoint* joint,pgReal stepTime);
} pgJoint;

typedef struct _pgDistanceJoint{
	pgJoint		joint;
	pgReal		distance;
	pgVector2	anchor1,anchor2;
} pgDistanceJoint;

pgJoint* CreateDistanceJoint(pgBody* _body1,pgBody* _body2,pgReal _dist,pgVector2 anchor1,pgVector2 anchor2);
void	 DestroyDistanceJoint(pgJoint* joint);


#endif //_PYGAME_PHYSICS_JOINT_