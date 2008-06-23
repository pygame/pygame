#ifndef _PYGAME_PHYSICS_JOINT_
#define _PYGAME_PHYSICS_JOINT_

#include "pgBodyObject.h"
#include "pgDeclare.h"

struct _pgJointObject{
	PyObject_HEAD

	pgBodyObject*	body1;
	pgBodyObject*	body2;
	int		isCollideConnect;
	void	(*SolveConstraintPosition)(pgJointObject* joint,double stepTime);
	void	(*SolveConstraintVelocity)(pgJointObject* joint,double stepTime);
	void	(*Destroy)(pgJointObject* joint);
};

void PG_JointDestroy(pgJointObject* joint);

typedef struct _pgDistanceJointObject{
	pgJointObject		joint;

	double		distance;
	pgVector2	anchor1,anchor2;
} pgDistanceJointObject;

pgJointObject* PG_DistanceJointNew(pgBodyObject* b1,pgBodyObject* b2,int bCollideConnect,double distance,pgVector2 a1,pgVector2 a2);

#endif //_PYGAME_PHYSICS_JOINT_