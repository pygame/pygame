#ifndef _PYGAME_PHYSICS_JOINT_
#define _PYGAME_PHYSICS_JOINT_

#include "pgBodyObject.h"

typedef struct _pgJoint pgJoint;

typedef struct _pgJoint{
	PyObject_HEAD

	pgBodyObject*	body1;
	pgBodyObject*	body2;
	bool	isCollideConnect;
	void	(*SolveConstraint)(pgJoint* joint,double stepTime);
} pgJoint;

typedef struct _pgDistanceJoint{
	pgJoint		joint;
	double		distance;
	Py_complex	anchor1,anchor2;
} pgDistanceJoint;



#endif //_PYGAME_PHYSICS_JOINT_