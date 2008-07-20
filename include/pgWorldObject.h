#ifndef _PYGAME_PHYSICS_WORLD_
#define _PYGAME_PHYSICS_WORLD_


#include <Python.h>
#include "pgAABBBox.h"
#include "pgDeclare.h"

struct _pgWorldObject 
{
	PyObject_HEAD

	PyObject*	bodyList;
	PyObject*	jointList;
	PyObject*	contactList;

	Py_complex		vecGravity;
	double			fDamping;

	double			fStepTime;
	double			fTotalTime;
	pgAABBBox		worldBox;

};

pgWorldObject* PG_WorldNew();
void	PG_WorldDestroy(pgWorldObject* world);

void	PG_Update(pgWorldObject* world,double stepTime);
int		PG_AddBodyToWorld(pgWorldObject* world,pgBodyObject* body);
int		PG_RemoveBodyFromWorld(pgWorldObject* world,pgBodyObject* body);
int		PG_AddJointToWorld(pgWorldObject* world,pgJointObject* joint);
int		PG_RemoveJointFromWorld(pgWorldObject* world,pgJointObject* joint);


#endif
