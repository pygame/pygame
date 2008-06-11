#ifndef _PYGAME_PHYSICS_WORLD_
#define _PYGAME_PHYSICS_WORLD_


#include <Python.h>
#include "pgAABBBox.h"

typedef struct _pgBodyObject pgBodyObject;
typedef struct _pgJointObject pgJointObject;

typedef struct _pgWorldObject 
{
	PyObject_HEAD

	PyListObject*	bodyList;
	PyListObject*	jointList;

	Py_complex		vecGravity;
	double			fDamping;

	double			fStepTime;
	double			fTotalTime;
	pgAABBBox		worldBox;

} pgWorldObject;

pgWorldObject* PG_WorldNew();
void	PG_WorldDestroy(pgWorldObject* world);

void	PG_Update(pgWorldObject* world,double stepTime);
void	PG_AddBodyToWorld(pgWorldObject* world,pgBodyObject* body);
void	PG_RemoveBodyFromWorld(pgWorldObject* world,pgBodyObject* body);
void	PG_AddJointToWorld(pgWorldObject* world,pgJointObject* joint);
void	PG_RemoveJointFromWorld(pgWorldObject* world,pgJointObject* joint);

#endif