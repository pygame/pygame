#include "pgWorldObject.h"
#include "pgBodyObject.h"
#include "pgJointObject.h"

void _PG_FreeBodySimulation(pgWorldObject* world,double stepTime)
{
	Py_ssize_t size = PyList_Size((PyObject*)(world->bodyList));
	Py_ssize_t i;
	for (i = 0;i < size;i++)
	{
		pgBodyObject* body = (pgBodyObject*)(PyList_GetItem((PyObject*)(world->bodyList),i));
		PG_FreeUpdateBodyVel(world,body,stepTime);
	}
}

void _PG_BodyCollisionDetection(pgWorldObject* world)
{

}

void _PG_JointSolve(pgWorldObject* world,double stepTime)
{
	Py_ssize_t size = PyList_Size((PyObject*)(world->jointList));
	Py_ssize_t i;
	for (i = 0;i < size;i++)
	{
		pgJointObject* joint = (pgJointObject*)(PyList_GetItem((PyObject*)(world->jointList),i));
		if (joint->SolveConstraint)
		{
			joint->SolveConstraint(joint,stepTime);
		}
	}
}

void _PG_BodyPositionUpdate(pgWorldObject* world,double stepTime)
{
	Py_ssize_t size = PyList_Size((PyObject*)(world->bodyList));
	Py_ssize_t i;
	for (i = 0;i < size;i++)
	{
		pgBodyObject* body = (pgBodyObject*)(PyList_GetItem((PyObject*)(world->bodyList),i));
		PG_FreeUpdateBodyPos(world,body,stepTime);
	}
}


void PG_Update(pgWorldObject* world,double stepTime)
{
	_PG_FreeBodySimulation(world, stepTime);
	_PG_BodyCollisionDetection(world);
	_PG_JointSolve(world,stepTime);
	_PG_BodyPositionUpdate(world, stepTime);
}


void PG_AddBodyToWorld(pgWorldObject* world,pgBodyObject* body)
{
	PyList_Append((PyObject*)world->bodyList,(PyObject*)body);
}

void PG_RemoveBodyFromWorld(pgWorldObject* world,pgBodyObject* body)
{
	
}

void PG_AddJointToWorld(pgWorldObject* world,pgJointObject* joint)
{
	PyList_Append((PyObject*)world->jointList,(PyObject*)joint);
}

void PG_RemoveJointFromWorld(pgWorldObject* world,pgJointObject* joint)
{

}

void PG_WorldInit(pgWorldObject* world)
{
	world->bodyList = (PyListObject*)PyList_New(0);
	world->jointList = (PyListObject*)PyList_New(0);
	world->fDamping = 0.0;
	world->fStepTime = 0.1;
	PG_Set_Vector2(world->vecGravity,0.0,-50);
	world->fTotalTime = 0.0;

}

pgWorldObject* PG_WorldNew()
{
	pgWorldObject* op;
	op = (pgWorldObject*)PyObject_MALLOC(sizeof(pgWorldObject));
	PG_WorldInit(op);
	return op;
}

void	PG_WorldDestroy(pgWorldObject* world);