#include "pgWorldObject.h"
#include "pgBodyObject.h"

void _PG_FreeBodySimulation(pgWorldObject* world,double stepTime)
{
	Py_ssize_t size = PyList_Size((PyObject*)(world->bodyList));
	Py_ssize_t i;
	for (i = 0;i < size;i++)
	{
		pgBodyObject* body = (pgBodyObject*)(PyList_GetItem((PyObject*)(world->bodyList),i));
		PG_FreeUpdateBody(world,body,stepTime);
	}
}

void PG_Update(pgWorldObject* world,double stepTime)
{
	_PG_FreeBodySimulation(world,stepTime);
}


void PG_AddBodyToWorld(pgWorldObject* world,pgBodyObject* body)
{
	PyList_Append((PyObject*)world->bodyList,(PyObject*)body);
}

void PG_RemoveBodyFromWorld(pgWorldObject* world,pgBodyObject* body)
{
	
}

void PG_WorldInit(pgWorldObject* world)
{
	world->bodyList = (PyListObject*)PyList_New(0);
	world->jointList = (PyListObject*)PyList_New(0);
	world->fDamping = 0.0;
	world->fStepTime = 0.1;
	world->fTotalTime = 0.0;
	world->vecGravity.real = 0.0;
	world->vecGravity.imag = -9.8;
}

pgWorldObject* PG_WorldNew()
{
	pgWorldObject* op;
	op = (pgWorldObject*)PyObject_MALLOC(sizeof(pgWorldObject));
	PG_WorldInit(op);
	return op;
}

void	PG_WorldDestroy(pgWorldObject* world);