#ifndef _PYGAME_PHYSICS_SHAPE_
#define _PYGAME_PHYSICS_SHAPE_

#include <Python.h>
#include "pgAABBBox.h"

typedef struct _pgBodyObject pgBodyObject;


// shape base type
typedef struct _pgShapeObject{
	PyObject_HEAD

	pgBodyObject*		body;
	pgAABBBox	box;

	void (*DestroyShape)();
	void (*UpdateAABBBox)();
} pgShapeObject;

void	PG_ShapeDestroy(pgShapeObject* shape);

//subclass type

typedef struct _pgRectShape{
	pgShapeObject		shape;

	pgAABBBox			rectBox;
} pgPolygonShape;

pgShapeObject*	PG_RectShapeNew(pgAABBBox rect);


typedef struct _pgPolygonShape{
	pgShapeObject		shape;

	PyListObject*		vertexList;
};

#endif //_PYGAME_PHYSICS_SHAPE_