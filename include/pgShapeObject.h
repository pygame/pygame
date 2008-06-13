#ifndef _PYGAME_PHYSICS_SHAPE_
#define _PYGAME_PHYSICS_SHAPE_

#include <Python.h>
#include "pgAABBBox.h"
#include "pgVector2.h"

typedef struct _pgBodyObject pgBodyObject;
typedef struct _pgShapeObject pgShapeObject;

// shape base type
typedef struct _pgShapeObject{
	PyObject_HEAD

	pgAABBBox box;
	pgBodyObject* body;
	//pgVector2 centroid;

	//virtual functions
	void (*Destroy)(pgShapeObject *shape);
	int (*IsPointIn)(pgShapeObject* shape, pgVector2* point);
} pgShapeObject;


void	PG_ShapeDestroy(pgShapeObject* shape);

//subclass type
typedef struct _pgRectShape{
	pgShapeObject shape;

	union
	{
		struct
		{
			pgVector2 point[4];
		};
		struct
		{
			pgVector2 bottomLeft, bottomRight, topRight, topLeft;
		};
	};
	
} pgRectShape;

pgShapeObject*	PG_RectShapeNew(pgBodyObject* body, double width, double height, double seta);


//typedef struct _pgPolygonShape{
//	pgShapeObject		shape;
//
//	PyListObject*		vertexList;
//}pgPolygonShape;

#endif //_PYGAME_PHYSICS_SHAPE_