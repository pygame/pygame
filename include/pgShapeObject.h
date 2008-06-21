#ifndef _PYGAME_PHYSICS_SHAPE_
#define _PYGAME_PHYSICS_SHAPE_

#include <Python.h>
#include "pgAABBBox.h"
#include "pgVector2.h"
#include "pgDeclare.h"


typedef enum _ShapeType
{
	ST_RECT,
	ST_CIRCLE
}ShapeType;


// shape base type
typedef struct _pgShapeObject{
	PyObject_HEAD

	pgAABBBox box;
	ShapeType type;

	//virtual functions
	void (*Destroy)(pgShapeObject* shape);
	int (*Collision)(pgBodyObject* selfBody, pgBodyObject* incidBody, PyListObject* contactPoints, 
		               pgVector2* contactNormal);
	void (*UpdateAABB)(pgShapeObject* shape);
};


void	PG_ShapeObjectDestroy(pgShapeObject* shape);

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
