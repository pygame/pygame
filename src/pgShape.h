#ifndef _PYGAME_PHYSICS_SHAPE_
#define _PYGAME_PHYSICS_SHAPE_

#include "pgVector2.h"
#include "pgAABBBox.h"

typedef struct _pgBody pgBody;


typedef struct _pgShape{
	pgBody*		body;
	pgAABBBox	box;

	void (*InitShape)(pgBody* bd);
	void (*DestroyShape)();
	void (*UpdateAABBBox)();
} pgShape;


typedef struct _pgPolygonShape{
	pgShape		shape;
	
	int			iVertexNum;
	pgVector2*	iVertexArray;
} pgPolygonShape;

typedef struct _pgCircleShape{
	pgShape		shape;

	pgVector2	vecCenter;
	pgReal		fRadius;
} pgCircleShape;

#endif //_PYGAME_PHYSICS_SHAPE_