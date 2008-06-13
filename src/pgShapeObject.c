#include "pgShapeObject.h"
#include "pgBodyObject.h"
#include <string.h>
#include <math.h>


void PG_ShapeObjectInit(pgShapeObject* shape)
{
	//TODO: maybe these init methods are not suitable. need refined.
	memset(&(shape->box), 0, sizeof(shape->box));
	shape->Destroy = NULL;
	shape->IsPointIn = NULL;
	//shape->centroid.real = 0;
	//shape->centroid.imag = 0;
}

void PG_ShapeObjectDestroy(pgShapeObject* shape)
{
	shape->Destroy(shape);
}

int PG_RectShapeIsPointIn(pgShapeObject* shape, pgVector2* point)
{
	pgRectShape* ps = (pgRectShape*)shape;
	pgVector2 t1, t2;
	pgVector2 gp[4];
	double s1, s2;
	int i;

	t1 = c_diff(ps->bottomRight, ps->bottomLeft);
	t2 = c_diff(ps->topLeft, ps->bottomLeft);
	s1 = fabs(c_cross(t1, t2));

	s2 = 0;

	for(i = 0; i < 4; ++i)
		gp[i] = PG_GetGlobalCor(ps->shape.body, &(ps->point[i]));

	for(i = 0; i < 4; ++i)
	{
		t1 = c_diff(gp[i], *point);
		t2 = c_diff(gp[(i+1)%4], *point);
		s2 += fabs(c_cross(t1, t2)); 
	}

	return is_equal(s1, s2);
}

void PG_RectShapeDestroy(pgShapeObject* rectShape)
{
	PyObject_Free((pgRectShape*)rectShape);
}

pgShapeObject*	PG_RectShapeNew(pgBodyObject* body, double width, double height, double seta)
{
	int i;
	pgRectShape* p = (pgRectShape*)PyObject_MALLOC(sizeof(pgRectShape));

	PG_ShapeObjectInit(&(p->shape));
	p->shape.IsPointIn = PG_RectShapeIsPointIn;
	p->shape.Destroy = PG_RectShapeDestroy;

	PG_Set_Vector2(p->bottomLeft, -width/2, -height/2);
	PG_Set_Vector2(p->bottomRight, width/2, -height/2);
	PG_Set_Vector2(p->topRight, width/2, height/2);
	PG_Set_Vector2(p->topLeft, -width/2, height/2);
	for(i = 0; i < 4; ++i)
		c_rotate(&(p->point[i]), seta);

	//p->shape.centroid = center;
	p->shape.body = body;

	return (pgShapeObject*)p;
}

