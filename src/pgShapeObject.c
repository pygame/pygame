#include "pgShapeObject.h"
#include "pgBodyObject.h"
#include "pgCollision.h"
#include <string.h>
#include <math.h>
#include <assert.h>

//functions of pgShapeObject

void PG_ShapeObjectInit(pgShapeObject* shape)
{
	//TODO: maybe these init methods are not suitable. needing refined.
	memset(&(shape->box), 0, sizeof(shape->box));
	shape->Destroy = NULL;
	shape->Collision = NULL;
}

void PG_ShapeObjectDestroy(pgShapeObject* shape)
{
	shape->Destroy(shape);
}



//functions of pgRectShape

void PG_RectShapeUpdateAABB(pgShapeObject* rectShape)
{
	int i;
	pgRectShape *p = (pgRectShape*)rectShape;
	PG_AABBClear(&(p->shape.box));
	for(i = 0; i < 4; ++i)
		PG_AABBExpandTo(&(p->shape.box), &(p->point[i]));
}

void PG_RectShapeDestroy(pgShapeObject* rectShape)
{
	PyObject_Free((pgRectShape*)rectShape);
}

int PG_RectShapeCollision(pgBodyObject* selfBody, pgBodyObject* incidBody, 
						  PyListObject* contactPoints, pgVector2* contactNormal);


pgShapeObject*	PG_RectShapeNew(pgBodyObject* body, double width, double height, double seta)
{
	int i;
	pgRectShape* p = (pgRectShape*)PyObject_MALLOC(sizeof(pgRectShape));

	PG_ShapeObjectInit(&(p->shape));
	p->shape.Destroy = PG_RectShapeDestroy;
	p->shape.UpdateAABB = PG_RectShapeUpdateAABB;
	p->shape.type = ST_RECT;

	PG_Set_Vector2(p->bottomLeft, -width/2, -height/2);
	PG_Set_Vector2(p->bottomRight, width/2, -height/2);
	PG_Set_Vector2(p->topRight, width/2, height/2);
	PG_Set_Vector2(p->topLeft, -width/2, height/2);
	for(i = 0; i < 4; ++i)
		c_rotate(&(p->point[i]), seta);

	return (pgShapeObject*)p;
}

//-------------box's collision test------------------

//we use a simple SAT to select the contactNormal:
//Supposing the relative velocity between selfBody and incidBody in
//two frame is small, the
static void _SAT_GetContactNormal(pgAABBBox* clipBox, PyListObject* contactPoints,
								  pgVector2* contactNormal)
{

}


//TODO: now just detect Box-Box collision, later add Box-Circle
int PG_RectShapeCollision(pgBodyObject* selfBody, pgBodyObject* incidBody, 
						  PyListObject* contactPoints, pgVector2* contactNormal)
{
	int i, i1;
	int apart;
	pgVector2 ip[4];
	int has_ip[4]; //use it to prevent from duplication
	pgVector2 pf, pt;
	pgRectShape *self, *incid;
	pgAABBBox clipBox;

	self = (pgRectShape*)selfBody->shape;
	incid = (pgRectShape*)incidBody->shape;

	//transform incidBody's coordinate according to selfBody's coordinate
	for(i = 0; i < 4; ++i)
		ip[i] = PG_GetRelativePos(selfBody, incidBody, &(incid->point[i]));

	//clip incidBody by selfBody
	clipBox = PG_GenAABB(self->bottomLeft.real, self->topRight.real,
		self->bottomLeft.imag, self->topRight.imag);
	apart = 1;
	memset(has_ip, 0, sizeof(has_ip));
	//watch out! we create contactPoints here
	contactPoints = (PyListObject*)PyList_New(0);

	for(i = 0; i < 4; ++i)
	{
		i1 = (i+1)%4;
		//if collision happens, clip incident object and append the 
		//clipped points to contact points list
		//note: clipped vertices of incident object will be appended later
		//      to prevent from duplication
		if(PG_LiangBarskey(&clipBox, &ip[i], &ip[i1], &pf, &pt))
		{
			apart = 0;
			if(pf.real == ip[i].real && pf.imag == ip[i].imag)
				has_ip[i] = 1;
			else
				PyList_Append((PyObject*)contactPoints, (PyObject*)PyComplex_FromCComplex(pf));
			if(pt.real == ip[i1].real && pt.imag == ip[i1].imag)
				has_ip[i1] = 1;
			else
				PyList_Append((PyObject*)contactPoints, (PyObject*)PyComplex_FromCComplex(pt));
		}
	}

	if(apart)
		return 0;

	for(i = 0; i < 4; ++i)
	{
		if(has_ip[i])
			PyList_Append((PyObject*)contactPoints, (PyObject*)PyComplex_FromCComplex(ip[i]));
	}
	//now all the contact points are added to list
	//note at the moment they are in selfBody's locate coordinate system
	


	return 1;
}
