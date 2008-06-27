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

void PG_RectShapeUpdateAABB(pgBodyObject* body)
{
	int i;
	pgVector2 gp[4];

	if(body->shape->type == ST_RECT)
	{
		pgRectShape *p = (pgRectShape*)body->shape;
		
		PG_AABBClear(&(p->shape.box));
		for(i = 0; i < 4; ++i)
			gp[i] = PG_GetGlobalPos(body, &(p->point[i]));
		for(i = 0; i < 4; ++i)
			PG_AABBExpandTo(&(p->shape.box), &gp[i]);
	}
}

void PG_RectShapeDestroy(pgShapeObject* rectShape)
{
	PyObject_Free((pgRectShape*)rectShape);
}


int PG_RectShapeCollision(pgBodyObject* selfBody, pgBodyObject* incidBody, PyObject* contactList);


pgShapeObject*	PG_RectShapeNew(pgBodyObject* body, double width, double height, double seta)
{
	int i;
	pgRectShape* p = (pgRectShape*)PyObject_MALLOC(sizeof(pgRectShape));

	PG_ShapeObjectInit(&(p->shape));
	p->shape.Destroy = PG_RectShapeDestroy;
	p->shape.UpdateAABB = PG_RectShapeUpdateAABB;
	p->shape.Collision = PG_RectShapeCollision;
	p->shape.type = ST_RECT;
	p->shape.rInertia = body->fMass*(width*width + height*height)/12; // I = M(a^2 + b^2)/12

	PG_Set_Vector2(p->bottomLeft, -width/2, -height/2);
	PG_Set_Vector2(p->bottomRight, width/2, -height/2);
	PG_Set_Vector2(p->topRight, width/2, height/2);
	PG_Set_Vector2(p->topLeft, -width/2, height/2);
	for(i = 0; i < 4; ++i)
		c_rotate(&(p->point[i]), seta);

	return (pgShapeObject*)p;
}

//-------------box's collision test------------------
//TEST: these functions have been partly tested.

//we use a simple SAT to select the contactNormal:
//Supposing the relative velocity between selfBody and incidBody in
//two frame is "small", the face(actually is an edge in 2D) with minimum 
//average penetrating depth is considered to be contact face, then we
//get the contact normal.
//note: this method is not available in CCD(continue collision detection)
//since the velocity is not small.
static void _SAT_GetContactNormal(pgAABBBox* clipBox, PyObject* contactList,
								  int from, int to)
{
	int i;
	int id;
	double deps[4], min_dep;
	pgContact* p;
	pgVector2 normal;
		
	memset(deps, 0, sizeof(deps));
	for(i = from; i <= to; ++i)
	{
		p = (pgContact*)PyList_GetItem(contactList, i);
		deps[0] += p->pos.real - clipBox->left; //left
		deps[1] += p->pos.imag - clipBox->bottom; //bottom
		deps[2] += clipBox->right - p->pos.real; //right
		deps[3] += clipBox->top - p->pos.imag; //top
	}
	
	//find min penetrating face
	id = 0;
	min_dep = deps[0];
	for(i = 1; i < 4; ++i)
	{
		if(min_dep > deps[i])
		{
			min_dep = deps[i];
			id = i;
		}
	}
	PG_Set_Vector2(normal, 0, 0);
	//generate contactNormal
	switch(id)
	{
	case 0://left
		PG_Set_Vector2(normal, -1, 0);
		break;
	case 1://bottom
		PG_Set_Vector2(normal, 0, -1);
		break;
	case 2://right
		PG_Set_Vector2(normal, 1, 0);
		break;
	case 3://top
		PG_Set_Vector2(normal, 0, 1);
		break;
	}

    for(i = from; i <= to; ++i)
    {
        p = (pgContact*)PyList_GetItem(contactList, i);
        p->normal = normal;
    }

}


//TODO: now just detect Box-Box collision, later add Box-Circle
int PG_RectShapeCollision(pgBodyObject* selfBody, pgBodyObject* incidBody, PyObject* contactList)
{
	int i, i1;
	int from, to;
	int apart;
	pgVector2 ip[4];
	int has_ip[4]; //use it to prevent from duplication
	pgVector2 pf, pt;
	pgRectShape *self, *incid;
	pgAABBBox clipBox;
	pgContact* contact;

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

	from = PyList_Size(contactList);
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
			{
				has_ip[i] = 1;
			}
			else
			{
				contact = (pgContact*)PG_ContactNew(selfBody, incidBody);
				contact->pos = pf;
				PyList_Append(contactList, (PyObject*)contact);
			}
			
			if(pt.real == ip[i1].real && pt.imag == ip[i1].imag)
			{	
				has_ip[i1] = 1;
			}
			else
			{
				contact = (pgContact*)PG_ContactNew(selfBody, incidBody);
				contact->pos = pt;
				PyList_Append(contactList, (PyObject*)contact);
			}

		}
	}

	if(apart)
		return 0;

	for(i = 0; i < 4; ++i)
	{
		if(has_ip[i])
		{
			contact = (pgContact*)PG_ContactNew(selfBody, incidBody);
			contact->pos = ip[i];
			PyList_Append(contactList, (PyObject*)contact);
		}
	}
	//now all the contact points are added to list
	to = PyList_Size(contactList) - 1;

	
	_SAT_GetContactNormal(&clipBox, contactList, from, to);

	//transform from selfBody's locate coordinate to global coordinate
	for(i = from; i <= to; ++i)
	{
		contact = (pgContact*)PyList_GetItem(contactList, i);

		c_rotate(&(contact->pos), selfBody->fRotation);
		contact->pos = c_sum(contact->pos, selfBody->vecPosition);
		
		c_rotate(&(contact->normal), selfBody->fRotation);
	}


	return 1;
}
