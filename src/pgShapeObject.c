#include "pgShapeObject.h"
#include "pgBodyObject.h"
#include "pgCollision.h"
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>

//PyTypeObject pgShapeType;
//PyTypeObject pgRectShapeType;

//functions of pgShapeObject

void PG_ShapeObjectInit(pgShapeObject* shape)
{
	//TODO: maybe these init methods are not suitable. needing refined.
	memset(&(shape->box), 0, sizeof(shape->box));
	shape->Destroy = NULL;
	shape->Collision = NULL;
}

//pgShapeObject* _PG_ShapeNewInternal(PyTypeObject *type)
//{
//	pgShapeObject* op = (pgShapeObject*)type->tp_alloc(type, 0);
//	PG_ShapeObjectInit(op);
//	return op;
//}
//
void PG_ShapeObjectDestroy(pgShapeObject* shape)
{
	if(shape != NULL)
		shape->Destroy(shape);
	PyObject_Free(shape);
}
//
//PyObject* _pgShapeNew(PyTypeObject *type, PyObject *args, PyObject *kwds)
//{
//	/* In case we have arguments in the python code, parse them later
//	* on.
//	*/
//	PyObject* shape;
//	if(PyType_Ready(type)==-1) return NULL;
//
//	shape = (PyObject*) _PG_ShapeNewInternal(type);
//	return shape;
//}
//
//static int _pgShape_init(pgShapeObject* shape,PyObject *args, PyObject *kwds)
//{
//	PG_ShapeObjectInit(shape);
//	return 0;
//}
//
//PyTypeObject pgShapeType =
//{
//	PyObject_HEAD_INIT(NULL)
//	0,
//	"physics.Shape",            /* tp_name */
//	sizeof(pgShapeObject),      /* tp_basicsize */
//	0,                          /* tp_itemsize */
//	(destructor)PG_ShapeObjectDestroy,/* tp_dealloc */
//	0,                          /* tp_print */
//	0,                          /* tp_getattr */
//	0,                          /* tp_setattr */
//	0,                          /* tp_compare */
//	0,                          /* tp_repr */
//	0,                          /* tp_as_number */
//	0,                          /* tp_as_sequence */
//	0,                          /* tp_as_mapping */
//	0,                          /* tp_hash */
//	0,                          /* tp_call */
//	0,                          /* tp_str */
//	0,                          /* tp_getattro */
//	0,                          /* tp_setattro */
//	0,                          /* tp_as_buffer */
//	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
//	"",                         /* tp_doc */
//	0,                          /* tp_traverse */
//	0,                          /* tp_clear */
//	0,                          /* tp_richcompare */
//	0,                          /* tp_weaklistoffset */
//	0,                          /* tp_iter */
//	0,                          /* tp_iternext */
//	0,		   	/* tp_methods */
//	0,            /* tp_members */
//	0,          /* tp_getset */
//	0,                          /* tp_base */
//	0,                          /* tp_dict */
//	0,                          /* tp_descr_get */
//	0,                          /* tp_descr_set */
//	0,                          /* tp_dictoffset */
//	_pgShape_init,                          /* tp_init */
//	0,                          /* tp_alloc */
//	_pgShapeNew,                /* tp_new */
//	0,                          /* tp_free */
//	0,                          /* tp_is_gc */
//	0,                          /* tp_bases */
//	0,                          /* tp_mro */
//	0,                          /* tp_cache */
//	0,                          /* tp_subclasses */
//	0,                          /* tp_weaklist */
//	0                           /* tp_del */
//};



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

//just for C test usage, not for python???
pgShapeObject*	PG_RectShapeNew(pgBodyObject* body, double width, double height, double seta)
{
	int i;
	pgRectShape* p = (pgRectShape*)PyObject_MALLOC(sizeof(pgRectShape));
	//pgRectShape* p = PyObject_New(pgRectShape,&pgRectShapeType);

	//PG_ShapeObjectInit(&(p->shape));
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

static int _Get_Depth(pgBodyObject* refBody, pgBodyObject* incBody,
					   int* faceId, double* min_dep, pgVector2* gp_in_ref, 
					   pgAABBBox* clipBox)
{
#define _EPS_DEPTH 1e-8

	int i, apart;
	pgRectShape *ref, *inc;
	double deps[4];

	ref = (pgRectShape*)refBody->shape;
	inc = (pgRectShape*)incBody->shape;
	memset(gp_in_ref, 0, sizeof(gp_in_ref));
	memset(deps, 0, sizeof(deps));
	for(i = 0; i < 4; ++i)
		gp_in_ref[i] = PG_GetRelativePos(refBody, incBody, &(inc->point[i]));

	*clipBox = PG_GenAABB(ref->bottomLeft.real, ref->topRight.real,
		ref->bottomLeft.imag, ref->topRight.imag);

	apart = 1;
	for(i = 0; i < 4; ++i)
		if(PG_IsIn(&gp_in_ref[i], clipBox, _EPS_DEPTH))
		{
			apart = 0;
			deps[CF_LEFT] += fabs(gp_in_ref[i].real - clipBox->left);
			deps[CF_RIGHT] += fabs(clipBox->right - gp_in_ref[i].real);
			deps[CF_BOTTOM] += fabs(gp_in_ref[i].imag - clipBox->bottom);
			deps[CF_TOP] += fabs(clipBox->top - gp_in_ref[i].imag);
		}

	if(apart) return 0;

	*min_dep = deps[0];
	*faceId = 0;

	for(i = 1; i < 4; ++i)
		if(deps[i] < *min_dep)
		{
			*min_dep = deps[i];
			*faceId = i;
		}

	return 1;
}

static int _SAT_Select(pgBodyObject* body1, pgBodyObject* body2,
					   pgBodyObject** refBody, pgBodyObject** incBody,
					   int* face_id, pgVector2* gp_in_ref, pgAABBBox* clipBox,
					   double* minDep)
{
	double min_dep[2];
	int id[2];
	pgVector2 gp[2][4];
	pgAABBBox cb[2];
	int is_in[2];
	int i;
	
	min_dep[0] = min_dep[1] = DBL_MAX;
	is_in[0] = _Get_Depth(body1, body2, &id[0], &min_dep[0], gp[0], &cb[0]);
	is_in[1] = _Get_Depth(body2, body1, &id[1], &min_dep[1], gp[1], &cb[1]);

	if(!is_in[0] && !is_in[1]) return 0;

	if(min_dep[0] < min_dep[1])
	{
		*refBody = body1;
		*incBody = body2;
		*face_id = id[0];
		for(i = 0; i < 4; ++i)
			gp_in_ref[i] = gp[0][i];
		*clipBox = cb[0];
		*minDep = min_dep[0];
	}
	else
	{
		*refBody = body2;
		*incBody = body1;
		*face_id = id[1];
		for(i = 0; i < 4; ++i)
			gp_in_ref[i] = gp[1][i];
		*clipBox = cb[1];
		*minDep = min_dep[1];
	}

	return 1;
}

int _Build_Contacts(pgVector2* gp, pgAABBBox* clipBox, int axis,
					 pgVector2 contacts[], int* size)
{
	int i, i1;
	int apart = 1;
	int has_ip[4];
	pgVector2 pf, pt;
	int valid_pf, valid_pt;

	*size = 0;
	memset(has_ip, 0, sizeof(has_ip));
	for(i = 0; i < 4; ++i)
	{
		i1 = (i+1)%4;
		if(PG_PartlyLB(clipBox, &gp[i], &gp[i1], axis, 
			&pf, &pt, &valid_pf, &valid_pt))
		{
			apart = 0;
			if(valid_pf)
			{
				if(c_equal(&pf, &gp[i]))
					has_ip[i] = 1;
				else
					contacts[(*size)++] = pf;
			}
			if(valid_pt)
			{
				if(c_equal(&pt, &gp[i1]))
					has_ip[i1] = 1;
				else
					contacts[(*size)++] = pt;
			}
		}
	}
	for(i = 0; i < 4; ++i)
		if(has_ip[i])
			contacts[(*size)++] = gp[i];

	return !apart;
}

int PG_RectShapeCollision(pgBodyObject* selfBody, pgBodyObject* incidBody, 
						  PyObject* contactList)
{
#define MAX_CONTACTS 8

	pgBodyObject* ref = NULL, *inc = NULL;
	int face_id;
	pgVector2 gp[4];
	pgVector2 contacts[MAX_CONTACTS];
	int csize;
	pgVector2 normal;
	pgAABBBox clipBox;
	int overlap;
	pgVector2* pAcc, * pSplitAcc;
	pgContact* contact;
	int i;
	double minDep;


	overlap = _SAT_Select(selfBody, incidBody,
						  &ref, &inc,
						  &face_id, gp, &clipBox, &minDep);

	if(!overlap) return 0;

	switch(face_id)
	{
	case CF_LEFT:
		PG_Set_Vector2(normal, -1, 0);
		assert(_Build_Contacts(gp, &clipBox, CA_X, contacts, &csize)); 
		break;
	case CF_BOTTOM:
		PG_Set_Vector2(normal, 0, -1);
		assert(_Build_Contacts(gp, &clipBox, CA_Y, contacts, &csize));
		break;
	case CF_RIGHT:
		PG_Set_Vector2(normal, 1, 0);
		assert(_Build_Contacts(gp, &clipBox, CA_X, contacts, &csize));
		break;
	case CF_TOP:
		PG_Set_Vector2(normal, 0, 1);
		assert(_Build_Contacts(gp, &clipBox, CA_Y, contacts, &csize));
		break;
	default:
		assert(0);
		break;
	}

	pAcc = PyObject_Malloc(sizeof(pgVector2));
	pAcc->real = pAcc->imag = 0;
	pSplitAcc = PyObject_Malloc(sizeof(pgVector2));
	pSplitAcc->real = pSplitAcc->imag = 0;
	for(i = 0; i < csize; ++i)
	{
		contact = (pgContact*)PG_ContactNew(ref, inc);
		contact->pos = contacts[i];
		contact->normal = normal;
		c_rotate(&(contact->pos), ref->fRotation);
		contact->pos = c_sum(contact->pos, ref->vecPosition);
		c_rotate(&(contact->normal), ref->fRotation);

		contact->ppAccMoment = PyObject_Malloc(sizeof(pgVector2*));
		*(contact->ppAccMoment) = pAcc;
		contact->ppSplitAccMoment = PyObject_Malloc(sizeof(pgVector2*));
		*(contact->ppSplitAccMoment) = pSplitAcc;

		contact->weight = csize;
		contact->depth = minDep;
		PyList_Append(contactList, (PyObject*)contact);
	}

	return 1;

}
