/*
  pygame physics - Pygame physics module

  Copyright (C) 2008 Zhang Fan

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public
  License along with this library; if not, write to the Free
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#define PHYSICS_SHAPE_INTERNAL
#include <float.h>
#include <assert.h>
#include "pgDeclare.h"
#include "pgAABBBox.h"
#include "pgVector2.h"
#include "pgCollision.h"
#include "pgBodyObject.h"
#include "pgShapeObject.h"

static PyObject* _ShapeNew(PyTypeObject *type, PyObject *args, PyObject *kwds);
static void _ShapeObjectDestroy(PyShapeObject* shape);

static void _RectShapeUpdateAABB(PyBodyObject* body);
static int _RectShape_init(PyRectShapeObject* shape,PyObject *args, PyObject *kwds);
static void _RectShape_InitInternal (PyRectShapeObject *shape, double width,
    double height, double seta);
static PyObject* _RectShapeNew(PyTypeObject *type, PyObject *args, PyObject *kwds);


static int _RectShapeCollision(PyBodyObject* selfBody,
    PyBodyObject* incidBody, PyObject* contactList);


/* collision test for RectShape */

#define MAX_CONTACTS 16

typedef struct _Candidate_
{
	PyVector2 normal;
	PyVector2 contacts[MAX_CONTACTS];
	double kFactors[MAX_CONTACTS];
	int contact_size;
	double min_depth;
}_Candidate;


static int _ClipTest(AABBBox* box, PyVector2* points, _Candidate* candi)
{
	int  i, i1;
	int apart;
	PyVector2 pf, pt;
	int has_ip[4];

	memset(has_ip, 0, sizeof(has_ip));
	apart = 1;
	candi->contact_size = 0;
	for(i = 0; i < 4; ++i)
	{
		i1 = (i + 1)%4;
		if(Collision_LiangBarskey(box, &points[i], &points[i1], &pf, &pt))
		{
			apart = 0;
			if(PyVector2_Equal(&pf, &points[i]))
				has_ip[i] = 1;
			else
				candi->contacts[candi->contact_size++] = pf;

			if(PyVector2_Equal(&pt, &points[i1]))
				has_ip[i1] = 1;
			else
				candi->contacts[candi->contact_size++] = pt;
		}
	}

	if(apart) return 0;

	for(i = 0; i < 4; ++i)
		if(has_ip[i])
			candi->contacts[candi->contact_size++] = points[i];

	return 1;

}

static void _SATFindCollisionProperty(PyBodyObject* selfBody, PyBodyObject* incBody, 
							 AABBBox* selfBox, AABBBox* incBox, _Candidate *candi,
							 PyBodyObject** ans_ref, PyBodyObject** ans_inc)
{
	int i, k;
	double deps[4];
	double min_dep[2];
	int face_id[2];
	PyVector2 conts[2][MAX_CONTACTS];
	AABBBox* box[2];
	PyBodyObject* self[2], * inc[2];
	PyVector2 refR, incidR;
	int size;
	double tmp1, tmp2;
		
	for(i = 0; i < candi->contact_size; ++i)
	{
		conts[0][i] = candi->contacts[i];
		conts[1][i] = PyBodyObject_GetRelativePos(incBody, selfBody, &conts[0][i]);
	}
	box[0] = selfBox;
	box[1] = incBox;
	self[0] = inc[1] = selfBody;
	inc[0] = self[1] = incBody;

	for(k = 0; k <= 1; ++k)
	{
		memset(deps, 0, sizeof(deps));
		for(i = 0; i < candi->contact_size; ++i)
		{
			deps[CF_LEFT] += fabs(conts[k][i].real - box[k]->left);
			deps[CF_RIGHT] += fabs(box[k]->right - conts[k][i].real);
			deps[CF_BOTTOM] += fabs(conts[k][i].imag - box[k]->bottom);
			deps[CF_TOP] += fabs(box[k]->top - conts[k][i].imag);
		}

		min_dep[k] = DBL_MAX;
		for(i = CF_LEFT; i <= CF_TOP; ++i)
			if(min_dep[k] > deps[i])
			{
				face_id[k] = i;
				min_dep[k] = deps[i];
			}
	}

	//now select min depth one
	k = min_dep[0] < min_dep[1] ? 0 : 1;

	candi->min_depth = min_dep[k];
	size = candi->contact_size;
	candi->contact_size = 0;
	switch(face_id[k])
	{
	case CF_LEFT:
		PyVector2_Set(candi->normal, -1, 0);
		for(i = 0; i < size; ++i)
			if(!PyMath_IsNearEqual(conts[k][i].real, box[k]->left))
				candi->contacts[candi->contact_size++] = conts[k][i];
		break;
	case CF_RIGHT:
		PyVector2_Set(candi->normal, 1, 0);
		for(i = 0; i < size; ++i)
			if(!PyMath_IsNearEqual(conts[k][i].real, box[k]->right))
				candi->contacts[candi->contact_size++] = conts[k][i];
		break;
	case CF_BOTTOM:
		PyVector2_Set(candi->normal, 0, -1);
		for(i = 0; i < size; ++i)
			if(!PyMath_IsNearEqual(conts[k][i].imag, box[k]->bottom))
				candi->contacts[candi->contact_size++] = conts[k][i];
		break;
	case CF_TOP:
		PyVector2_Set(candi->normal, 0, 1);
		for(i = 0; i < size; ++i)
			if(!PyMath_IsNearEqual(conts[k][i].imag, box[k]->top))
				candi->contacts[candi->contact_size++] = conts[k][i];		
		break;
	default:
		assert(0);
	}
	
	//translate to global coordinate
	PyVector2_Rotate(&(candi->normal), self[k]->fRotation);
	for(i = 0; i < candi->contact_size; ++i)
	{
		PyVector2_Rotate(&(candi->contacts[i]), self[k]->fRotation);
		candi->contacts[i] = c_sum(candi->contacts[i], self[k]->vecPosition);
		//precompute kFactor
		refR = c_diff(candi->contacts[i], self[k]->vecPosition);
		incidR = c_diff(candi->contacts[i], inc[k]->vecPosition);
		tmp1 = PyVector2_Dot(PyVector2_fCross(PyVector2_Cross(refR, candi->normal), refR), candi->normal)
			 /((PyShapeObject*)self[k]->shape)->rInertia;
		tmp2 = PyVector2_Dot(PyVector2_fCross(PyVector2_Cross(incidR, candi->normal), incidR), candi->normal)
			 /((PyShapeObject*)inc[k]->shape)->rInertia;

		candi->kFactors[i] = 1/self[k]->fMass + 1/inc[k]->fMass + tmp1 + tmp2;
	}

	*ans_ref = self[k];
	*ans_inc = inc[k];

}

static int _RectShapeCollision(PyBodyObject* selfBody, PyBodyObject* incidBody, 
							   PyObject* contactList)
{
	
	PyVector2 p_in_self[4], p_in_inc[4];
	AABBBox box_self, box_inc;
	int i;
	PyRectShapeObject * self, * inc;
	_Candidate candi;
	PyContact* contact;
	PyVector2 * pAcc, * pSplitAcc;
	PyBodyObject* ans_ref, * ans_inc;

	
	self = (PyRectShapeObject*)selfBody->shape;
	inc = (PyRectShapeObject*)incidBody->shape;

	p_in_self[0] = PyBodyObject_GetRelativePos(selfBody, incidBody, &(inc->bottomleft));
	p_in_self[1] = PyBodyObject_GetRelativePos(selfBody, incidBody, &(inc->bottomright));
	p_in_self[2] = PyBodyObject_GetRelativePos(selfBody, incidBody, &(inc->topright));
	p_in_self[3] = PyBodyObject_GetRelativePos(selfBody, incidBody, &(inc->topleft));
	
	p_in_inc[0] = PyBodyObject_GetRelativePos(incidBody, selfBody, &(self->bottomleft));
	p_in_inc[1] = PyBodyObject_GetRelativePos(incidBody, selfBody, &(self->bottomright));
	p_in_inc[2] = PyBodyObject_GetRelativePos(incidBody, selfBody, &(self->topright));
	p_in_inc[3] = PyBodyObject_GetRelativePos(incidBody, selfBody, &(self->topleft));


	box_self = AABB_Gen(self->bottomleft.real, self->topright.real,
		self->bottomleft.imag, self->topright.imag);
	box_inc = AABB_Gen(inc->bottomleft.real, inc->topright.real,
		inc->bottomleft.imag, inc->topright.imag);

	if(!_ClipTest(&box_self, p_in_self, &candi)) return 0;

	assert(0);

	if(AABB_IsIn(&p_in_inc[0], &box_inc, 0.f))
		candi.contacts[candi.contact_size++] = self->bottomleft;
	if(AABB_IsIn(&p_in_inc[1], &box_inc, 0.f))
		candi.contacts[candi.contact_size++] = self->bottomright;
	if(AABB_IsIn(&p_in_inc[2], &box_inc, 0.f))
		candi.contacts[candi.contact_size++] = self->topright;
	if(AABB_IsIn(&p_in_inc[3], &box_inc, 0.f))
		candi.contacts[candi.contact_size++] = self->topleft;

	_SATFindCollisionProperty(selfBody, incidBody, &box_self, &box_inc, &candi, &ans_ref, &ans_inc);

	
	pAcc = PyObject_Malloc(sizeof(PyVector2));
	pAcc->real = pAcc->imag = 0;
	pSplitAcc = PyObject_Malloc(sizeof(PyVector2));
	pSplitAcc->real = pSplitAcc->imag = 0;
	for(i = 0; i < candi.contact_size; ++i)
	{
		contact = (PyContact*)PyContact_New(ans_ref, ans_inc);
		contact->pos = candi.contacts[i];
		contact->normal = candi.normal;

		contact->ppAccMoment = PyObject_Malloc(sizeof(PyVector2*));
		*(contact->ppAccMoment) = pAcc;
		contact->ppSplitAccMoment = PyObject_Malloc(sizeof(PyVector2*));
		*(contact->ppSplitAccMoment) = pSplitAcc;

		contact->weight = candi.contact_size;
		contact->depth = candi.min_depth;
		contact->kFactor = candi.kFactors[i];

		PyList_Append(contactList, (PyObject*)contact);
	}

	return 1;
}


#undef MAX_CONTACTS





/* C API */
static PyObject *PyShape_New (void);
static PyObject *PyRectShape_New (double width, double height, double seta);


PyTypeObject PyShape_Type =
{
    PyObject_HEAD_INIT(NULL)
    0,
    "physics.Shape",            /* tp_name */
    sizeof(PyShapeObject),      /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor)_ShapeObjectDestroy,/* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    0,                          /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "",                         /* tp_doc */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    0,		  	                /* tp_methods */
    0,                          /* tp_members */
    0,                          /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    0,                          /* tp_init */
    0,                          /* tp_alloc */
    _ShapeNew,                  /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0                           /* tp_del */
};

//functions of pgShapeObject

static PyObject* _ShapeNew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    /* In case we have arguments in the python code, parse them later
     * on.
     */
    PyShapeObject* shape = (PyShapeObject*)type->tp_alloc(type, 0);
    memset(&(shape->box), 0, sizeof(shape->box));
    shape->rInertia = 0;
    shape->Collision = NULL;
    shape->UpdateAABB = NULL;
    
    return (PyObject*) shape;
}

static void _ShapeObjectDestroy(PyShapeObject* shape)
{
    shape->ob_type->tp_free((PyObject*)shape);
}

PyTypeObject PyRectShape_Type =
{
    PyObject_HEAD_INIT(NULL)
    0,
    "physics.RectShape",        /* tp_name */
    sizeof(PyRectShapeObject),  /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) 0,				/* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    0,                          /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "",                         /* tp_doc */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    0,				            /* tp_methods */
    0,	                        /* tp_members */
    0,				            /* tp_getset */
    0,				            /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)_RectShape_init,   /* tp_init */
    0,                          /* tp_alloc */
    _RectShapeNew,              /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0                           /* tp_del */
};

//functions of pgRectShape

static void _RectShapeUpdateAABB(PyBodyObject* body)
{
    int i;
    PyVector2 gp[4];

    if(((PyShapeObject*)body->shape)->type == ST_RECT)
    {
        PyRectShapeObject *p = (PyRectShapeObject*)body->shape;
		
        AABB_Clear(&(p->shape.box));

        gp[0] = PyBodyObject_GetGlobalPos(body, &(p->bottomleft));
        gp[1] = PyBodyObject_GetGlobalPos(body, &(p->bottomright));
        gp[2] = PyBodyObject_GetGlobalPos(body, &(p->topright));
        gp[3] = PyBodyObject_GetGlobalPos(body, &(p->topleft));

        for(i = 0; i < 4; ++i)
            AABB_ExpandTo(&(p->shape.box), &gp[i]);
    }
}

static void _RectShape_InitInternal (PyRectShapeObject *shape, double width,
    double height, double seta)
{
    PyVector2_Set(shape->bottomleft, -width/2, -height/2);
    PyVector2_Set(shape->bottomright, width/2, -height/2);
    PyVector2_Set(shape->topright, width/2, height/2);
    PyVector2_Set(shape->topleft, -width/2, height/2);
    PyVector2_Rotate(&(shape->bottomleft), seta);
    PyVector2_Rotate(&(shape->bottomright), seta);
    PyVector2_Rotate(&(shape->topright), seta);
    PyVector2_Rotate(&(shape->topleft), seta);
}

static int _RectShape_init(PyRectShapeObject* shape,PyObject *args, PyObject *kwds)
{
    double width, height, seta = 0;
    if (PyShape_Type.tp_init((PyObject*)shape, args, kwds) < 0)
        return -1;
    if (!PyArg_ParseTuple (args, "dd|d", &width, &height, &seta))
        return -1;

    _RectShape_InitInternal (shape, width, height, seta);
    return 0;
}

static PyObject* _RectShapeNew(PyTypeObject *type, PyObject *args,
    PyObject *kwds)
{
    PyRectShapeObject *shape = (PyRectShapeObject*) _ShapeNew (type, args, kwds);
    if (!shape)
        return NULL;
    
    shape->shape.UpdateAABB = _RectShapeUpdateAABB;
    shape->shape.Collision = _RectShapeCollision;
    shape->shape.type = ST_RECT;
    return (PyObject*)shape;
}

/* C API */
static PyObject *PyShape_New (void)
{
    return (PyObject*)_ShapeNew (&PyShape_Type, NULL, NULL);
}

static PyObject *PyRectShape_New (double width, double height, double seta)
{
    PyRectShapeObject *shape = (PyRectShapeObject*)
        _RectShapeNew (&PyRectShape_Type, NULL, NULL);
    _RectShape_InitInternal (shape, width, height, seta);
    return (PyObject*) shape;
}

void PyShapeObject_ExportCAPI (void **c_api)
{
    c_api[PHYSICS_SHAPE_FIRSTSLOT] = &PyShape_Type;
    c_api[PHYSICS_SHAPE_FIRSTSLOT + 1] = &PyShape_New;
    c_api[PHYSICS_SHAPE_FIRSTSLOT + 2] = &PyRectShape_Type;
    c_api[PHYSICS_SHAPE_FIRSTSLOT + 3] = &PyRectShape_New;
}

