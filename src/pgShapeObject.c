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

#include "pgDeclare.h"
#include "pgphysics.h"
#include "pgAABBBox.h"
#include "pgBodyObject.h"
#include "pgShapeObject.h"
#include "pgCollision.h"
#include "pgVector2.h"

static PyObject* _ShapeNew(PyTypeObject *type, PyObject *args, PyObject *kwds);
static void _ShapeObjectDestroy(PyShapeObject* shape);

static void _RectShapeUpdateAABB(PyBodyObject* body);
static int _RectShape_init(PyRectShape* shape,PyObject *args, PyObject *kwds);
static void _RectShape_InitInternal (PyRectShape *shape, double width,
    double height, double seta);
static PyObject* _RectShapeNew(PyTypeObject *type, PyObject *args, PyObject *kwds);

static int _Get_Depth(PyBodyObject* refBody, PyBodyObject* incBody,
    int* faceId, double* min_dep, PyVector2* gp_in_ref, AABBBox* clipBox);
static int _SAT_Select(PyBodyObject* body1, PyBodyObject* body2,
    PyBodyObject** refBody, PyBodyObject** incBody,
    int* face_id, PyVector2* gp_in_ref, AABBBox* clipBox, double* minDep);
static int _Build_Contacts(PyVector2* gp, AABBBox* clipBox, int axis,
    PyVector2 contacts[], int* size);
static int _RectShapeCollision(PyBodyObject* selfBody,
    PyBodyObject* incidBody, PyObject* contactList);

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
    sizeof(PyRectShape),        /* tp_basicsize */
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
        PyRectShape *p = (PyRectShape*)body->shape;
		
        AABB_Clear(&(p->shape.box));

        gp[0] = PyBodyObject_GetGlobalPos(body, &(p->bottomleft));
        gp[1] = PyBodyObject_GetGlobalPos(body, &(p->bottomright));
        gp[2] = PyBodyObject_GetGlobalPos(body, &(p->topright));
        gp[3] = PyBodyObject_GetGlobalPos(body, &(p->topleft));

        for(i = 0; i < 4; ++i)
            AABB_ExpandTo(&(p->shape.box), &gp[i]);
    }
}

static void _RectShape_InitInternal (PyRectShape *shape, double width,
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

static int _RectShape_init(PyRectShape* shape,PyObject *args, PyObject *kwds)
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
    PyRectShape *shape = (PyRectShape*) _ShapeNew (type, args, kwds);
    if (!shape)
        return NULL;
    
    shape->shape.UpdateAABB = _RectShapeUpdateAABB;
    shape->shape.Collision = _RectShapeCollision;
    shape->shape.type = ST_RECT;
    return (PyObject*)shape;
}

static int _Get_Depth(PyBodyObject* refBody, PyBodyObject* incBody,
    int* faceId, double* min_dep, PyVector2* gp_in_ref, AABBBox* clipBox)
{
#define _EPS_DEPTH 1e-2

    int i, apart;
    PyRectShape *ref, *inc;
    double deps[4];

    ref = (PyRectShape*)refBody->shape;
    inc = (PyRectShape*)incBody->shape;
    memset(gp_in_ref, 0, sizeof(gp_in_ref));
    memset(deps, 0, sizeof(deps));
    
    gp_in_ref[0] = PyBodyObject_GetRelativePos(refBody, incBody,
        &(inc->bottomleft));
    gp_in_ref[1] = PyBodyObject_GetRelativePos(refBody, incBody,
        &(inc->bottomright));
    gp_in_ref[2] = PyBodyObject_GetRelativePos(refBody, incBody,
        &(inc->topright));
    gp_in_ref[3] = PyBodyObject_GetRelativePos(refBody, incBody,
        &(inc->topleft));

    *clipBox = AABB_Gen(ref->bottomleft.real, ref->topright.real,
        ref->bottomleft.imag, ref->topright.imag);

    apart = 1;
    for(i = 0; i < 4; ++i)
        if(AABB_IsIn(&gp_in_ref[i], clipBox, _EPS_DEPTH))
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

static int _SAT_Select(PyBodyObject* body1, PyBodyObject* body2,
    PyBodyObject** refBody, PyBodyObject** incBody,
    int* face_id, PyVector2* gp_in_ref, AABBBox* clipBox, double* minDep)
{
    double min_dep[2];
    int id[2];
    PyVector2 gp[2][4];
    AABBBox cb[2];
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

static int _Build_Contacts(PyVector2* gp, AABBBox* clipBox, int axis,
    PyVector2 contacts[], int* size)
{
    int i, i1;
    int apart = 1;
    int has_ip[4];
    PyVector2 pf, pt;
    int valid_pf, valid_pt;

    *size = 0;
    memset(has_ip, 0, sizeof(has_ip));

    for(i = 0; i < 4; ++i)
    {
        i1 = (i+1)%4;
        if(Collision_PartlyLB(clipBox, &gp[i], &gp[i1], axis, 
                &pf, &pt, &valid_pf, &valid_pt))
        {
            apart = 0;
            if(valid_pf)
            {
                if(PyVector2_Equal(&pf, &gp[i]))
                    has_ip[i] = 1;
                else
                {
                    //assert(0);
                    contacts[(*size)++] = pf;
                }
            }
            if(valid_pt)
            {
                if(PyVector2_Equal(&pt, &gp[i1]))
                    has_ip[i1] = 1;
                else
                {
                    //assert(0);
                    contacts[(*size)++] = pt;
                }
            }
        }
    }
    for(i = 0; i < 4; ++i)
        if(has_ip[i])
        {
            //assert(0);
            contacts[(*size)++] = gp[i];
        }

    return !apart;
}

static int _RectShapeCollision(PyBodyObject* selfBody,
    PyBodyObject* incidBody, PyObject* contactList)
{
#define MAX_CONTACTS 8

    PyBodyObject* ref = NULL, *inc = NULL;
    int face_id;
    PyVector2 gp[4];
    PyVector2 contacts[MAX_CONTACTS];
    int csize = 0;
    PyVector2 normal = { 0, 0 };
    AABBBox clipBox;
    int overlap;
    PyVector2* pAcc, * pSplitAcc;
    PyContact* contact;
    int i;
    double minDep;
    PyVector2 refR, incidR;
    double tmp1, tmp2;

    //for test
    PyVector2 tgp[4];


    overlap = _SAT_Select(selfBody, incidBody,
        &ref, &inc, &face_id, gp, &clipBox, &minDep);

    if(!overlap) return 0;

    switch(face_id)
    {
    case CF_LEFT:
        PyVector2_Set (normal, -1, 0);
        _Build_Contacts(gp, &clipBox, CA_X, contacts, &csize);
        break;
    case CF_BOTTOM:
        PyVector2_Set (normal, 0, -1);
        _Build_Contacts(gp, &clipBox, CA_Y, contacts, &csize);
        break;
    case CF_RIGHT:
        PyVector2_Set (normal, 1, 0);
        _Build_Contacts(gp, &clipBox, CA_X, contacts, &csize);
        break;
    case CF_TOP:
        PyVector2_Set (normal, 0, 1);
        _Build_Contacts(gp, &clipBox, CA_Y, contacts, &csize);
        break;
    default:
        assert(0);
        break;
    }

    //for test
/*
  for(i = 0; i < 4; ++i)
  tgp[i] = PG_GetGlobalPos(ref, &(((pgRectShape*)ref->shape)->point[i]));
  glColor3f(1.f, 0.0f, 0.5f);
  glBegin(GL_LINES);
  switch(face_id)
  {
  case CF_LEFT:
  glVertex2f(tgp[0].real, tgp[0].imag);
  glVertex2f(tgp[3].real, tgp[3].imag);
  break;
  case CF_RIGHT:
  glVertex2f(tgp[1].real, tgp[1].imag);
  glVertex2f(tgp[2].real, tgp[2].imag);
  break;
  case CF_TOP:
  glVertex2f(tgp[2].real, tgp[2].imag);
  glVertex2f(tgp[3].real, tgp[3].imag);
  break;
  case CF_BOTTOM:
  glVertex2f(tgp[0].real, tgp[0].imag);
  glVertex2f(tgp[1].real, tgp[1].imag);
  break;
  default:
  break;
  }
  glEnd();
*/

    pAcc = PyObject_Malloc(sizeof(PyVector2));
    pAcc->real = pAcc->imag = 0;
    pSplitAcc = PyObject_Malloc(sizeof(PyVector2));
    pSplitAcc->real = pSplitAcc->imag = 0;

    //for test
    //printf("face id: %d; csize: %d\n", face_id, csize);

    for(i = 0; i < csize; ++i)
    {
        contact = (PyContact*) PyContact_New(ref, inc);
        contact->pos = contacts[i];
        contact->normal = normal;
        PyVector2_Rotate(&(contact->pos), ref->fRotation);
        contact->pos = c_sum(contact->pos, ref->vecPosition);
        PyVector2_Rotate(&(contact->normal), ref->fRotation);

        contact->ppAccMoment = PyObject_Malloc(sizeof(PyVector2*));
        *(contact->ppAccMoment) = pAcc;
        contact->ppSplitAccMoment = PyObject_Malloc(sizeof(PyVector2*));
        *(contact->ppSplitAccMoment) = pSplitAcc;

        contact->weight = csize;
        contact->depth = minDep;

        //precompute kFactor
        refR = c_diff(contact->pos, selfBody->vecPosition);
        incidR = c_diff(contact->pos, incidBody->vecPosition);
        tmp1 = PyVector2_Dot(PyVector2_fCross(PyVector2_Cross(refR, contact->normal), refR), contact->normal)
            /((PyShapeObject*)selfBody->shape)->rInertia;
        tmp2 = PyVector2_Dot(PyVector2_fCross(PyVector2_Cross(incidR, contact->normal), incidR), contact->normal)
            /((PyShapeObject*)incidBody->shape)->rInertia;

        contact->kFactor = 1/selfBody->fMass + 1/incidBody->fMass + tmp1 + tmp2;
        PyList_Append(contactList, (PyObject*)contact);
    }
    return 1;
}

/* C API */
static PyObject *PyShape_New (void)
{
    return (PyObject*)_ShapeNew (&PyShape_Type, NULL, NULL);
}

static PyObject *PyRectShape_New (double width, double height, double seta)
{
    PyRectShape *shape = (PyRectShape*)_RectShapeNew (&PyRectShape_Type, NULL, NULL);
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
