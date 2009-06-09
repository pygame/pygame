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

#ifndef _PHYSICS_PHYSICSMOD_H_
#define _PHYSICS_PHYSICSMOD_H_

#include "pgcompat.h"
#include <structmember.h>

#define PHYSICS_MATH_INTERNAL
#define PHYSICS_AABBOX_INTERNAL
#define PHYSICS_WORLD_INTERNAL
#define PHYSICS_BODY_INTERNAL
#define PHYSICS_CONTACT_INTERNAL
#define PHYSICS_SHAPE_INTERNAL
#define PHYSICS_RECTSHAPE_INTERNAL
#define PHYSICS_JOINT_INTERNAL

#include "pgphysics.h"

/* Math API */
int PyMath_IsNearEqual (double a, double b);
int PyMath_LessEqual (double a, double b);
int PyMath_MoreEqual (double a, double b);
int PyVector2_Equal (PyVector2 a, PyVector2 b);
PyVector2 PyVector2_MultiplyWithReal (PyVector2 a, double f);
PyVector2 PyVector2_DivideWithReal (PyVector2 a, double f);
PyVector2 PyVector2_fCross (double f, PyVector2 a);
PyVector2 PyVector2_Crossf (PyVector2 a, double f);
PyVector2 PyVector2_Project (PyVector2 a, PyVector2 p);
PyObject* PyVector2_AsTuple (PyVector2 v);
int PyVector2_FromSequence (PyObject *seq, PyVector2 *vector);
PyVector2 PyVector2_Transform (PyVector2 v, PyVector2 vlocal, double vrotation,
    PyVector2 tlocal, double trotation);
void PyVector2_TransformMultiple (PyVector2 *vin, PyVector2 *vout, int count,
    PyVector2 vlocal, double vrotation, PyVector2 tlocal, double trotation);

void math_export_capi (void **capi);

/* AABBox API */
AABBox AABBox_New (double left, double right, double bottom, double top);
void AABBox_Reset (AABBox* box);
void AABBox_ExpandTo (AABBox* box, PyVector2* p);
int AABBox_Overlaps (AABBox* boxA, AABBox* boxB, double eps);
int AABBox_Contains (AABBox* box, PyVector2* p, double eps);
PyObject* AABBox_AsFRect (AABBox *box);
int AABBox_FromSequence (PyObject *seq, AABBox *box);
int AABBox_FromRect (PyObject *rect, AABBox *box);
void aabbox_export_capi (void **capi);

/* World API */
extern PyTypeObject PyWorld_Type;
#define PyWorld_Check(x) (PyObject_TypeCheck(x, &PyWorld_Type))
PyObject* PyWorld_New (void);
int PyWorld_AddBody (PyObject *world, PyObject *body);
int PyWorld_RemoveBody (PyObject *world, PyObject *body);
int PyWorld_AddJoint (PyObject *world, PyObject *joint);
int PyWorld_RemoveJoint (PyObject *world, PyObject *joint);
int PyWorld_Update (PyObject* world, double step);
int PyWorld_Update_FAST (PyWorld *world, double step);
void world_export_capi (void **capi);


/* Body API */
extern PyTypeObject PyBody_Type;
#define PyBody_Check(x) (PyObject_TypeCheck(x, &PyBody_Type))
#define PyBody_GetGlobalPos(body,v,r)              \
    {                                              \
        PyVector2_Rotate (&(v), (body)->rotation); \
        (r) = c_sum (v, (body)->position);         \
    }

#define PyBody_CorrectPosition(body,steptime)                           \
    if (!body->isstatic)                                                \
    {                                                                   \
        body->position = c_sum (body->position,                         \
            PyVector2_MultiplyWithReal (body->bias_lv, steptime));      \
        body->rotation += body->bias_w * steptime;                      \
    }

#define PyBody_UpdatePosition(body,steptime)                            \
    if (!body->isstatic)                                                \
    {                                                                   \
        body->position = c_sum (body->position,                         \
            PyVector2_MultiplyWithReal (body->linear_velocity, steptime)); \
        body->rotation += body->angle_velocity * steptime;              \
    }

#define PyBody_UpdateVelocity(body,gravity,steptime)                    \
    if (!body->isstatic)                                                \
    {                                                                   \
        PyVector2 __total;                                              \
        double __k1, __k2;                                              \
        __total = c_sum (body->force,                                   \
            PyVector2_MultiplyWithReal(gravity, body->mass));           \
        /* What about massless bodies? */                               \
        body->linear_velocity = c_sum (body->linear_velocity,           \
            PyVector2_MultiplyWithReal (__total, steptime / body->mass)); \
                                                                        \
        __k1 = CLAMP (1 - steptime * body->linear_vel_damping, 0.0, 1.0); \
        __k2 = CLAMP (1 - steptime * body->angle_vel_damping, 0.0, 1.0); \
        body->linear_velocity = PyVector2_MultiplyWithReal              \
            (body->linear_velocity, __k1);                              \
        body->angle_velocity *= __k2;                                   \
    }

PyObject* PyBody_New (PyObject *shape);
PyObject* PyBody_CheckCollision (PyObject *body1, PyObject *body2);
PyObject* PyBody_CheckCollision_FAST (PyBody *body1, PyBody *body2);
PyObject* PyBody_GetPoints (PyObject *body);
void body_export_capi (void **capi);

/* Shape API */
extern PyTypeObject PyShape_Type;
#define PyShape_Check(x) (PyObject_TypeCheck(x, &PyShape_Type))
PyObject* PyShape_Collide (PyObject *shape1, PyVector2 pos1, double rot1,
    PyObject *shape2, PyVector2 pos2, double rot2, int *refid);
PyObject* PyShape_Collide_FAST (PyShape *shape1, PyVector2 pos1, double rot1,
    PyShape *shape2, PyVector2 pos2, double rot2, int *refid);
int PyShape_Update (PyObject *shape, PyObject *body);
int PyShape_Update_FAST (PyShape *shape, PyBody *body);
int PyShape_GetAABBox (PyObject *shape, AABBox* box);
int PyShape_GetAABBox_FAST (PyShape *shape, AABBox* box);
PyVector2* PyShape_GetVertices (PyObject *shape, Py_ssize_t *count);
PyVector2* PyShape_GetVertices_FAST (PyShape *shape, Py_ssize_t *count);
void shape_export_capi (void **capi);

/* RectShape API */
extern PyTypeObject PyRectShape_Type;
#define PyRectShape_Check(x) (PyObject_TypeCheck(x, &PyRectShape_Type))
PyObject* PyRectShape_New (AABBox box);
void rectshape_export_capi (void **capi);

/* Contact API */
extern PyTypeObject PyContact_Type;
#define PyContact_Check(x) (PyObject_TypeCheck(x, &PyContact_Type))
PyObject* PyContact_New (void);
int PyContact_Collision (PyObject *contact, double steptime);
int PyContact_Collision_FAST (PyContact *contact, double steptime);
void contact_export_capi (void **capi);

/* Joint API */
extern PyTypeObject PyJoint_Type;
#define PyJoint_Check(x) (PyObject_TypeCheck(x, &PyJoint_Type))
int PyJoint_SolveConstraints (PyObject *joint, double steptime);
int PyJoint_SolveConstraints_FAST (PyJoint *joint, double steptime);
void joint_export_capi (void **capi);

/* Collision API */
collisionfunc PyCollision_GetCollisionFunc (ShapeType t1, ShapeType t2,
    int *swap); 

#endif /* _PHYSICS_PHYSICSMOD_H_ */
