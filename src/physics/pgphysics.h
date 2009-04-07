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

#ifndef _PHYSICS_H_
#define _PHYSICS_H_

#include <math.h>
#include <Python.h>
#include "pgcompat.h"
#include "pgbase.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MIN
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef ABS
#define ABS(x) (((x) < 0) ? -(x) : (x))
#endif

/**
 * Zero tolerance constants for vector calculations.
 */
#define ZERO_EPSILON 1e-6
#define RELATIVE_ZERO 1e-6
#define OVERLAP_ZERO 1e-8

#define IS_NEAR_ZERO(num) (fabs(num) <= ZERO_EPSILON)
#define CLAMP(x,low,high)                                               \
    ((x < high) ? ((x > low) ? x : low) : ((high > low) ? high : low))

/**
 * 2D vector definition.
 */
typedef Py_complex PyVector2;

#define PyVector2_Check(x) (PyObject_TypeCheck(op, &PyComplex_Type))
#define PyVector2_CheckExact(x) ((op)->ob_type == &PyComplex_Type)
#define PyVector2_Set(vec, x, y)                \
    {                                           \
        (vec).real = (x);                       \
        (vec).imag = (y);                       \
    }

#define PyVector2_GetLengthSquare(x) ((x).real * (x).real + (x).imag * (x).imag)
#define PyVector2_GetLength(x) (sqrt(PyVector2_GetLengthSquare(x)))
#define PyVector2_Dot(x, y) ((x).real * (y).real + (x).imag * (y).imag)
#define PyVector2_Cross(x, y) ((x).real * (y).imag - (x).imag * (y).real)
#define PyVector2_Normalize(x)                          \
    {                                                   \
        double __pg_tmp = PyVector2_GetLength(*(x));    \
        (x)->real /=  __pg_tmp;                         \
        (x)->imag /=  __pg_tmp;                         \
    }

#define PyVector2_Rotate(x, a)                          \
    {                                                   \
        double __pg_x = (x)->real;                      \
        double __pg_y = (x)->imag;                      \
        (x)->real = __pg_x * cos(a) - __pg_y * sin(a);  \
        (x)->imag = __pg_x * sin(a) + __pg_y * cos(a);  \
    }

#define PHYSICS_MATH_FIRSTSLOT 0
#define PHYSICS_MATH_NUMSLOTS 13
#ifndef PHYSICS_MATH_INTERNAL
#define PyMath_IsNearEqual                                              \
    (*(int(*)(double,double))PyPhysics_C_API[PHYSICS_MATH_FIRSTSLOT+0])
#define PyMath_LessEqual                                                \
    (*(int(*)(double,double))PyPhysics_C_API[PHYSICS_MATH_FIRSTSLOT+1])
#define PyMath_MoreEqual                                                \
    (*(int(*)(double,double))PyPhysics_C_API[PHYSICS_MATH_FIRSTSLOT+2])

#define PyVector2_Equal                                                 \
    (*(int(*)(PyVector2,PyVector2))PyPhysics_C_API[PHYSICS_MATH_FIRSTSLOT+3])
#define PyVector2_MultiplyWithReal                                      \
    (*(PyVector2(*)(PyVector2,double))PyPhysics_C_API[PHYSICS_MATH_FIRSTSLOT+4])
#define PyVector2_DivideWithReal                                        \
    (*(PyVector2(*)(PyVector2,double))PyPhysics_C_API[PHYSICS_MATH_FIRSTSLOT+5])
#define PyVector2_fCross                                                \
    (*(PyVector2(*)(double,PyVector2))PyPhysics_C_API[PHYSICS_MATH_FIRSTSLOT+6])
#define PyVector2_Crossf                                                \
    (*(PyVector2(*)(PyVector2,double))PyPhysics_C_API[PHYSICS_MATH_FIRSTSLOT+7])
#define PyVector2_Project                                               \
    (*(PyVector2(*)(PyVector2,PyVector2))PyPhysics_C_API[PHYSICS_MATH_FIRSTSLOT+8])
#define PyVector2_AsTuple                                               \
    (*(PyObject*(*)(PyVector2))PyPhysics_C_API[PHYSICS_MATH_FIRSTSLOT+9])
#define PyVector2_FromSequence                                          \
    (*(int(*)(PyObject*, PyVector2*))PyPhysics_C_API[PHYSICS_MATH_FIRSTSLOT+10])
#define PyVector2_Transform                                             \
    (*(PyVector2(*)(PyVector2,PyVector2,double,PyVector2,double))PyPhysics_C_API[PHYSICS_MATH_FIRSTSLOT+11])
#define PyVector2_TransformMultiple                                     \
    (*(PyVector2(*)(PyVector2*,PyVector2*,int,double,PyVector2,double))PyPhysics_C_API[PHYSICS_MATH_FIRSTSLOT+12])
#endif /* PHYSICS_MATH_INTERNAL */

/**
 * AABBox
 */
typedef struct
{
    double top;
    double left;
    double right;
    double bottom;
} AABBox;

#define PHYSICS_AABBOX_FIRSTSLOT \
    (PHYSICS_MATH_FIRSTSLOT + PHYSICS_MATH_NUMSLOTS)
#define PHYSICS_AABBOX_NUMSLOTS 8
#ifndef PHYSICS_AABBOX_INTERNAL
#define AABBox_New                                                     \
    (*(AABBox(*)(double,double,double,double))PyPhysics_C_API[PHYSICS_AABBOX_FIRSTSLOT+0])
#define AABBox_Reset                                                   \
    (*(void(*)(AABBox*))PyPhysics_C_API[PHYSICS_AABBOX_FIRSTSLOT+1])
#define AABBox_ExpandTo                                                \
    (*(void(*)(AABBox*,PyVector2*))PyPhysics_C_API[PHYSICS_AABBOX_FIRSTSLOT+2])
#define AABBox_Overlaps                                                \
    (*(int(*)(AABBox*,AABBox*,double))PyPhysics_C_API[PHYSICS_AABBOX_FIRSTSLOT+3])
#define AABBox_Contains                                                \
    (*(int(*)(AABBox*,PyVector2*,double))PyPhysics_C_API[PHYSICS_AABBOX_FIRSTSLOT+4])
#define AABBox_AsFRect                                                 \
    (*(PyObject*(*)(AABBox*))PyPhysics_C_API[PHYSICS_AABBOX_FIRSTSLOT+5])
#define AABBox_FromSequence                                            \
    (*(int(*)(PyObject*,AABBox*))PyPhysics_C_API[PHYSICS_AABBOX_FIRSTSLOT+6])
#define AABBox_FromRect                                                \
    (*(int(*)(PyObject*,AABBox*))PyPhysics_C_API[PHYSICS_AABBOX_FIRSTSLOT+7])
#endif /* PHYSICS_AABBOX_INTERNAL */

/**
 */
typedef struct
{
    PyObject_HEAD

    PyObject *dict;
    PyObject *shape;
    double    mass;
    int       isstatic : 1;
    double    rotation;
    PyVector2 position;
    PyVector2 impulse;
    PyVector2 force;
    double    torque;
    PyVector2 linear_velocity;
    double    angle_velocity;
    double    restitution;
    double    friction;
    double    linear_vel_damping;
    double    angle_vel_damping;
    PyVector2 bias_lv;
    double    bias_w;
} PyBody;

#define PHYSICS_BODY_FIRSTSLOT \
    (PHYSICS_AABBOX_FIRSTSLOT + PHYSICS_AABBOX_NUMSLOTS)
#define PHYSICS_BODY_NUMSLOTS 5
#ifndef PHYSICS_BODY_INTERNAL
#define PyBody_Type                                             \
    (*(PyTypeObject*)PyPhysics_C_API[PHYSICS_BODY_FIRSTSLOT+0])
#define PyBody_Check(x)                                                 \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyPhysics_C_API[PHYSICS_BODY_FIRSTSLOT+0]))
#define PyBody_New                                                      \
    (*(PyObject*(*)(PyObject*))PyPhysics_C_API[PHYSICS_BODY_FIRSTSLOT+1])
#define PyBody_CheckCollision                                           \
    (*(PyObject*(*)(PyObject*,PyObject*))PyPhysics_C_API[PHYSICS_BODY_FIRSTSLOT+2])
#define PyBody_CheckCollision_FAST                                      \
    (*(PyObject*(*)(PyBody*,PyBody*))PyPhysics_C_API[PHYSICS_BODY_FIRSTSLOT+3])
#define PyBody_GetPoints                                                \
    (*(PyObject*(*)(PyObject*))PyPhysics_C_API[PHYSICS_BODY_FIRSTSLOT+4])
#endif /* PHYSICS_BODY_INTERNAL */

typedef enum
{
    UNKNOWN = -1,
    ST_RECT,
    ST_CIRCLE,
} ShapeType;

/**
 */
typedef struct _PyShape PyShape;
struct _PyShape
{
    PyObject_HEAD

    ShapeType   type;
    double      inertia;
    double      rotation;
    PyObject   *dict;

    PyVector2*  (*get_vertices)(PyShape*,Py_ssize_t*);
    int         (*get_aabbox)(PyShape*,AABBox*);
    int         (*update)(PyShape*,PyBody*);
};

/**
 */
typedef PyObject* (*collisionfunc)(PyShape*,PyVector2,double,PyShape*,PyVector2,double,int*);

#define PHYSICS_SHAPE_FIRSTSLOT \
    (PHYSICS_BODY_FIRSTSLOT + PHYSICS_BODY_NUMSLOTS)
#define PHYSICS_SHAPE_NUMSLOTS 9
#ifndef PHYSICS_SHAPE_INTERNAL
#define PyShape_Type                                            \
    (*(PyTypeObject*)PyPhysics_C_API[PHYSICS_SHAPE_FIRSTSLOT+0])
#define PyShape_Check(x)                                                \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyPhysics_C_API[PHYSICS_SHAPE_FIRSTSLOT+0]))
#define PyShape_Collide                                                 \
    (*(PyObject*(*)(PyObject*,PyVector2,double,PyObject*,PyVector2,double,int*))PyPhysics_C_API[PHYSICS_SHAPE_FIRSTSLOT+1])
#define PyShape_Collide_FAST                                            \
    (*(PyObject*(*)(PyShape*,PyVector2,double,PyShape*,PyVector2,double,int*))PyPhysics_C_API[PHYSICS_SHAPE_FIRSTSLOT+2])
#define PyShape_Update                                                  \
    (*(int(*)(PyObject*,PyObject*))PyPhysics_C_API[PHYSICS_SHAPE_FIRSTSLOT+3])
#define PyShape_Update_FAST                                             \
    (*(int(*)(PyShape*,PyBody*))PyPhysics_C_API[PHYSICS_SHAPE_FIRSTSLOT+4])
#define PyShape_GetAABBox                                              \
    (*(int(*)(PyObject*,AABBox*))PyPhysics_C_API[PHYSICS_SHAPE_FIRSTSLOT+5])
#define PyShape_GetAABBox_FAST                                         \
    (*(int(*)(PyShape*,AABBox*))PyPhysics_C_API[PHYSICS_SHAPE_FIRSTSLOT+6])
#define PyShape_GetVertices                                             \
    (*(PyVector2*(*)(PyObject*,Py_ssize_t*))PyPhysics_C_API[PHYSICS_SHAPE_FIRSTSLOT+7])
#define PyShape_GetVertices_FAST                                        \
    (*(PyVector2*(*)(PyShape*,Py_ssize_t*))PyPhysics_C_API[PHYSICS_SHAPE_FIRSTSLOT+8])
#endif /* PHYSICS_SHAPE_INTERNAL */

/**
 */
typedef struct
{
    PyShape  shape;
    AABBox   box;
    
    PyVector2 topleft;
    PyVector2 topright;
    PyVector2 bottomleft;
    PyVector2 bottomright;
} PyRectShape;

#define PHYSICS_RECTSHAPE_FIRSTSLOT \
    (PHYSICS_SHAPE_FIRSTSLOT + PHYSICS_SHAPE_NUMSLOTS)
#define PHYSICS_RECTSHAPE_NUMSLOTS 2
#ifndef PHYSICS_RECTSHAPE_INTERNAL
#define PyRectShape_Type                                            \
    (*(PyTypeObject*)PyPhysics_C_API[PHYSICS_RECTSHAPE_FIRSTSLOT+0])
#define PyRectShape_Check(x)                                            \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyPhysics_C_API[PHYSICS_RECTSHAPE_FIRSTSLOT+0]))
#define PyRectShape_New                                                 \
    (*(PyObject*(*)(AABBox))PyPhysics_C_API[PHYSICS_RECTSHAPE_FIRSTSLOT+1])
#endif /* PHYSICS_SHAPE_INTERNAL */

/**
 *
 */
typedef struct _PyJoint PyJoint;
struct _PyJoint
{
    PyObject_HEAD
    PyObject *dict;
    PyObject *body1;
    PyObject *body2;
    int       iscollideconnect : 1;
    void     (*solve_constraints)(PyJoint*,double);
};

#define PHYSICS_JOINT_FIRSTSLOT                                 \
    (PHYSICS_RECTSHAPE_FIRSTSLOT + PHYSICS_RECTSHAPE_NUMSLOTS)
#define PHYSICS_JOINT_NUMSLOTS 2
#ifndef PHYSICS_JOINT_INTERNAL
#define PyJoint_Type                                                    \
    (*(PyTypeObject*)PyPhysics_C_API[PHYSICS_JOINT_FIRSTSLOT+0])
#define PyJoint_Check(x)                                                \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyPhysics_C_API[PHYSICS_JOINT_FIRSTSLOT+0]))
#define PyJoint_New                                                 \
    (*(PyObject*(*)(void))PyPhysics_C_API[PHYSICS_JOINT_FIRSTSLOT+1])
#endif /* PHYSICS_SHAPE_INTERNAL */

/**
 */
typedef struct
{
    PyJoint joint;

    PyVector2 position;
    PyVector2 normal;
    PyVector2 dv;

    double depth;
    double weight;
    double resist;
    double kfactor;
    double tfactor;
    
    PyVector2 acc_moment;
    PyVector2 split_acc_moment;
} PyContact;

#define PHYSICS_CONTACT_FIRSTSLOT \
    (PHYSICS_JOINT_FIRSTSLOT + PHYSICS_JOINT_NUMSLOTS)
#define PHYSICS_CONTACT_NUMSLOTS 2
#ifndef PHYSICS_CONTACT_INTERNAL
#define PyContact_Type                                                  \
    (*(PyTypeObject*)PyPhysics_C_API[PHYSICS_CONTACT_FIRSTSLOT+0])
#define PyContact_Check(x)                                              \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyPhysics_C_API[PHYSICS_CONTACT_FIRSTSLOT+0]))
#define PyContact_New                                                   \
    (*(PyObject*(*)(void))PyPhysics_C_API[PHYSICS_CONTACT_FIRSTSLOT+1])
#endif /* PHYSICS_CONTACT_INTERNAL */

/**
 * 
 */
typedef struct
{
    PyObject_HEAD
    
    PyObject  *dict;
    PyObject  *bodylist;
    PyObject  *jointlist;
    PyObject  *contactlist;
    PyVector2  gravity;
    AABBox     area;
    double     damping;
    double     steptime;
    double     totaltime;
} PyWorld;

#define PHYSICS_WORLD_FIRSTSLOT \
    (PHYSICS_CONTACT_FIRSTSLOT + PHYSICS_CONTACT_NUMSLOTS)
#define PHYSICS_WORLD_NUMSLOTS 8
#ifndef PHYSICS_WORLD_INTERNAL
#define PyWorld_Type                                                    \
    (*(PyTypeObject*)PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+0])
#define PyWorld_Check(x)                                                \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+0]))
#define PyWorld_New                                                     \
    (*(PyObject*(*)(void))PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+1])
#define PyWorld_AddBody                                                 \
    (*(int(*)(PyObject*,PyObject*))PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+2])
#define PyWorld_RemoveBody                                              \
    (*(int(*)(PyObject*,PyObject*))PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+3])
#define PyWorld_AddJoint                                                \
    (*(int(*)(PyObject*,PyObject*))PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+4])
#define PyWorld_RemoveJoint                                             \
    (*(int(*)(PyObject*,PyObject*))PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+5])
#define PyWorld_Update                                                  \
    (*(int(*)(PyObject*,double))PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+6])
#define PyWorld_Update_FAST                                             \
    (*(int(*)(PyWorld*,double))PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+7])
#endif /* PHYSICS_WORLD_INTERNAL */

/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyPhysics_C_API;
#else
static void **PyPhysics_C_API;
#endif

#define PHYSICS_SLOTS \
    (PHYSICS_WORLD_FIRSTSLOT + PHYSICS_WORLD_NUMSLOTS)
#define PHYSICS_ENTRY "_PHYSICS_CAPI"

static int
import_pygame2_physics (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.physics");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PHYSICS_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyPhysics_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

#ifdef __cplusplus
}
#endif

#endif /* _PHYSICS_H_ */
