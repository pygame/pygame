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

#ifndef ABS
#define ABS(x) ( ((x) < 0) ?  -(x) : (x) )
#endif
#ifndef MAX
#define MAX(x, y) ( ((x) > (y)) ? (x) : (y) )
#endif
#ifndef MIN
#define MIN(x, y) ( ((x) < (y)) ? (x) : (y) )
#endif

/**
 * Zero tolerance constants for vector calculations.
 */
#define ZERO_EPSILON 1e-6
#define RELATIVE_ZERO 1e-6

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#define IS_NEAR_ZERO(num) (fabs(num) <= ZERO_EPSILON)

/**
 * 2D vector definition.
 */
typedef Py_complex PyVector2;

#define PyVector2_Check(x) (PyObject_TypeCheck(op, &PyComplex_Type))
#define PyVector2_CheckExact(x) ((op)->ob_type == &PyComplex_Type)
#define PyVector2_Set(vec, x, y)                \
    (vec).real = (x);                           \
    (vec).imag = (y);

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
#define PHYSICS_MATH_NUMSLOTS 9
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

#endif /* PYGAME_MATH_INTERNAL */

/**
 * TODO
 */
typedef struct
{
    double left;
    double bottom;
    double right;
    double top;
} AABBBox;

/**
 * TODO
 */
typedef struct
{
    PyObject_HEAD

    double    fMass;
    PyVector2 vecLinearVelocity;
    double    fAngleVelocity;
    int       bStatic;
    
    PyVector2 vecPosition;
    double    fRotation;
    PyVector2 vecImpulse;
    PyVector2 vecForce;
    double    fTorque;
    
    double    fRestitution;
    double    fFriction;
	double	  fLinearVelDamping;
	double	  fAngleVelDamping;
    
    PyObject* shape;
    
    PyVector2 cBiasLV;
    double    cBiasW;

} PyBodyObject;

#define PHYSICS_BODY_FIRSTSLOT (PHYSICS_MATH_FIRSTSLOT + PHYSICS_MATH_NUMSLOTS)
#define PHYSICS_BODY_NUMSLOTS 4
#ifndef PHYSICS_BODY_INTERNAL
#define PyBody_Check(x)                                                 \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyPhysics_C_API[PHYSICS_BODY_FIRSTSLOT+0]))
#define PyBody_New                                                      \
    (*(PyObject*(*)(void))PyPhysics_C_API[PHYSICS_BODY_FIRSTSLOT+1])
#define PyBody_SetShape                                                 \
    (*(int(*)(PyObject*,PyObject*))PyPhysics_C_API[PHYSICS_BODY_FIRSTSLOT+2])
#define PyBody_GetGlobalPos                                             \
    (*(PyVector2(*)(PyObject*,PyVector2))PyPhysics_C_API[PHYSICS_BODY_FIRSTSLOT+3])
#endif /* PYGAME_BODY_INTERNAL */

/**
 * TODO
 */
typedef struct _PyJointObject PyJointObject;
struct _PyJointObject
{
    PyObject_HEAD

    PyObject* body1;
    PyObject* body2;
    int       isCollideConnect;
    void      (*SolveConstraintPosition)(PyJointObject* joint,double stepTime);
    void      (*SolveConstraintVelocity)(PyJointObject* joint,double stepTime);
    void      (*Destroy)(PyJointObject* joint);
};

/**
 * TODO
 */
typedef struct
{
    PyJointObject joint;
    double        distance;
    PyVector2     anchor1;
    PyVector2     anchor2;
} PyDistanceJointObject;

typedef struct  
{
    PyJointObject joint;
    //notice : we can't set local anchor directly, because init position may violate the constraint.
    PyVector2	anchor1;
    PyVector2	anchor2;
} PyRevoluteJointObject;

#define PHYSICS_JOINT_FIRSTSLOT \
    (PHYSICS_BODY_FIRSTSLOT + PHYSICS_BODY_NUMSLOTS)
#define PHYSICS_JOINT_NUMSLOTS 8
#ifndef PHYSICS_JOINT_INTERNAL
#define PyJoint_Check(x)                                                \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyPhysics_C_API[PHYSICS_JOINT_FIRSTSLOT+0]))
#define PyJoint_New                                                     \
    (*(PyObject*(*)(PyObject*,PyObject*,int))PyPhysics_C_API[PHYSICS_JOINT_FIRSTSLOT+1])
#define PyDistanceJoint_Check(x)                                        \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyPhysics_C_API[PHYSICS_JOINT_FIRSTSLOT+2]))
#define PyDistanceJoint_New                                             \
    (*(PyObject*(*)(PyObject*,PyObject*,int))PyPhysics_C_API[PHYSICS_JOINT_FIRSTSLOT+3])
#define PyDistanceJoint_SetAnchors                                      \
    (*(int(*)(PyObject*,PyVector2,PyVector2))PyPhysics_C_API[PHYSICS_JOINT_FIRSTSLOT+4])
#define PyRevoluteJoint_Check(x)                                        \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyPhysics_C_API[PHYSICS_JOINT_FIRSTSLOT+5]))
#define PyRevoluteJoint_New                                             \
    (*(PyObject*(*)(PyObject*,PyObject*,int))PyPhysics_C_API[PHYSICS_JOINT_FIRSTSLOT+6])
#define PyRevoluteJoint_SetAnchorsFromConnectWorldAnchor                \
    (*(int(*)(PyObject*,PyVector2))PyPhysics_C_API[PHYSICS_JOINT_FIRSTSLOT+7])
#endif /* PYGAME_JOINT_INTERNAL */

/**
 * TODO
 */
typedef enum
{
    ST_RECT,
    ST_CIRCLE
} ShapeType;

/**
 * TODO
 */
typedef struct _PyShapeObject PyShapeObject;
struct _PyShapeObject
{
    PyObject_HEAD

    AABBBox   box;
    ShapeType type;
    double    rInertia; //Rotor inertia  

    // virtual functions
    int       (*Collision)(PyBodyObject* selfBody, PyBodyObject* incidBody,
                           PyObject* contactList);
    void      (*UpdateAABB)(PyBodyObject* body);
};

/**
 * TODO
 */
typedef struct
{
    PyShapeObject shape;

    PyVector2 bottomleft;
    PyVector2 bottomright;
    PyVector2 topright;
    PyVector2 topleft;
} PyRectShapeObject;

#define PHYSICS_SHAPE_FIRSTSLOT \
    (PHYSICS_JOINT_FIRSTSLOT + PHYSICS_JOINT_NUMSLOTS)
#define PHYSICS_SHAPE_NUMSLOTS 4
#ifndef PHYSICS_SHAPE_INTERNAL
#define PyShape_Check(x)                                                \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyPhysics_C_API[PHYSICS_SHAPE_FIRSTSLOT+0]))
#define PyShape_New                                                     \
    (*(PyObject*(*)(void))PyPhysics_C_API[PHYSICS_SHAPE_FIRSTSLOT+1])
#define PyRectShape_Check(x)                                            \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyPhysics_C_API[PHYSICS_SHAPE_FIRSTSLOT+2]))
#define PyRectShape_New                                                 \
    (*(PyObject*(*)(double,double,double))PyPhysics_C_API[PHYSICS_SHAPE_FIRSTSLOT+3])
#endif /* PYGAME_SHAPE_INTERNAL */


/**
 * TODO
 */
typedef struct
{
    PyObject_HEAD
    
    PyObject*  bodyList;
    PyObject*  jointList;
    PyObject*  contactList;

    PyVector2  vecGravity;
    double     fDamping;

    double     fStepTime;
    double     fTotalTime;
    AABBBox    worldBox;

} PyWorldObject;

#define PHYSICS_WORLD_FIRSTSLOT \
    (PHYSICS_SHAPE_FIRSTSLOT + PHYSICS_SHAPE_NUMSLOTS)
#define PHYSICS_WORLD_NUMSLOTS 7
#ifndef PHYSICS_WORLD_INTERNAL
#define PyWorld_Check(x)                                                \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+0])
#define PyWorld_New                                                     \
    (*(PyObject*(*)(void))PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+1])
#define PyWorld_AddBody                                                 \
    (*(PyObject*(*)(PyObject*,PyObject*))PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+2])
#define PyWorld_RemoveBody                                              \
    (*(PyObject*(*)(PyObject*,PyObject*))PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+3])
#define PyWorld_AddJoint                                                \
    (*(PyObject*(*)(PyObject*,PyObject*))PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+4])
#define PyWorld_RemoveJoint                                             \
    (*(PyObject*(*)(PyObject*,PyObject*))PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+5])
#define PyWorld_Update                                                  \
    (*(int(*)(PyObject*,double))PyPhysics_C_API[PHYSICS_WORLD_FIRSTSLOT+6])

#endif /* PYGAME_WORLD_INTERNAL */

/**
 * C API slots.
 */
static void **PyPhysics_C_API;
#define PHYSICS_API_SLOTS (PHYSICS_WORLD_FIRSTSLOT + PHYSICS_WORLD_NUMSLOTS)
#define PHYSICS_CAPI_ENTRY "_PHYSICS_C_API"

static int import_physics (void)
{
    PyObject *_module = PyImport_ImportModule ("physics");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PHYSICS_CAPI_ENTRY);
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

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initphysics (void);

#endif /*_PYGAME_PHYSICS_H_*/
