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

#ifndef _PHYSICS_COLLISION_H_
#define _PHYSICS_COLLISION_H_

#include "pgphysics.h"

/**
 * TODO
 */
typedef struct
{
    //assert body2 is the incident rigid body
    //and body1 is the reference rigid body
    PyJointObject joint;

    PyVector2 pos;
    PyVector2 normal;
    PyVector2 dv;
    double depth;
    double weight;
    double resist;
    double kFactor, tFactor;
    PyVector2** ppAccMoment;
    PyVector2** ppSplitAccMoment;
} PyContact;

/**
 * TODO
 */
typedef enum
{
    MOVING_AWAY,
    RESTING,
    MOVING_TOWARD
} CollisionType;

/**
 * TODO
 */
typedef enum
{
    CA_X = 0,
    CA_Y = 1
} CollisionAxis;

/**
 * TODO
 */
typedef enum
{
    CF_LEFT,
    CF_BOTTOM,
    CF_RIGHT,
    CF_TOP
} CollisionFace;

/**
 * TODO
 *
 * @param
 * @param
 * @param
 * @param
 * @param
 * @return
 */
int Collision_LiangBarskey(AABBBox* box, PyVector2* p1, PyVector2* p2, 
    PyVector2* ans_p1, PyVector2* ans_p2);

/**
 * TODO
 *
 * @param
 * @param
 * @param
 * @param
 * @param
 * @param
 * @param
 * @param
 * @return
 */
int Collision_PartlyLB(AABBBox* box, PyVector2* p1, PyVector2* p2, 
    CollisionAxis axis, PyVector2* ans_p1, PyVector2* ans_p2,
    int* valid_p1, int* valid_p2);

/**
 * TODO
 *
 * @param
 * @param
 * @return
 */
PyJointObject* Collision_ContactNew(PyBodyObject* refBody,
    PyBodyObject* incidBody);

/**
 * TODO
 *
 * @param
 * @param
 */
void Collision_ApplyContact(PyObject* contactObject, double step);

/**
 * TODO
 *
 * @param refBody
 * @param incidBody
 */
PyObject* PyContact_New(PyBodyObject* refBody, PyBodyObject* incidBody);

#endif /* _PHYSICS_COLLISION_H_ */
