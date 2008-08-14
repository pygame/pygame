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
 * pgCollision.h mainly contains structures and functions related with
   collision reaction functions. The main algorithm is a complex empirical 
   formula. You can find it in Helmut Garstenauer's thesis, Page 60. It also 
   employ split impulse to correct the position after collision reaction. You can
   learn this method thoroughly from Erin Canto's www.gphysics.com. I'm sorry i don't
   explain them in detail here, since it's a complicated mission for me while all the
   two references illustrate them perfectly:)

   Note the only relation between collision test and collision reaction is PyContact.
   So you can change two parts almost independently.

   TODO: fraction simulation is in construction. I'll commit the code as soon as it is stable.
 */

/*
 *
	PyContact includes essential data for collision reaction computation.
	@param body1 (inheriting from PyJointObject) the reference body
	@param body2 (inheriting from PyJointObject) the incident body
	@param pos the contact point's position
	@param dv the relative velocity between reference body and incident body(in reference body's local coordinate)
	@param depth the penetrating depth
	@param weight to average impulses caused by each contact point
	@param kFactor precomputed factor for impulse-based collision reaction calculation.
	@param tFactor similar with kFactor but is not used at the moment
	@param ppAccMoment the accumulated moment(impulse)
	@param ppSplitAccMoment the accumulated split moment(impulse)
 */
typedef struct
{
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
 * represent the collision face which collision happens on
 */
typedef enum
{
    CF_LEFT,
    CF_BOTTOM,
    CF_RIGHT,
    CF_TOP
} CollisionFace;

/**
 * Collision_LiangBarskey is an Liang-Barskey 2d line clipping method
 *
 * Directed line segment(p1, p2) will be clipped against the AABB box
 * and result in (ans_p1, ans_p2).
 *
 * @param box AABB box used for 2D line clipping
 * @param p1 Start point of the directed line segment to be clipped
 * @param p2 End point of the directed line segment to be clipped
 * @param ans_p1 Start point of clipped line
 * @param ans_p2 End point of clipped line
 * @return If there is no overlap, return 0. Otherwise return 1
 */
int Collision_LiangBarskey(AABBBox* box, PyVector2* p1, PyVector2* p2, 
    PyVector2* ans_p1, PyVector2* ans_p2);

/**
 * Collision_ApplyContact does collision reaction calculation based on
 * information in contactObject
 *
 * @contactObject a contact contains collision information
 * @param step time step
 */
void Collision_ApplyContact(PyObject* contactObject, double step);

/**
* PyContact_New creates a new contact connected with refBody and incidBody
*
* @param refBody the reference body
* @param incidBody the incident body
* @return a new PyContact
*/
PyObject* PyContact_New(PyBodyObject* refBody, PyBodyObject* incidBody);

#endif /* _PHYSICS_COLLISION_H_ */
