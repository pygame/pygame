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

#ifndef _PHYSICS_BODY_H_
#define _PHYSICS_BODY_H_

#include "pgphysics.h"

/**
 * TODO
 *
 * @param body
 * @param gravity
 * @param dt
 */
void PyBodyObject_FreeUpdateVel(PyBodyObject* body, PyVector2 gravity,
    double dt);

/**
 * TODO
 *
 * @param body
 * @param dt
 */
void PyBodyObject_FreeUpdatePos (PyBodyObject* body, double dt);

/**
 * TODO
 *
 * @param body
 * @param dt
 */
void PyBodyObject_CorrectPos(PyBodyObject* body, double dt);

/**
 * Transform point local_p's position from body's local coordinate to
 * the world's global one.
 * TODO: is the local coordinate necessary?  anyway let it alone right
 * now.
 *
 * @param body
 * @param local_p
 * @return
 */
PyVector2 PyBodyObject_GetGlobalPos(PyBodyObject* body, PyVector2* local_p);

/**
 * Translate vector from coordinate B to coordinate A
 *
 * @param bodyA
 * @param bodyB
 * @param p_in_B
 * @return
 */
PyVector2 PyBodyObject_GetRelativePos(PyBodyObject* bodyA, PyBodyObject* bodyB,
    PyVector2* p_in_B);

/**
 * Get velocity with a local point of a body,assume center is local (0,0)
 *
 * @param body
 * @param localPoint
 * @return
 */
PyVector2 PyBodyObject_GetLocalPointVelocity(PyBodyObject* body,
    PyVector2 localPoint);

/**
 * Python C API export hook
 *
 * @param c_api Pointer to the C API array.
 */
void PyBodyObject_ExportCAPI (void **c_api);

#endif /* _PHYSICS_BODY_H_ */
