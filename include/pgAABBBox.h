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

#ifndef _PHYSICS_AABBBOX_H_
#define _PHYSICS_AABBBOX_H_

#include "pgphysics.h"

/**
 * TODO
 *
 * @param left
 * @param right
 * @param bottom
 * @param top
 * @return 
 */
AABBBox AABB_Gen(double left, double right, double bottom, double top);

/**
 * TODO
 *
 * @param box
 * @param p
 */
void AABB_ExpandTo(AABBBox* box, PyVector2* p);

/**
 * TODO
 *
 * @param box
 */
void AABB_Clear(AABBBox* box);

/**
 * TODO
 *
 * @param boxA
 * @param boxB
 * @param eps
 * @return 
 */
int AABB_IsOverlap(AABBBox* boxA, AABBBox* boxB, double eps);

/**
 * TODO
 *
 * @param p
 * @param box
 * @param eps
 * @return 
 */
int AABB_IsIn(PyVector2* p, AABBBox* box, double eps);

#endif /* _PHYSICS_AABBBOX_H_ */
