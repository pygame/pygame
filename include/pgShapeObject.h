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

#ifndef _PHYSICS_SHAPE_H_
#define _PHYSICS_SHAPE_H_

#include "pgphysics.h"

/**
 * TODO
 *
 * @param shape
 * @param refbody
 */
int PyShapeObject_UpdateAABB (PyShapeObject *shape, PyBodyObject *refbody);

/**
 * TODO
 *
 * @param shape
 * @param refbody
 * @param incbody
 * @param conactlist
 */
int PyShapeObject_Collision (PyShapeObject *shape, PyBodyObject *refbody,
    PyBodyObject *incbody, PyObject *contactlist);

/**
 * Python C API export hook
 *
 * @param c_api Pointer to the C API array.
 */
void PyShapeObject_ExportCAPI (void **c_api);

#endif /* _PHYSICS_SHAPE_H_ */
