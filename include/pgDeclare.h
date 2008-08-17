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

#ifndef _PHYSICS_DECLARE_H_
#define _PHYSICS_DECLARE_H_

#include <Python.h>

/*
 * Internally used declarations.
 */

#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION <= 4
typedef int Py_ssize_t;
#endif

#ifdef PHYSICS_INTERNAL

extern PyTypeObject PyBody_Type;
extern PyTypeObject PyContact_Type;
extern PyTypeObject PyJoint_Type;
extern PyTypeObject PyDistanceJoint_Type;
extern PyTypeObject PyRevoluteJoint_Type;
extern PyTypeObject PyWorld_Type;
extern PyTypeObject PyShape_Type;
extern PyTypeObject PyRectShape_Type;

#define PHYSICS_BODY_INTERNAL
#define PHYSICS_JOINT_INTERNAL
#define PHYSICS_SHAPE_INTERNAL
#define PHYSICS_WORLD_INTERNAL
#define PHYSICS_MATH_INTERNAL

#define PyBody_Check(x) (PyObject_TypeCheck(x, &PyBody_Type))
#define PyContact_Check(x) (PyObject_TypeCheck(x, &PyContact_Type))
#define PyJoint_Check(x) (PyObject_TypeCheck(x, &PyJoint_Type))
#define PyDistanceJoint_Check(x) (PyObject_TypeCheck(x, &PyDistanceJoint_Type))
#define PyRevoluteJoint_Check(x) (PyObject_TypeCheck(x, &PyRevoluteJoint_Type))
#define PyWorld_Check(x) (PyObject_TypeCheck(x, &PyWorld_Type))
#define PyShape_Check(x) (PyObject_TypeCheck(x, &PyShape_Type))
#define PyRechtShape_Check(x) (PyObject_TypeCheck(x, &PyRectShape_Type))

#endif /* PHYSICS_INTERNAL */

#endif /* _PHYSICS_DECLARE_H_ */
