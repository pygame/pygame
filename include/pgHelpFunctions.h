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

#ifndef _PHYSICS_HELPFUNCTIONS_H_
#define _PHYSICS_HELPFUNCTIONS_H_

#include "pgphysics.h"

/**
 * Tries to retrieve the double value from a python object.
 * Taken from the pygame codebase.
 *
 * @param obj The PyObject to get the double from.
 * @param val Pointer to the double to store the value in.
 * @return 0 on failure, 1 on success
 */
int
DoubleFromObj (PyObject* obj, double* val);

/**
 * Returns a two-value tuple containing floats representing the passed
 * PyVector2.
 *
 * @param v2 The PyVector2 to convert to a tw-value tuple.
 * @return A tuple.
 */
PyObject*
FromPhysicsVector2ToPoint (PyVector2 v2);

/**
* Return a clamp value between low and high value
*
* @param 
* @return 
*/
double
PG_Clamp(double x,double low,double high);

#endif /* _PHYSICS_HELPFUNCTIONS_H_ */
