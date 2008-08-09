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

#ifndef _PYGAME_VECTOR2_H_
#define _PYGAME_VECTOR2_H_

#define PHYSICS_MATH_INTERNAL
#include "pgphysics.h"

/*
 * Internal math function declarations. Used to circumvent the C API
 * array.
 */

/**
 * TODO
 *
 * @param a
 * @param b
 * return
 */
int PyMath_IsNearEqual(double a, double b);

/**
 * TODO
 *
 * @param a
 * @param b
 * return
 */
int PyMath_LessEqual(double a, double b);

/**
 * TODO
 *
 * @param a
 * @param b
 * return
 */
int PyMath_MoreEqual(double a, double b);

/**
 * TODO
 *
 * @param a
 * @param b
 * return
 */
int PyVector2_Equal(PyVector2* a, PyVector2* b);

/**
 * TODO
 *
 * @param a
 * @param f
 * return
 */
PyVector2 PyVector2_MultiplyWithReal (PyVector2 a, double f);

/**
 * TODO
 *
 * @param a
 * @param f
 * return
 */
PyVector2 PyVector2_DivideWithReal (PyVector2 a, double f);

/**
 * TODO
 *
 * @param f
 * @param a
 * return
 */
PyVector2 PyVector2_fCross(double f, PyVector2 a);

/**
 * TODO
 *
 * @param a
 * @param f
 * return
 */
PyVector2 PyVector2_Crossf(PyVector2 a, double f);

/**
 * TODO
 *
 * @param a
 * @param p
 * return
 */
PyVector2 PyVector2_Project(PyVector2 a, PyVector2 p);

/**
 * Python C API export hook
 *
 * @param c_api Pointer to the C API array.
 */
void PyMath_ExportCAPI (void **c_api);

#endif
