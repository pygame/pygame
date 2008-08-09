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

#include "pgVector2.h"

int PyMath_IsNearEqual(double a, double b)
{
    double rErr;
    if(IS_NEAR_ZERO(a - b)) return 1;

    if(fabs(b) > fabs(a))
        rErr = fabs((a - b) / b);
    else
        rErr = fabs((a - b) / a);

    return rErr <= RELATIVE_ZERO;
}

int PyMath_LessEqual(double a, double b)
{
    return a < b || PyMath_IsNearEqual(a, b);
}

int PyMath_MoreEqual(double a, double b)
{
    return a > b || PyMath_IsNearEqual(a, b);
}

int PyVector2_Equal(PyVector2* a, PyVector2* b)
{
    return PyMath_IsNearEqual(a->real, b->real) && PyMath_IsNearEqual(a->imag, b->imag);
}

PyVector2 PyVector2_MultiplyWithReal (PyVector2 a, double f)
{
    PyVector2 res;
    res.real = a.real * f;
    res.imag = a.imag * f;
    return res;
}

PyVector2 PyVector2_DivideWithReal (PyVector2 a, double f)
{
    PyVector2 res;
    res.real = a.real / f;
    res.imag = a.imag / f;
    return res;
}


PyVector2 PyVector2_fCross(double f, PyVector2 a)
{
    PyVector2 res;
    res.real = -f * a.imag;
    res.imag = f * a.real;
    return res;
}

PyVector2 PyVector2_Crossf(PyVector2 a, double f)
{
    PyVector2 res;
    res.real = f * a.imag;
    res.imag = -f * a.real;
    return res;
}   

PyVector2 PyVector2_Project(PyVector2 a, PyVector2 p)
{
    double lp;
    PyVector2_Normalize(&a);
    lp = PyVector2_Dot(a, p);
    return PyVector2_MultiplyWithReal (a, lp);
}

void PyMath_ExportCAPI (void **c_api)
{
    c_api[PHYSICS_MATH_FIRSTSLOT] = &PyMath_IsNearEqual;
    c_api[PHYSICS_MATH_FIRSTSLOT + 1] = &PyMath_LessEqual;
    c_api[PHYSICS_MATH_FIRSTSLOT + 2] = &PyMath_MoreEqual;
    c_api[PHYSICS_MATH_FIRSTSLOT + 3] = &PyVector2_Equal;
    c_api[PHYSICS_MATH_FIRSTSLOT + 4] = &PyVector2_MultiplyWithReal;
    c_api[PHYSICS_MATH_FIRSTSLOT + 5] = &PyVector2_DivideWithReal;
    c_api[PHYSICS_MATH_FIRSTSLOT + 6] = &PyVector2_fCross;
    c_api[PHYSICS_MATH_FIRSTSLOT + 7] = &PyVector2_Crossf;
    c_api[PHYSICS_MATH_FIRSTSLOT + 8] = &PyVector2_Project;
}
