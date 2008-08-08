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

#include <float.h>
#include "pgAABBBox.h"

AABBBox AABB_Gen(double left, double right, double bottom, double top)
{
    AABBBox box;
    box.left = left;
    box.right = right;
    box.bottom = bottom;
    box.top = top;
    return box;
}

void AABB_Clear(AABBBox* box)
{
    box->left = DBL_MAX;
    box->bottom = DBL_MAX;
    box->top = -DBL_MAX;
    box->right = -DBL_MAX;
}

void AABB_ExpandTo(AABBBox* box, PyVector2* p)
{
    box->left = MIN(box->left, p->real);
    box->right = MAX(box->right, p->real);
    box->bottom = MIN(box->bottom, p->imag);
    box->top = MAX(box->top, p->imag);
}

int AABB_IsOverlap(AABBBox* boxA, AABBBox* boxB, double eps)
{
    double from_x, from_y, to_x, to_y;
    from_x = MAX(boxA->left, boxB->left);
    from_y = MAX(boxA->bottom, boxB->bottom);
    to_x = MIN(boxA->right, boxB->right);
    to_y = MIN(boxA->top, boxB->top);
    return from_x - eps <= to_x + eps && from_y - eps <= to_y + eps;
}

int AABB_IsIn(PyVector2* p, AABBBox* box, double eps)
{
    return box->left - eps < p->real && p->real < box->right + eps
        && box->bottom - eps < p->imag && p->imag < box->top + eps;
}
