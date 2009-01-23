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

#define PHYSICS_AABBOX_INTERNAL

#include <float.h>
#include "physicsmod.h"
#include "pgphysics.h"

AABBox
AABBox_New (double left, double right, double bottom, double top)
{
    AABBox box;
    box.left = left;
    box.right = right;
    box.bottom = bottom;
    box.top = top;
    return box;
}

void
AABBox_Reset (AABBox* box)
{
    if (!box)
        return;

    box->top = -DBL_MAX;
    box->right = -DBL_MAX;
    box->bottom = DBL_MAX;
    box->left = DBL_MAX;
}

void
AABBox_ExpandTo (AABBox* box, PyVector2* p)
{
    if (!box || !p)
        return;
    box->left = MIN (box->left, p->real);
    box->right = MAX (box->right, p->real);
    box->bottom = MIN (box->bottom, p->imag);
    box->top = MAX (box->top, p->imag);
}

int
AABBox_Overlaps (AABBox* boxA, AABBox* boxB, double eps)
{
    double from_x, from_y, to_x, to_y;

    if (!boxA || !boxB)
        return 0;

    from_x = MAX (boxA->left, boxB->left);
    from_y = MAX (boxA->bottom, boxB->bottom);
    to_x = MIN (boxA->right, boxB->right);
    to_y = MIN (boxA->top, boxB->top);

    return (from_x - eps <= to_x + eps) && (from_y - eps <= to_y + eps);
}

int
AABBox_Contains (AABBox* box, PyVector2* p, double eps)
{
    if (!box || !p)
        return 0;
    return box->left - eps < p->real && p->real < box->right + eps
        && box->bottom - eps < p->imag && p->imag < box->top + eps;
}

PyObject*
AABBox_AsFRect (AABBox *box)
{
    if (!box)
    {
        PyErr_SetString (PyExc_RuntimeError, "argument is NULL");
        return NULL;
    }
    return PyFRect_New (box->left, box->top, box->right - box->left,
        box->top - box->bottom);
}

AABBox*
AABBox_FromSequence (PyObject *seq)
{
    double x, y, w, h;
    AABBox *box;
    if (!seq || !PySequence_Check (seq) || PySequence_Size (seq) < 4)
    {
        PyErr_SetString (PyExc_TypeError,
            "argument must be a 4-value sequence");
        return NULL;
    }

    if (!DoubleFromSeqIndex (seq, 0, &x))
        return NULL;
    if (!DoubleFromSeqIndex (seq, 1, &y))
        return NULL;
    if (!DoubleFromSeqIndex (seq, 2, &w))
        return NULL;
    if (!DoubleFromSeqIndex (seq, 3, &h))
        return NULL;
    
    box = PyMem_New (AABBox, 1);
    if (!box)
        return NULL;

    box->top = y;
    box->left = x;
    box->bottom = y + h;
    box->right = x + w;
    return box;
}

AABBox*
AABBox_FromRect (PyObject *rect)
{
    double t, l, b, r;
    AABBox *box;

    if (!rect)
    {
        PyErr_SetString (PyExc_TypeError,
            "argument must be a Rect, FRect or 4-value sequence");
        return NULL;
    }

    if (PyRect_Check (rect))
    {
        l = ((PyRect*)rect)->x;
        t = ((PyRect*)rect)->y;
        b = ((PyRect*)rect)->h + t;
        r = ((PyRect*)rect)->w + l;
    }
    else if (PyFRect_Check (rect))
    {
        l = ((PyFRect*)rect)->x;
        t = ((PyFRect*)rect)->y;
        b = ((PyFRect*)rect)->h + t;
        r = ((PyFRect*)rect)->w + l;
    }
    else
    {
        box = AABBox_FromSequence (rect);
        if (box)
            return box;
        PyErr_SetString (PyExc_TypeError,
            "argument must be a Rect, FRect or 4-value sequence");
        return NULL;
    }
    
    box = PyMem_New (AABBox, 1);
    if (!box)
        return NULL;

    box->top = t;
    box->left = l;
    box->bottom = b;
    box->right = r;
    return box;
}

void
aabbox_export_capi (void **capi)
{
    capi[PHYSICS_AABBOX_FIRSTSLOT+0] = AABBox_New;
    capi[PHYSICS_AABBOX_FIRSTSLOT+1] = AABBox_Reset;
    capi[PHYSICS_AABBOX_FIRSTSLOT+2] = AABBox_ExpandTo;
    capi[PHYSICS_AABBOX_FIRSTSLOT+3] = AABBox_Overlaps;
    capi[PHYSICS_AABBOX_FIRSTSLOT+4] = AABBox_Contains;
    capi[PHYSICS_AABBOX_FIRSTSLOT+5] = AABBox_AsFRect;
    capi[PHYSICS_AABBOX_FIRSTSLOT+6] = AABBox_FromSequence;
    capi[PHYSICS_AABBOX_FIRSTSLOT+7] = AABBox_FromRect;
}
