/*
  pygame - Python Game Library
  Copyright (C) 2008 Marcus von Appen

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
#ifndef _PYGAME_MASK_H_
#define _PYGAME_MASK_H_

#include "bitmask.h"
#include "pgbase.h"

typedef struct {
    PyObject_HEAD
    bitmask_t *mask;
} PyMask;
#define PyMask_AsBitmask(x) (((PyMask*)x)->mask)
#define PYGAME_MASK_FIRSTSLOT 0
#define PYGAME_MASK_NUMSLOTS 2
#ifndef PYGAME_MASK_INTERNAL
#define PyMask_Type                                             \
    (*(PyTypeObject*)PyGameMask_C_API[PYGAME_MASK_FIRSTSLOT+0])
#define PyMask_Check(x)                                                 \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameMask_C_API[PYGAME_MASK_FIRSTSLOT+0]))
#define PyMask_New                                                      \
    (*(PyObject*(*)(int,int))PyGameMask_C_API[PYGAME_MASK_FIRSTSLOT+1])

#endif /* PYGAME_MASK_INTERNAL */

/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyGameMask_C_API;
#else
static void **PyGameMask_C_API;
#endif

#define PYGAME_MASK_SLOTS                                   \
    (PYGAME_MASK_FIRSTSLOT + PYGAME_MASK_NUMSLOTS)
#define PYGAME_MASK_ENTRY "_PYGAME_MASK_CAPI"

static int
import_pygame2_mask (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.mask");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PYGAME_MASK_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyGameMask_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

#endif /* _PYGAME_MASK_H_ */
