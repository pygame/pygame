/*
  pygame - Python Game Library

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

#ifndef PGMASK_H
#define PGMASK_H

#include <Python.h>
#include "bitmask.h"

#define PYGAMEAPI_MASK_FIRSTSLOT 0
#define PYGAMEAPI_MASK_NUMSLOTS 1

typedef struct {
  PyObject_HEAD
  bitmask_t *mask;
} pgMaskObject;

#define pgMask_AsBitmap(x) (((pgMaskObject*)x)->mask)

#ifndef PYGAMEAPI_MASK_INTERNAL

#include "pgimport.h"

PYGAMEAPI_DEFINE_SLOTS(PyMASK_C_API, PYGAMEAPI_MASK_NUMSLOTS);

#define pgMask_Type     (*(PyTypeObject*) \
    PYGAMEAPI_GET_SLOT(PyMASK_C_API, 0))
#define pgMask_Check(x) ((x)->ob_type == &pgMask_Type)

#define import_pygame_mask() \
    _IMPORT_PYGAME_MODULE(mask, MASK, PyMASK_C_API)

#endif /* ~PYGAMEAPI_MASK_INTERNAL */

#endif /* ~PGMASK_H */
