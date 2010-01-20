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
#ifndef _PYGAME_OPENAL_H_
#define _PYGAME_OPENAL_H_

#include "pgcompat.h"
#include "pgdefines.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PYGAME_OPENAL_FIRSTSLOT 0
#define PYGAME_OPENAL_NUMSLOTS 0
#ifndef PYGAME_OPENAL_INTERNAL
#endif /* PYGAME_OPENAL_INTERNAL */

/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyGameOpenAL_C_API;
#else
static void **PyGameOpenAL_C_API;
#endif

#define PYGAME_OPENAL_SLOTS (PYGAME_OPENAL_FIRSTSLOT + PYGAME_OPENAL_NUMSLOTS)
#define PYGAME_OPENAL_ENTRY "_PYGAME_OPENAL_CAPI"

static int
import_pygame2_openal (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.openal");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PYGAME_OPENAL_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyGameOpenAL_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

#ifdef __cplusplus
}
#endif

#endif /* _PYGAME_OPENAL_H_ */
