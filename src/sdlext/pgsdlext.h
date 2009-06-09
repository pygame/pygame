/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners, 2007-2008 Marcus von Appen

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
#ifndef _PYGAME_SDLEXT_H_
#define _PYGAME_SDLEXT_H_

#include <SDL.h>

#include "pgbase.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    PyObject_HEAD
    PyObject *dict;     /* dict for subclassing */
    PyObject *weakrefs; /* Weakrefs for subclassing */
    PyObject *surface;  /* Surface associated with the array. */
    PyObject *parent;   /* Parent pixel array, if any. */
    Uint32 xstart;      /* X offset for subarrays */
    Uint32 ystart;      /* Y offset for subarrays */
    Uint32 xlen;        /* X segment length. */
    Uint32 ylen;        /* Y segment length. */
    Sint32 xstep;       /* X offset step width. */
    Sint32 ystep;       /* Y offset step width. */
    Uint32 padding;     /* Padding to get to the next x offset. */
} PyPixelArray;
#define PYGAME_SDLEXTPIXELARRAY_FIRSTSLOT 0
#define PYGAME_SDLEXTPIXELARRAY_NUMSLOTS 2
#ifndef PYGAME_SDLEXTPIXELARRAY_INTERNAL
#define PyPixelArray_Type \
    (*(PyTypeObject*)PyGameSDLExt_C_API[PYGAME_SDLPIXELARRAY_FIRSTSLOT+0])
#define PyPixelArray_Check(x)                                           \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameSDLExt_C_API[PYGAME_SDLPIXELARRAY_FIRSTSLOT+0]))
#define PyPixelArray_New                                                \
    (*(PyObject*(*)(PyObject*))PyGameSDLExt_C_API[PYGAME_SDLPIXELARRAY_FIRSTSLOT+1])
#endif /* PYGAME_SDLPIXELARRAY_INTERNAL */

/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyGameSDLExt_C_API;
#else
static void **PyGameSDLExt_C_API;
#endif

#define PYGAME_SDLEXT_SLOTS \
    (PYGAME_SDLEXTPIXELARRAY_FIRSTSLOT + PYGAME_SDLEXTPIXELARRAY_NUMSLOTS)
#define PYGAME_SDLEXT_ENTRY "_PYGAME_SDLEXT_CAPI"
    
static int
import_pygame2_sdlext_base (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.sdlext.base");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PYGAME_SDLEXT_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyGameSDLExt_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

#ifdef __cplusplus
}
#endif

#endif /* _PYGAME_SDLEXT_H_ */
