/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2008 Marcus von Appen

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
#ifndef _PYGAME_SDLTTF_H_
#define _PYGAME_SDLTTF_H_

#include <SDL.h>
#include <SDL_ttf.h>

#include "pgbase.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ASSERT_TTF_INIT(x)                                              \
    if (!TTF_WasInit())                                                 \
    {                                                                   \
        PyErr_SetString(PyExc_PyGameError, "ttf subsystem not initialized"); \
        return (x);                                                     \
    }

#define PYGAME_SDLTTF_FIRSTSLOT 0
#define PYGAME_SDLTTF_NUMSLOTS 0
#ifndef PYGAME_SDLTTF_INTERNAL
#endif /* PYGAME_SDLTTF_INTERNAL */

typedef struct
{
    PyFont pyfont;
    TTF_Font *font;
} PySDLFont_TTF;
#define PySDLFont_TTF_AsFont(x) (((PySDLFont_TTF*)x)->font)
#define PYGAME_SDLTTFFONT_FIRSTSLOT \
    (PYGAME_SDLTTF_FIRSTSLOT + PYGAME_SDLTTF_NUMSLOTS)
#define PYGAME_SDLTTFFONT_NUMSLOTS 1
#ifndef PYGAME_SDLTTFFONT_INTERNAL
#define PySDLFont_TTF_Type \
    (*(PyTypeObject*)PyGameSDLTTF_C_API[PYGAME_SDLTTFFONT_FIRSTSLOT+0])
#define PySDLFont_TTF_Check(x)                                                 \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameSDLTTF_C_API[PYGAME_SDLTTFFONT_FIRSTSLOT+0]))
#define PySDLFont_TTF_New                                                   \
    (*(PyObject*(*)(char*,int))PyGameSDLTTF_C_API[PYGAME_SDLTTFFONT_FIRSTSLOT+1])
#endif /* PYGAME_SDLTTFFONT_INTERNAL */

/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyGameSDLTTF_C_API;
#else
static void **PyGameSDLTTF_C_API;
#endif

#define PYGAME_SDLTTF_SLOTS                                    \
    (PYGAME_SDLTTFFONT_FIRSTSLOT + PYGAME_SDLTTFFONT_NUMSLOTS)
#define PYGAME_SDLTTF_ENTRY "_PYGAME_SDLTTF_CAPI"

static int
import_pygame2_sdlttf_base (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.sdlttf.base");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PYGAME_SDLTTF_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyGameSDLTTF_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

#ifdef __cplusplus
}
#endif

#endif /* _PYGAME_SDLTTF_H_ */
