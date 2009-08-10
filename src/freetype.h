/*
  pygame - Python Game Library
  Copyright (C) 2009 Vicent Marti

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
#ifndef _PYGAME_FREETYPE_H_
#define _PYGAME_FREETYPE_H_

#define PGFT_PYGAME1_COMPAT
#define HAVE_PYGAME_SDL_VIDEO
#define HAVE_PYGAME_SDL_RWOPS

#include "pygame.h"
#include "pgcompat.h"

#if PY3
#   define IS_PYTHON_3
#endif

#include <ft2build.h>  
#include FT_FREETYPE_H 
#include FT_CACHE_H
#include FT_XFREE86_H

/**********************************************************
 * Global module constants
 **********************************************************/

/* Render styles */
#define FT_STYLE_NORMAL     0x00
#define FT_STYLE_BOLD		0x01
#define FT_STYLE_ITALIC     0x02
#define FT_STYLE_UNDERLINE  0x04
#define FT_STYLE_DEFAULT    0xFF

/* Bounding box modes */
#define FT_BBOX_EXACT           FT_GLYPH_BBOX_SUBPIXELS
#define FT_BBOX_EXACT_GRIDFIT   FT_GLYPH_BBOX_GRIDFIT
#define FT_BBOX_PIXEL           FT_GLYPH_BBOX_TRUNCATE
#define FT_BBOX_PIXEL_GRIDFIT   FT_GLYPH_BBOX_PIXELS

/* Rendering flags */
#define FT_RFLAG_NONE           (0)
#define FT_RFLAG_ANTIALIAS      (1 << 0)
#define FT_RFLAG_AUTOHINT       (1 << 1)
#define FT_RFLAG_VERTICAL       (1 << 2)
#define FT_RFLAG_HINTED         (1 << 3)
#define FT_RFLAG_DEFAULTS       (FT_RFLAG_NONE | FT_RFLAG_HINTED)


#define FT_RENDER_NEWBYTEARRAY      0x0
#define FT_RENDER_NEWSURFACE        0x1
#define FT_RENDER_EXISTINGSURFACE   0x2

/**********************************************************
 * Global module types
 **********************************************************/

typedef struct
{
    int face_index;
    FT_Open_Args open_args;
} FontId;

typedef struct
{
    PyObject_HEAD
    FontId id;

    FT_Int16 ptsize;
    FT_Byte style;
    FT_Byte vertical;
    FT_Byte antialias;

    void *_internals;
} PyFreeTypeFont;


/**********************************************************
 * Module declaration
 **********************************************************/
#define PYGAMEAPI_FREETYPE_NUMSLOTS 2

#define PyFreeTypeFont_AsFont(x) (((PyFreeTypeFont *)x)->font)

#ifndef PYGAME_FREETYPE_INTERNAL

#define PyFreeTypeFont_Check(x) ((x)->ob_type == (PyTypeObject*)PyFREETYPE_C_API[0])
#define PyFreeTypeFont_Type (*(PyTypeObject*)PyFREETYPE_C_API[0])
#define PyFont_New (*(PyObject*(*)(const char*, int))PyFREETYPE_C_API[1])

#define import_pygame_freetype() { \
	PyObject *module = PyImport_ImportModule(MODPREFIX "freetype"); \
	if (module != NULL) { \
		PyObject *dict = PyModule_GetDict(module); \
		PyObject *c_api = PyDict_GetItemString(dict, PYGAMEAPI_LOCAL_ENTRY); \
		if(PyCObject_Check(c_api)) {\
			void** localptr = (void**)PyCObject_AsVoidPtr(c_api); \
			memcpy(PyFREETYPE_C_API, localptr, sizeof(void*)*PYGAMEAPI_FREETYPE_NUMSLOTS); \
} Py_DECREF(module); } }

#endif /* PYGAME_FREETYPE_INTERNAL */

static void *PyFREETYPE_C_API[PYGAMEAPI_FREETYPE_NUMSLOTS] = {NULL};

#endif /* _PYGAME_FREETYPE_H_ */
