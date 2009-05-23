/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners
  Copyright (C) 2008 Marcus von Appen
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

#ifndef _PYGAME_FREETYPE_WRAP_H_
#define _PYGAME_FREETYPE_WRAP_H_

#define PYGAME_FREETYPE_INTERNAL
#include "pgfreetype.h"

typedef struct
{
    FT_Library library;
    FTC_Manager cache_manager;
    FTC_SBitCache cache_bitmap;
    FTC_CMapCache cache_charmap;

    char *_error_msg;
} FreeTypeInstance;

FreeTypeInstance *_get_freetype(void);

#define ASSERT_GRAB_FREETYPE(ft_ptr, rvalue)                    \
    ft_ptr = _get_freetype();                                   \
    if (ft_ptr == NULL)                                         \
    {                                                           \
        PyErr_SetString(PyExc_PyGameError,                      \
            "The FreeType 2 library hasn't been initialized");  \
        return (rvalue);                                        \
    }

#define GET_FONT_ID(f) (&((PyFreeTypeFont *)f)->id)

const char *PGFT_GetError(FreeTypeInstance *);
void    PGFT_Quit(FreeTypeInstance *);
int     PGFT_Init(FreeTypeInstance **);
int     PGFT_TryLoadFont(FreeTypeInstance *, PyFreeTypeFont *);
void    PGFT_UnloadFont(FreeTypeInstance *, PyFreeTypeFont *);

int     PGFT_Face_GetHeight(PyFreeTypeFont *);
int     PGFT_Face_IsFixedWidth(PyFreeTypeFont *);
const char * PGFT_Face_GetName(PyFreeTypeFont *);
const char * PGFT_Face_GetFormat(PyFreeTypeFont *);


#endif
