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

#ifndef _PYGAME_FREETYPE_WRAP_H_
#define _PYGAME_FREETYPE_WRAP_H_

#define PYGAME_FREETYPE_INTERNAL
#include "pgfreetype.h"

#ifdef HAVE_PYGAME_SDL_VIDEO
#   include "pgsdl.h"
#endif

typedef struct
{
    FT_Library library;
    FTC_Manager cache_manager;
    FTC_SBitCache cache_sbit;
    FTC_ImageCache cache_img;
    FTC_CMapCache cache_charmap;

    char *_error_msg;
} FreeTypeInstance;


typedef struct  FontGlyph_
{
    FT_UInt    glyph_index;
    FT_Glyph   image;    

    FT_Pos     delta;    
    FT_Vector  vvector;  
    FT_Vector  vadvance; 

} FontGlyph;


typedef struct FontText_
{
    FontGlyph *glyphs;
    int length;

    FT_UInt render_mode;

    FT_UInt32 _hash;

} FontText;

typedef struct {
    FreeTypeInstance *freetype;
} _FreeTypeState;

#ifdef IS_PYTHON_3
extern struct PyModuleDef _freetypemodule;
#define FREETYPE_MOD_STATE(mod) ((_FreeTypeState*)PyModule_GetState(mod))
#define FREETYPE_STATE FREETYPE_MOD_STATE(PyState_FindModule(&_freetypemodule))
#else
extern _FreeTypeState _modstate;
#define FREETYPE_MOD_STATE(mod) (&_modstate)
#define FREETYPE_STATE FREETYPE_MOD_STATE(NULL)
#endif

#define ASSERT_GRAB_FREETYPE(ft_ptr, rvalue)                    \
    ft_ptr = FREETYPE_STATE->freetype;                          \
    if (ft_ptr == NULL)                                         \
    {                                                           \
        PyErr_SetString(PyExc_PyGameError,                      \
            "The FreeType 2 library hasn't been initialized");  \
        return (rvalue);                                        \
    }

#define GET_FONT_ID(f) (&((PyFreeTypeFont *)f)->id)

#define FT_FLOOR(X)	((X & -64) / 64)
#define FT_CEIL(X)	(((X + 63) & -64) / 64)

const char *PGFT_GetError(FreeTypeInstance *);
void    PGFT_Quit(FreeTypeInstance *);
int     PGFT_Init(FreeTypeInstance **);
int     PGFT_TryLoadFont_Filename(FreeTypeInstance *, 
            PyFreeTypeFont *, const char *, int);
void    PGFT_UnloadFont(FreeTypeInstance *, PyFreeTypeFont *);

int     PGFT_Face_GetHeight(FreeTypeInstance *ft, PyFreeTypeFont *);
int     PGFT_Face_IsFixedWidth(FreeTypeInstance *ft, PyFreeTypeFont *);
const char * PGFT_Face_GetName(FreeTypeInstance *ft, PyFreeTypeFont *);

FT_UInt16 *PGFT_BuildUnicodeString(PyObject *, int *);
PyObject  *PGFT_BuildSDLSurface(FT_Byte *buffer, int width, int height);

int     PGFT_GetTextSize(FreeTypeInstance *, PyFreeTypeFont *,
            const FT_UInt16 *, int, int *, int *, int *);

int     PGFT_GetMetrics(FreeTypeInstance *ft, PyFreeTypeFont *font,
            int character, int font_size, int bbmode, 
            void *minx, void *maxx, void *miny, void *maxy, void *advance);

PyObject *PGFT_Render_PixelArray(FreeTypeInstance *ft, PyFreeTypeFont *font,
        const FT_UInt16 *text, int font_size, int *_width, int *_height);
PyObject *PGFT_Render_NewSurface(FreeTypeInstance *ft, PyFreeTypeFont *font,
        const FT_UInt16 *text, int font_size, int *_width, int *_height,
        PyColor *fg_color, PyColor *bg_color);
int PGFT_Render_ExistingSurface(FreeTypeInstance *ft, PyFreeTypeFont *font,
    const FT_UInt16 *text, int font_size, PySDLSurface *_surface,
    int *_width, int *_height, int x, int y,
    PyColor *py_fgcolor);

FontText *PGFT_BuildFontText(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    PyObject *text, int pt_size);


#endif
