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

#define FP_1616_FLOAT(i)    ((float)((int)(i) / 65536.0f))
#define FP_248_FLOAT(i)     ((float)((int)(i) / 256.0f))
#define FP_266_FLOAT(i)     ((float)((int)(i) / 64.0f))

#define PGFT_FLOOR(x)  (   (x)        & -64 )
#define PGFT_CEIL(x)   ( ( (x) + 63 ) & -64 )
#define PGFT_ROUND(x)  ( ( (x) + 32 ) & -64 )
#define PGFT_TRUNC(x)  (   (x) >> 6 )


#define PGFT_CHECK_PTSIZE()                             \
    if (ptsize == -1)                                   \
    {                                                   \
        if (font->default_ptsize == -1)                 \
        {                                               \
            PyErr_SetString(PyExc_ValueError,           \
                    "Missing font size argument "       \
                    "and no default size specified");   \
            return NULL;                                \
        }                                               \
                                                        \
        ptsize = font->default_ptsize;                  \
    }

#define PGFT_CHECK_BOOL(_pyobj, _var)               \
    if (_pyobj)                                     \
    {                                               \
        if (!PyBool_Check(_pyobj))                  \
        {                                           \
            PyErr_SetString(PyExc_TypeError,        \
                #_var " must be a boolean value");  \
            return NULL;                            \
        }                                           \
                                                    \
        _var = PyObject_IsTrue(_pyobj);             \
    }

#define UNICODE_BOM_NATIVE	0xFEFF
#define UNICODE_BOM_SWAPPED	0xFFFE

#define FONT_RENDER_HORIZONTAL  0
#define FONT_RENDER_VERTICAL    1

#define FONT_RENDER_ALIASED     0
#define FONT_RENDER_ANTIALIAS   1

#define MAX_GLYPHS      64

typedef struct
{
    FT_Library library;
    FTC_Manager cache_manager;
    FTC_SBitCache cache_sbit;
    FTC_ImageCache cache_img;
    FTC_CMapCache cache_charmap;

    char *_error_msg;
} FreeTypeInstance;

typedef struct __fontsurface
{
    void *buffer;
    void *buffer_cap;

    int x_offset;
    int y_offset;

    int width;
    int height;
    int pitch;

    SDL_PixelFormat *format;
    void (* render)(int, int, struct __fontsurface *, FT_Bitmap *, PyColor *);

} FontSurface;

typedef struct __rendermode
{
    int         kerning_degree;
    FT_Matrix*  matrix;

    FT_Matrix   _rotation_matrix;
    FT_Fixed    _rotation_angle;

    FT_Byte     hinted;
    FT_Byte     vertical;
    FT_Byte     antialias;
    FT_Byte     kerning_mode;
    FT_Byte     autohint;
} FontRenderMode;

typedef struct  FontGlyph_
{
    FT_UInt     glyph_index;
    FT_Glyph    image;    

    FT_Pos      delta;    
    FT_Vector   vvector;  
    FT_Vector   vadvance; 

} FontGlyph;

typedef struct FontText_
{
    FontGlyph *glyphs;
    int length;
    FT_UInt32 _hash;

    FT_Vector glyph_size;       /* 26.6 */
    FT_Vector text_size;        /* integer */
    FT_Vector baseline_offset;  /* 26.6 */

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

/********************************************************* General functions ****/
const char *PGFT_GetError(FreeTypeInstance *);
void        PGFT_Quit(FreeTypeInstance *);
int         PGFT_Init(FreeTypeInstance **);
int         PGFT_TryLoadFont_Filename(FreeTypeInstance *, 
                PyFreeTypeFont *, const char *, int);
void        PGFT_UnloadFont(FreeTypeInstance *, PyFreeTypeFont *);

int         PGFT_Face_GetHeight(FreeTypeInstance *ft, PyFreeTypeFont *);
int         PGFT_Face_IsFixedWidth(FreeTypeInstance *ft, PyFreeTypeFont *);
const char *PGFT_Face_GetName(FreeTypeInstance *ft, PyFreeTypeFont *);



/********************************************************* Metrics management ****/
int         PGFT_GetTextSize(FreeTypeInstance *ft, PyFreeTypeFont *font, int pt_size,
                FontRenderMode *render, PyObject *text, int *w, int *h);

int         PGFT_GetMetrics(FreeTypeInstance *ft, PyFreeTypeFont *font,
                int character, int font_size, int bbmode, 
                void *minx, void *maxx, void *miny, void *maxy, void *advance);

int         _PGFT_GetTextSize_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font, 
                int pt_size, FontRenderMode *render, FontText *text);

void        _PGFT_GetMetrics_INTERNAL(FT_Glyph, FT_UInt, int *, int *, int *, int *, int *);



/******************************************************************* Rendering ****/
PyObject *  PGFT_Render_PixelArray(FreeTypeInstance *ft, PyFreeTypeFont *font,
                int ptsize, PyObject *text, int *_width, int *_height);

PyObject *  PGFT_Render_NewSurface(FreeTypeInstance *ft, PyFreeTypeFont *font,
                int ptsize, FontRenderMode *render, PyObject *text,
                PyColor *fgcolor, PyColor *bgcolor, int *_width, int *_height);

int         PGFT_Render_ExistingSurface(FreeTypeInstance *ft, PyFreeTypeFont *font,
                int font_size, FontRenderMode *render, PyObject *text, 
                PySDLSurface *_surface, int x, int y, PyColor *fgcolor, PyColor *bgcolor,
                int *_width, int *_height);

void        PGFT_BuildRenderMode(FontRenderMode *mode, int vertical, 
                int antialias, int rotation);

int         _PGFT_Render_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font, 
                FontText *text, int font_size, PyColor *fg_color, 
                FontSurface *surface, FontRenderMode *render);



/******************************************************** Font text management ****/
FontText *  PGFT_LoadFontText(FreeTypeInstance *ft, PyFreeTypeFont *font, int pt_size, 
                FontRenderMode *render, PyObject *text);

int         PGFT_GetTextAdvances(FreeTypeInstance *ft, PyFreeTypeFont *font, int pt_size, 
                FontRenderMode *render, FontText *text, FT_Vector *advances);

FT_UInt16 * PGFT_BuildUnicodeString(PyObject *, int *);



/******************************************************************* Internals ****/
void        _PGFT_SetError(FreeTypeInstance *, const char *, FT_Error);
FT_Face     _PGFT_GetFace(FreeTypeInstance *, PyFreeTypeFont *);
FT_Face     _PGFT_GetFaceSized(FreeTypeInstance *, PyFreeTypeFont *, int);
void        _PGFT_BuildScaler(PyFreeTypeFont *, FTC_Scaler, int);
int         _PGFT_LoadGlyph(FreeTypeInstance *, PyFreeTypeFont *, int,
                FTC_Scaler, int, FT_Glyph *, FT_UInt32 *);


#endif
