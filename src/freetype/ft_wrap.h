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
#include "../freetype.h"

/**********************************************************
 * Internal module defines
 **********************************************************/

/* Fixed point (26.6) math macros */
#define PGFT_FLOOR(x)  (   (x)        & -64 )
#define PGFT_CEIL(x)   ( ( (x) + 63 ) & -64 )
#define PGFT_ROUND(x)  ( ( (x) + 32 ) & -64 )
#define PGFT_TRUNC(x)  (   (x) >> 6 )
#define PGFT_CEIL16_TO_6(x)  ( ( (x) + 1023 ) >> 10 )
#define PGFT_INT_TO_6(x) ( (x) << 6 )
#define PGFT_INT_TO_16(x) ( (x) << 16 )

#define PGFT_MIN_6 ((FT_Pos)0x80000000)
#define PGFT_MAX_6 ((FT_Pos)0x7FFFFFFF)

/* Internal configuration variables */
#define PGFT_DEFAULT_CACHE_SIZE 64
#define PGFT_MIN_CACHE_SIZE     32
#undef  PGFT_DEBUG_CACHE
#define PGFT_DEFAULT_RESOLUTION 72 /* dots per inch */


/**********************************************************
 * Internal basic types
 **********************************************************/

typedef FT_UInt32 PGFT_char;


/**********************************************************
 * Internal data structures
 **********************************************************/
typedef struct
{
    FT_Library library;
    FTC_Manager cache_manager;
    FTC_CMapCache cache_charmap;

    int cache_size;
    char _error_msg[1024];
} FreeTypeInstance;

typedef struct __fontcolor
{
    FT_Byte r;
    FT_Byte g;
    FT_Byte b;
    FT_Byte a;
} FontColor;

typedef struct __rendermode
{
    FT_UInt16   pt_size;
    FT_Angle    rotation_angle;
    FT_UInt16   render_flags;
    FT_UInt16   style;
} FontRenderMode;

#if defined(Py_DEBUG) && !defined(PGFT_DEBUG_CACHE)
#define PGFT_DEBUG_CACHE 1
#endif

struct __cachenode;

typedef struct __fontcache
{
    struct __cachenode **nodes;
    struct __cachenode *free_nodes;

    FT_Byte    *depths;

#ifdef PGFT_DEBUG_CACHE
    FT_UInt32   _debug_count;
    FT_UInt32   _debug_delete_count;
    FT_UInt32   _debug_access;
    FT_UInt32   _debug_hit;
    FT_UInt32   _debug_miss;
#endif

    FT_UInt32   size_mask;
} FontCache;

typedef struct __fontmetrics
{
    /* All these are 26.6 precision */
    FT_Pos    bearing_x;
    FT_Pos    bearing_y;
    FT_Vector bearing_rotated;
    FT_Vector advance_rotated;
} FontMetrics;

typedef struct __fontglyph
{
    FT_UInt     glyph_index;
    FT_BitmapGlyph image;

    FT_Pos      width;         /* 26.6 */
    FT_Pos      height;        /* 26.6 */
    FT_Pos      bold_strength; /* 26.6 */
    FontMetrics h_metrics;
    FontMetrics v_metrics;
} FontGlyph;

typedef struct __fonttext
{
    int      length;
    int      width;     /* In pixels */
    int      height;    /* In pixels */
    int      top;       /* In pixels */
    int      left;      /* In pixels */

    FT_Vector offset;
    FT_Vector advance;
    FT_Pos   underline_size;
    FT_Pos   underline_pos;

    int       buffer_size;
    FontGlyph **glyphs;
    FT_Vector *posns;

    FontCache  glyph_cache;
} FontText;

struct __fontsurface;

typedef void (* FontRenderPtr)(int, int, struct __fontsurface *,
                   FT_Bitmap *, FontColor *);
typedef void (* FontFillPtr)(int, int, int, int, struct __fontsurface *,
                 FontColor *);

typedef struct __fontsurface
{
    void *buffer;

    FT_Vector offset;

    int width;
    int height;
    int pitch;

    SDL_PixelFormat *format;

    FontRenderPtr render_gray;
    FontRenderPtr render_mono;
    FontFillPtr fill;

} FontSurface;

#define PGFT_INTERNALS(f) ((FontInternals *)((f)->_internals))
typedef struct FontInternals_
{
    FontText    active_text;
} FontInternals;

typedef struct PGFT_String_
{
    Py_ssize_t length;
    PGFT_char data[1];
} PGFT_String;

#if defined(PGFT_DEBUG_CACHE)
#define PGFT_FONT_CACHE(f) (PGFT_INTERNALS(f)->active_text.glyph_cache)
#endif

/**********************************************************
 * Module state
 **********************************************************/
typedef struct {
    FreeTypeInstance *freetype;
    int cache_size;
    FT_UInt resolution;
} _FreeTypeState;

#ifdef IS_PYTHON_3
    extern struct PyModuleDef _freetypemodule;
#   define FREETYPE_MOD_STATE(mod) ((_FreeTypeState*)PyModule_GetState(mod))
#   define FREETYPE_STATE FREETYPE_MOD_STATE(PyState_FindModule(&_freetypemodule))
#else
    extern _FreeTypeState _modstate;
#   define FREETYPE_MOD_STATE(mod) (&_modstate)
#   define FREETYPE_STATE FREETYPE_MOD_STATE(NULL)
#endif

#define ASSERT_GRAB_FREETYPE(ft_ptr, rvalue)                    \
    ft_ptr = FREETYPE_STATE->freetype;                          \
    if (ft_ptr == NULL)                                         \
    {                                                           \
        PyErr_SetString(PyExc_RuntimeError,                     \
            "The FreeType 2 library hasn't been initialized");  \
        return (rvalue);                                        \
    }




/**********************************************************
 * Internal API
 **********************************************************/

/********************************************************* General functions ****/
const char *PGFT_GetError(FreeTypeInstance *);
void        PGFT_Quit(FreeTypeInstance *);
int         PGFT_Init(FreeTypeInstance **, int cache_size);

int         PGFT_Face_GetHeight(FreeTypeInstance *ft, PyFreeTypeFont *);
int         PGFT_Face_IsFixedWidth(FreeTypeInstance *ft, PyFreeTypeFont *);
const char *PGFT_Face_GetName(FreeTypeInstance *ft, PyFreeTypeFont *);

int         PGFT_TryLoadFont_Filename(FreeTypeInstance *, 
                PyFreeTypeFont *, const char *, int);

#ifdef HAVE_PYGAME_SDL_RWOPS
int         PGFT_TryLoadFont_RWops(FreeTypeInstance *, 
                PyFreeTypeFont *, SDL_RWops *, int);
#endif

void        PGFT_UnloadFont(FreeTypeInstance *, PyFreeTypeFont *);



/********************************************************* Metrics management ****/
int         PGFT_GetTextSize(FreeTypeInstance *ft, PyFreeTypeFont *font,
                const FontRenderMode *render, PGFT_String *text, int *w, int *h);

int         PGFT_GetMetrics(FreeTypeInstance *ft, PyFreeTypeFont *font,
                PGFT_char character, const FontRenderMode *render,
                long *minx, long *maxx, long *miny, long *maxy,
                double *advance_x, double *advance_y);

int         PGFT_GetSurfaceSize(FreeTypeInstance *ft, PyFreeTypeFont *font,
                const FontRenderMode *render, FontText *text, 
                int *width, int *height);

int         PGFT_GetTopLeft(FontText *text, int *top, int *left);


/******************************************************************* Rendering ****/
PyObject *  PGFT_Render_PixelArray(FreeTypeInstance *ft, PyFreeTypeFont *font,
                const FontRenderMode *render, PGFT_String *text, int *_width, int *_height);

SDL_Surface *PGFT_Render_NewSurface(FreeTypeInstance *ft, PyFreeTypeFont *font,
                const FontRenderMode *render, PGFT_String *text,
                FontColor *fgcolor, FontColor *bgcolor, int *_width, int *_height);

int         PGFT_Render_ExistingSurface(FreeTypeInstance *ft, PyFreeTypeFont *font,
                const FontRenderMode *render, PGFT_String *text, 
                SDL_Surface *_surface, int x, int y,
                FontColor *fgcolor, FontColor *bgcolor,
        int *_width, int *_height, FontMetrics *metrics);

int         PGFT_BuildRenderMode(FreeTypeInstance *ft, 
                PyFreeTypeFont *font, FontRenderMode *mode, int pt_size, 
                int style, int rotation);

int PGFT_CheckStyle(FT_UInt32 style);


/******************************************************************* Render callbacks ****/

void __fill_glyph_RGB1(int x, int y, int w, int h, FontSurface *surface, FontColor *color);
void __fill_glyph_RGB2(int x, int y, int w, int h, FontSurface *surface, FontColor *color);
void __fill_glyph_RGB3(int x, int y, int w, int h, FontSurface *surface, FontColor *color);
void __fill_glyph_RGB4(int x, int y, int w, int h, FontSurface *surface, FontColor *color);

void __fill_glyph_GRAY1(int x, int y, int w, int h, FontSurface *surface, FontColor *color);

void __render_glyph_MONO1(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);
void __render_glyph_MONO2(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);
void __render_glyph_MONO3(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);
void __render_glyph_MONO4(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);

void __render_glyph_RGB1(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);
void __render_glyph_RGB2(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);
void __render_glyph_RGB3(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);
void __render_glyph_RGB4(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);

void __render_glyph_GRAY1(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);
void __render_glyph_MONO_as_GRAY1(int x, int y, FontSurface *surface,
                  FT_Bitmap *bitmap, FontColor *fg_color);
void __render_glyph_GRAY_as_MONO1(int x, int y, FontSurface *surface,
                  FT_Bitmap *bitmap, FontColor *fg_color);

/******************************************************** Font text management ****/
int PGFT_FontTextInit(FreeTypeInstance *ft, PyFreeTypeFont *font);
void PGFT_FontTextFree(PyFreeTypeFont *ftext);
FontText *PGFT_LoadFontText(FreeTypeInstance *ft, PyFreeTypeFont *font, 
                            const FontRenderMode *render, PGFT_String *text);
int PGFT_LoadGlyph(FontGlyph *glyph, PGFT_char character, const FontRenderMode *render,
                   void *internal);

/******************************************************** Glyph cache management ****/
int         PGFT_Cache_Init(FreeTypeInstance *ft, FontCache *cache);
void        PGFT_Cache_Destroy(FontCache *cache);
void        PGFT_Cache_Cleanup(FontCache *cache);
FontGlyph * PGFT_Cache_FindGlyph(FT_UInt32 character, const FontRenderMode *render,
                                 FontCache *cache, void *internal);

/******************************************************************* Unicode ******/
PGFT_String * PGFT_EncodePyString(PyObject *obj, int ucs4);
#define       PGFT_String_GET_DATA(s) ((s)->data)
#define       PGFT_String_GET_LENGTH(s) ((s)->length)
#define       PGFT_FreeString _PGFT_free

/******************************************************************* Internals ****/
void        _PGFT_SetError(FreeTypeInstance *, const char *, FT_Error);
FT_Face     _PGFT_GetFace(FreeTypeInstance *, PyFreeTypeFont *);
FT_Face     _PGFT_GetFaceSized(FreeTypeInstance *, PyFreeTypeFont *, int);
void        _PGFT_BuildScaler(PyFreeTypeFont *, FTC_Scaler, int);
#define     _PGFT_malloc PyMem_Malloc
#define     _PGFT_free   PyMem_Free


#endif
