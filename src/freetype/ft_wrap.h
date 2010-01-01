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

/**********************************************************
 * Internal module defines
 **********************************************************/

/* Fixed point (26.6) math macros */
#define PGFT_FLOOR(x)  (   (x)        & -64 )
#define PGFT_CEIL(x)   ( ( (x) + 63 ) & -64 )
#define PGFT_ROUND(x)  ( ( (x) + 32 ) & -64 )
#define PGFT_TRUNC(x)  (   (x) >> 6 )

/* Internal configuration variables */
#define PGFT_MAX_GLYPHS         64
#define PGFT_DEFAULT_CACHE_SIZE 64
#define PGFT_MIN_CACHE_SIZE     32
#undef  PGFT_DEBUG_CACHE


/**********************************************************
 * Internal data structures
 **********************************************************/
typedef struct
{
    FT_Library library;
    FTC_Manager cache_manager;
    FTC_CMapCache cache_charmap;

    int cache_size;
    char *_error_msg;
} FreeTypeInstance;

typedef struct __fontcolor
{
    FT_Byte r;
    FT_Byte g;
    FT_Byte b;
    FT_Byte a;
} FontColor;

typedef struct __fontsurface
{
    void *buffer;

    int x_offset;
    int y_offset;

    int width;
    int height;
    int pitch;

    SDL_PixelFormat *format;

    void (* render) (int, int, struct __fontsurface *, FT_Bitmap *, FontColor *);
    void (* fill)   (int, int, int, int, struct __fontsurface *, FontColor *);

} FontSurface;

typedef struct __rendermode
{
    FT_UInt16   pt_size;
    FT_UInt16   rotation_angle;
    FT_UInt16   render_flags;
    FT_UInt16   style;
} FontRenderMode;

typedef struct  __fontglyph
{
    FT_UInt     glyph_index;
    FT_Glyph    image;    

    FT_Pos      delta;    
    FT_Vector   vvector;  
    FT_Vector   vadvance; 

    FT_Fixed    baseline;
    FT_Vector   size;
} FontGlyph;

typedef struct __fonttext
{
    FontGlyph **glyphs;
    FT_Vector *advances;
    int length;

    FT_Vector glyph_size;       /* 26.6 */
    FT_Vector text_size;        /* 26.6 */
    FT_Vector baseline_offset;  /* 26.6 */

    FT_Fixed underline_size;
    FT_Fixed underline_pos;
} FontText;

typedef struct __cachenode
{
    FontGlyph   glyph;
    struct __cachenode *next;
    FT_UInt32 hash;

} FontCacheNode;

typedef struct __fontcache
{
    FontCacheNode  **nodes;
    FontCacheNode  *free_nodes;

    FT_Byte    *depths;

#ifdef PGFT_DEBUG_CACHE
    FT_UInt32   count;
    FT_UInt32   _debug_delete_count;
    FT_UInt32   _debug_access;
    FT_UInt32   _debug_hit;
    FT_UInt32   _debug_miss;
#endif

    FT_UInt32   size_mask;
    PyFreeTypeFont  *font;
} FontCache;

#define PGFT_INTERNALS(f) ((FontInternals *)(f->_internals))
typedef struct FontInternals_
{
    FontCache  cache;
    FontText    active_text;
} FontInternals;



/**********************************************************
 * Module state
 **********************************************************/
typedef struct {
    FreeTypeInstance *freetype;
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
                const FontRenderMode *render, PyObject *text, int *w, int *h);

int         PGFT_GetMetrics(FreeTypeInstance *ft, PyFreeTypeFont *font,
                int character, const FontRenderMode *render, int bbmode, 
                void *minx, void *maxx, void *miny, void *maxy, void *advance);

int         PGFT_GetSurfaceSize(FreeTypeInstance *ft, PyFreeTypeFont *font,
                const FontRenderMode *render, FontText *text, 
                int *width, int *height);



/******************************************************************* Rendering ****/
PyObject *  PGFT_Render_PixelArray(FreeTypeInstance *ft, PyFreeTypeFont *font,
                const FontRenderMode *render, PyObject *text, int *_width, int *_height);

SDL_Surface *PGFT_Render_NewSurface(FreeTypeInstance *ft, PyFreeTypeFont *font,
                const FontRenderMode *render, PyObject *text,
                FontColor *fgcolor, FontColor *bgcolor, int *_width, int *_height);

int         PGFT_Render_ExistingSurface(FreeTypeInstance *ft, PyFreeTypeFont *font,
                const FontRenderMode *render, PyObject *text, 
                SDL_Surface *_surface, int x, int y, FontColor *fgcolor, FontColor *bgcolor,
                int *_width, int *_height);

int         PGFT_BuildRenderMode(FreeTypeInstance *ft, 
                PyFreeTypeFont *font, FontRenderMode *mode, int pt_size, 
                int style, int rotation);

FT_Fixed    PGFT_GetBoldStrength(FT_Face face);

int         _PGFT_Render_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font, 
                FontText *text, const FontRenderMode *render, FontColor *fg_color, 
                FontSurface *surface);

int PGFT_CheckStyle(FT_UInt32 style);


/******************************************************************* Render callbacks ****/

void __fill_glyph_RGB1(int x, int y, int w, int h, FontSurface *surface, FontColor *color);
void __fill_glyph_RGB2(int x, int y, int w, int h, FontSurface *surface, FontColor *color);
void __fill_glyph_RGB3(int x, int y, int w, int h, FontSurface *surface, FontColor *color);
void __fill_glyph_RGB4(int x, int y, int w, int h, FontSurface *surface, FontColor *color);

void __render_glyph_MONO1(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);
void __render_glyph_MONO2(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);
void __render_glyph_MONO3(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);
void __render_glyph_MONO4(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);

void __render_glyph_RGB1(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);
void __render_glyph_RGB2(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);
void __render_glyph_RGB3(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);
void __render_glyph_RGB4(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);

void __render_glyph_ByteArray(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, FontColor *color);

/******************************************************** Font text management ****/
FontText *  PGFT_LoadFontText(FreeTypeInstance *ft, PyFreeTypeFont *font, 
                const FontRenderMode *render, PyObject *text);

int         PGFT_LoadTextAdvances(FreeTypeInstance *ft, PyFreeTypeFont *font, 
                const FontRenderMode *render, FontText *text);

FT_UInt16 * PGFT_BuildUnicodeString(FreeTypeInstance *ft, PyObject *text);


/******************************************************** Glyph cache management ****/
void        PGFT_Cache_Init(FreeTypeInstance *ft, FontCache *cache, PyFreeTypeFont *parent);
void        PGFT_Cache_Destroy(FontCache *cache);
void        PGFT_Cache_Cleanup(FontCache *cache);
FontGlyph * PGFT_Cache_FindGlyph(FreeTypeInstance *ft, FontCache *cache, FT_UInt character, 
                const FontRenderMode *render);


/******************************************************************* Internals ****/
void        _PGFT_SetError(FreeTypeInstance *, const char *, FT_Error);
FT_Face     _PGFT_GetFace(FreeTypeInstance *, PyFreeTypeFont *);
FT_Face     _PGFT_GetFaceSized(FreeTypeInstance *, PyFreeTypeFont *, int);
void        _PGFT_BuildScaler(PyFreeTypeFont *, FTC_Scaler, int);


#endif
