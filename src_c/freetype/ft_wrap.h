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
#include "../_pygame.h"
#include "../freetype.h"

/**********************************************************
 * Internal module defines
 **********************************************************/

/* Fixed point (26.6) math macros */
#define FX6_ONE 64L
#define FX16_ONE 65536L
#define FX6_MAX (0x7FFFFFFFL)
#define FX6_MIN (~FX6_MAX)

#define FX6_FLOOR(x) ((x) & -64L)
#define FX6_CEIL(x) (((x) + 63L) & -64L)
#define FX6_ROUND(x) (((x) + 32L) & -64L)
#define FX6_TRUNC(x) ((x) >> 6)
#define FX16_CEIL_TO_FX6(x) (((x) + 1023L) >> 10)
#define FX16_ROUND_TO_INT(x) (((x) + 32768L) >> 16)
#define INT_TO_FX6(i) ((FT_Fixed)((i) << 6))
#define INT_TO_FX16(i) ((FT_Fixed)((i) << 16))
#define FX16_TO_DBL(x) ((x)*1.52587890625e-5 /* 2.0^-16 */)
#define DBL_TO_FX16(d) ((FT_Fixed)((d)*65536.0))
#define FX6_TO_DBL(x) ((x)*1.5625e-2 /* 2.0^-6 */)
#define DBL_TO_FX6(d) ((FT_Fixed)((d)*64.0))

/* Internal configuration variables */
#define PGFT_DEFAULT_CACHE_SIZE 64
#define PGFT_MIN_CACHE_SIZE 32
#if defined(PGFT_DEBUG_CACHE)
#undef PGFT_DEBUG_CACHE
#endif
#define PGFT_DEFAULT_RESOLUTION 72 /* dots per inch */

#define PGFT_DBL_DEFAULT_STRENGTH (1.0 / 36.0)

/* Rendering styles unsupported for bitmap fonts */
#define FT_STYLES_SCALABLE_ONLY (FT_STYLE_STRONG | FT_STYLE_OBLIQUE)

/**********************************************************
 * Internal basic types
 **********************************************************/

typedef FT_UInt32 PGFT_char;
typedef FT_UInt GlyphIndex_t;

/**********************************************************
 * Internal data structures
 **********************************************************/

/* FreeTypeInstance: the global freetype 2 library state.
 *
 * Instances of this struct are created by _PGFT_Init, and
 * destroyed by _PGFT_Quit. The instances are reference counted.
 * When adding a local reference, be sure to increment ref_count;
 * _PTFT_Init returns an instance with ref_count equal one.
 * When removing a reference, call _PGFT_Quit, which will decrement
 * the reference count and free the resource if the count reaches
 * zero.
 */
typedef struct freetypeinstance_ {
    Py_ssize_t ref_count;

    /* Internal */
    FT_Library library;
    FTC_Manager cache_manager;
    FTC_CMapCache cache_charmap;

    int cache_size;
    char _error_msg[1024];
} FreeTypeInstance;

typedef struct fontcolor_ {
    FT_Byte r;
    FT_Byte g;
    FT_Byte b;
    FT_Byte a;
} FontColor;

typedef struct rendermode_ {
    Scale_t face_size;
    FT_Angle rotation_angle;
    FT_UInt16 render_flags;
    FT_UInt16 style;

    /* All these are Fixed 16.16 */
    FT_Fixed strength;
    FT_Fixed underline_adjustment;
    FT_Matrix transform;
} FontRenderMode;

#if defined(Py_DEBUG) && !defined(PGFT_DEBUG_CACHE)
#define PGFT_DEBUG_CACHE 1
#endif

struct cachenode_;

typedef struct fontcache_ {
    struct cachenode_ **nodes;
    struct cachenode_ *free_nodes;

    FT_Byte *depths;

#ifdef PGFT_DEBUG_CACHE
    FT_UInt32 _debug_count;
    FT_UInt32 _debug_delete_count;
    FT_UInt32 _debug_access;
    FT_UInt32 _debug_hit;
    FT_UInt32 _debug_miss;
#endif

    FT_UInt32 size_mask;
} FontCache;

typedef struct fontmetrics_ {
    /* All these are 26.6 precision */
    FT_Pos bearing_x;
    FT_Pos bearing_y;
    FT_Vector bearing_rotated;
    FT_Vector advance_rotated;
} FontMetrics;

typedef struct fontglyph_ {
    FT_BitmapGlyph image;

    FT_Pos width;  /* 26.6 */
    FT_Pos height; /* 26.6 */
    FontMetrics h_metrics;
    FontMetrics v_metrics;
} FontGlyph;

typedef struct glyphslot_ {
    GlyphIndex_t id;
    FontGlyph *glyph;
    FT_Vector posn;
    FT_Vector kerning;
} GlyphSlot;

typedef struct layout_ {
    FontRenderMode mode;

    int length;

    int top;  /* In pixels */
    int left; /* In pixels */

    FT_Pos min_x;
    FT_Pos max_x;
    FT_Pos min_y;
    FT_Pos max_y;
    FT_Vector offset;
    FT_Vector advance;
    FT_Pos ascender;
    FT_Pos descender;
    FT_Pos height;
    FT_Pos max_advance;
    FT_Fixed underline_size;
    FT_Pos underline_pos;

    int buffer_size;
    GlyphSlot *glyphs;
} Layout;

struct fontsurface_;

typedef void (*FontRenderPtr)(int, int, struct fontsurface_ *,
                              const FT_Bitmap *, const FontColor *);
typedef void (*FontFillPtr)(FT_Fixed, FT_Fixed, FT_Fixed, FT_Fixed,
                            struct fontsurface_ *, const FontColor *);

typedef struct fontsurface_ {
    void *buffer;

    unsigned width;
    unsigned height;
    int item_stride;
    int pitch;

    SDL_PixelFormat *format;

    FontRenderPtr render_gray;
    FontRenderPtr render_mono;
    FontFillPtr fill;

} FontSurface;

typedef struct fontinternals_ {
    Layout active_text;
    FontCache glyph_cache;
} FontInternals;

typedef struct PGFT_String_ {
    Py_ssize_t length;
    PGFT_char data[1];
} PGFT_String;

#if defined(PGFT_DEBUG_CACHE)
#define PGFT_FONT_CACHE(f) ((f)->_internals->glyph_cache)
#endif

/**********************************************************
 * Module state
 **********************************************************/
typedef struct {
    FreeTypeInstance *freetype;
    int cache_size;
    FT_UInt resolution;
} _FreeTypeState;

#if !defined(PYPY_VERSION)
extern struct PyModuleDef _freetypemodule;
#define FREETYPE_MOD_STATE(mod) ((_FreeTypeState *)PyModule_GetState(mod))
#define FREETYPE_STATE FREETYPE_MOD_STATE(PyState_FindModule(&_freetypemodule))
#else /* defined(PYPY_VERSION) */
extern _FreeTypeState _modstate;
#define FREETYPE_MOD_STATE(mod) (&_modstate)
#define FREETYPE_STATE FREETYPE_MOD_STATE(0)
#endif /* defined(PYPY_VERSION) */

#define ASSERT_GRAB_FREETYPE(ft_ptr, rvalue)                               \
    ft_ptr = FREETYPE_STATE->freetype;                                     \
    if (!ft_ptr) {                                                         \
        PyErr_SetString(PyExc_RuntimeError,                                \
                        "The FreeType 2 library hasn't been initialized"); \
        return (rvalue);                                                   \
    }

/**********************************************************
 * Internal API
 **********************************************************/

/**************************************** General functions ******************/
const char *
_PGFT_GetError(FreeTypeInstance *);
void
_PGFT_Quit(FreeTypeInstance *);
int
_PGFT_Init(FreeTypeInstance **, int);
long
_PGFT_Font_GetAscender(FreeTypeInstance *, pgFontObject *);
long
_PGFT_Font_GetAscenderSized(FreeTypeInstance *, pgFontObject *, Scale_t);
long
_PGFT_Font_GetDescender(FreeTypeInstance *, pgFontObject *);
long
_PGFT_Font_GetDescenderSized(FreeTypeInstance *, pgFontObject *, Scale_t);
long
_PGFT_Font_GetHeight(FreeTypeInstance *, pgFontObject *);
long
_PGFT_Font_GetHeightSized(FreeTypeInstance *, pgFontObject *, Scale_t);
long
_PGFT_Font_GetGlyphHeightSized(FreeTypeInstance *, pgFontObject *, Scale_t);
int
_PGFT_Font_IsFixedWidth(FreeTypeInstance *, pgFontObject *);
int
_PGFT_Font_NumFixedSizes(FreeTypeInstance *, pgFontObject *);
int
_PGFT_Font_GetAvailableSize(FreeTypeInstance *, pgFontObject *, long, long *,
                            long *, long *, double *, double *);
const char *
_PGFT_Font_GetName(FreeTypeInstance *, pgFontObject *);
int
_PGFT_TryLoadFont_Filename(FreeTypeInstance *, pgFontObject *, const char *,
                           long);
int
_PGFT_TryLoadFont_RWops(FreeTypeInstance *, pgFontObject *, SDL_RWops *, long);
SDL_RWops *
_PGFT_GetRWops(pgFontObject *fontobj);
void
_PGFT_UnloadFont(FreeTypeInstance *, pgFontObject *);

/**************************************** Metrics management *****************/
int
_PGFT_GetTextRect(FreeTypeInstance *, pgFontObject *, const FontRenderMode *,
                  PGFT_String *, SDL_Rect *);
int
_PGFT_GetMetrics(FreeTypeInstance *, pgFontObject *, PGFT_char,
                 const FontRenderMode *, FT_UInt *, long *, long *, long *,
                 long *, double *, double *);
void
_PGFT_GetRenderMetrics(const FontRenderMode *, Layout *, unsigned *,
                       unsigned *, FT_Vector *, FT_Pos *, FT_Fixed *);

/**************************************** Rendering **************************/
PyObject *
_PGFT_Render_PixelArray(FreeTypeInstance *, pgFontObject *,
                        const FontRenderMode *, PGFT_String *, int, int *,
                        int *);
SDL_Surface *
_PGFT_Render_NewSurface(FreeTypeInstance *, pgFontObject *,
                        const FontRenderMode *, PGFT_String *, FontColor *,
                        FontColor *, SDL_Rect *);
int
_PGFT_Render_ExistingSurface(FreeTypeInstance *, pgFontObject *,
                             const FontRenderMode *, PGFT_String *,
                             SDL_Surface *, int, int, FontColor *, FontColor *,
                             SDL_Rect *);
int
_PGFT_Render_Array(FreeTypeInstance *, pgFontObject *, const FontRenderMode *,
                   PyObject *, PGFT_String *, int, int, int, SDL_Rect *);
int
_PGFT_BuildRenderMode(FreeTypeInstance *, pgFontObject *, FontRenderMode *,
                      Scale_t, int, Angle_t);
int _PGFT_CheckStyle(FT_UInt32);

/**************************************** Render callbacks *******************/
void
__fill_glyph_RGB1(FT_Fixed, FT_Fixed, FT_Fixed, FT_Fixed, FontSurface *,
                  const FontColor *);
void
__fill_glyph_RGB2(FT_Fixed, FT_Fixed, FT_Fixed, FT_Fixed, FontSurface *,
                  const FontColor *);
void
__fill_glyph_RGB3(FT_Fixed, FT_Fixed, FT_Fixed, FT_Fixed, FontSurface *,
                  const FontColor *);
void
__fill_glyph_RGB4(FT_Fixed, FT_Fixed, FT_Fixed, FT_Fixed, FontSurface *,
                  const FontColor *);

void
__fill_glyph_GRAY1(FT_Fixed, FT_Fixed, FT_Fixed, FT_Fixed, FontSurface *,
                   const FontColor *);

void
__fill_glyph_INT(FT_Fixed, FT_Fixed, FT_Fixed, FT_Fixed, FontSurface *,
                 const FontColor *);

void
__render_glyph_MONO1(int, int, FontSurface *, const FT_Bitmap *,
                     const FontColor *);
void
__render_glyph_MONO2(int, int, FontSurface *, const FT_Bitmap *,
                     const FontColor *);
void
__render_glyph_MONO3(int, int, FontSurface *, const FT_Bitmap *,
                     const FontColor *);
void
__render_glyph_MONO4(int, int, FontSurface *, const FT_Bitmap *,
                     const FontColor *);

void
__render_glyph_RGB1(int, int, FontSurface *, const FT_Bitmap *,
                    const FontColor *);
void
__render_glyph_RGB2(int, int, FontSurface *, const FT_Bitmap *,
                    const FontColor *);
void
__render_glyph_RGB3(int, int, FontSurface *, const FT_Bitmap *,
                    const FontColor *);
void
__render_glyph_RGB4(int, int, FontSurface *, const FT_Bitmap *,
                    const FontColor *);

void
__render_glyph_GRAY1(int, int, FontSurface *, const FT_Bitmap *,
                     const FontColor *);
void
__render_glyph_MONO_as_GRAY1(int, int, FontSurface *, const FT_Bitmap *,
                             const FontColor *);
void
__render_glyph_GRAY_as_MONO1(int, int, FontSurface *, const FT_Bitmap *,
                             const FontColor *);

void
__render_glyph_INT(int, int, FontSurface *, const FT_Bitmap *,
                   const FontColor *);
void
__render_glyph_MONO_as_INT(int, int, FontSurface *, const FT_Bitmap *,
                           const FontColor *);

/**************************************** Layout management ******************/
int
_PGFT_LayoutInit(FreeTypeInstance *, pgFontObject *);
void
_PGFT_LayoutFree(pgFontObject *);
Layout *
_PGFT_LoadLayout(FreeTypeInstance *, pgFontObject *, const FontRenderMode *,
                 PGFT_String *);
int
_PGFT_LoadGlyph(FontGlyph *, GlyphIndex_t, const FontRenderMode *, void *);

/**************************************** Glyph cache management *************/
int
_PGFT_Cache_Init(FreeTypeInstance *, FontCache *);
void
_PGFT_Cache_Destroy(FontCache *);
void
_PGFT_Cache_Cleanup(FontCache *);
FontGlyph *
_PGFT_Cache_FindGlyph(FT_UInt32, const FontRenderMode *, FontCache *, void *);

/**************************************** Unicode ****************************/
PGFT_String *
_PGFT_EncodePyString(PyObject *, int);
#define PGFT_String_GET_DATA(s) ((s)->data)
#define PGFT_String_GET_LENGTH(s) ((s)->length)
#define _PGFT_FreeString _PGFT_free

/**************************************** Internals **************************/
void
_PGFT_SetError(FreeTypeInstance *, const char *, FT_Error);
FT_Face
_PGFT_GetFont(FreeTypeInstance *, pgFontObject *);
FT_Face
_PGFT_GetFontSized(FreeTypeInstance *, pgFontObject *, Scale_t);
void
_PGFT_BuildScaler(pgFontObject *, FTC_Scaler, Scale_t);
#define _PGFT_malloc PyMem_Malloc
#define _PGFT_free PyMem_Free

#endif
