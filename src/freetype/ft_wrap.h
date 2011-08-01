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
#define FX6_ONE 64
#define FX16_ONE 65536
#define FX6_MIN ((FT_Pos)0x80000000)
#define FX6_MAX ((FT_Pos)0x7FFFFFFF)

#define FX6_FLOOR(x) ((x) & -64)
#define FX6_CEIL(x) (((x) + 63) & -64)
#define FX6_ROUND(x) (((x) + 32) & -64)
#define FX6_TRUNC(x)  ((x) >> 6)
#define FX16_CEIL_TO_FX6(x) (((x) + 1023) >> 10)
#define INT_TO_FX6(i) ((FT_Fixed)((i) << 6))
#define INT_TO_FX16(i) ((FT_Fixed)((i) << 16))
#define FX16_TO_DBL(x) ((x) * 1.5259e-5 /* 65536.0^-1 */)
#define DBL_TO_FX16(d) ((FT_Fixed)((d) * 65536.0))

/* Internal configuration variables */
#define PGFT_DEFAULT_CACHE_SIZE 64
#define PGFT_MIN_CACHE_SIZE 32
#if defined(PGFT_DEBUG_CACHE)
#undef  PGFT_DEBUG_CACHE
#endif
#define PGFT_DEFAULT_RESOLUTION 72 /* dots per inch */

#define PGFT_DBL_DEFAULT_STRENGTH (1.0 / 36.0)

/**********************************************************
 * Internal basic types
 **********************************************************/

typedef FT_UInt32 PGFT_char;


/**********************************************************
 * Internal data structures
 **********************************************************/
typedef struct {
    FT_Library library;
    FTC_Manager cache_manager;
    FTC_CMapCache cache_charmap;

    int cache_size;
    char _error_msg[1024];
} FreeTypeInstance;

typedef struct facecolor_ {
    FT_Byte r;
    FT_Byte g;
    FT_Byte b;
    FT_Byte a;
} FaceColor;

typedef struct rendermode_ {
    FT_UInt16 pt_size;
    FT_Angle rotation_angle;
    FT_UInt16 render_flags;
    FT_UInt16 style;

    /* All these are Fixed 16.16 */
    FT_Fixed strength;
    FT_Fixed underline_adjustment;
    FT_Matrix transform;
} FaceRenderMode;

#if defined(Py_DEBUG) && !defined(PGFT_DEBUG_CACHE)
#define PGFT_DEBUG_CACHE 1
#endif

struct cachenode_;

typedef struct facecache_ {
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
} FaceCache;

typedef struct facemetrics_ {
    /* All these are 26.6 precision */
    FT_Pos bearing_x;
    FT_Pos bearing_y;
    FT_Vector bearing_rotated;
    FT_Vector advance_rotated;
} FaceMetrics;

typedef struct faceglyph_ {
    FT_UInt glyph_index;
    FT_BitmapGlyph image;

    FT_Pos width;         /* 26.6 */
    FT_Pos height;        /* 26.6 */
    FaceMetrics h_metrics;
    FaceMetrics v_metrics;
} FaceGlyph;

typedef struct facetext_ {
    int length;

    int top;       /* In pixels */
    int left;      /* In pixels */

    FT_Pos min_x;
    FT_Pos max_x;
    FT_Pos min_y;
    FT_Pos max_y;
    FT_Vector offset;
    FT_Vector advance;
    FT_Pos ascender;
    FT_Fixed underline_size;
    FT_Pos underline_pos;

    int buffer_size;
    FaceGlyph **glyphs;
    FT_Vector *posns;

    FaceCache glyph_cache;
} FaceText;

struct facesurface_;

typedef void (* FaceRenderPtr)(int, int, struct facesurface_ *,
                               FT_Bitmap *, FaceColor *);
typedef void (* FaceFillPtr)(int, int, int, int, struct facesurface_ *,
                             FaceColor *);

typedef struct facesurface_ {
    void *buffer;

    unsigned width;
    unsigned height;
    int pitch;

    SDL_PixelFormat *format;

    FaceRenderPtr render_gray;
    FaceRenderPtr render_mono;
    FaceFillPtr fill;

} FaceSurface;

#define PGFT_INTERNALS(f) ((FaceInternals *)((f)->_internals))
typedef struct FaceInternals_ {
    FaceText active_text;
} FaceInternals;

typedef struct PGFT_String_ {
    Py_ssize_t length;
    PGFT_char data[1];
} PGFT_String;

#if defined(PGFT_DEBUG_CACHE)
#define PGFT_FACE_CACHE(f) (PGFT_INTERNALS(f)->active_text.glyph_cache)
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
#   define FREETYPE_STATE \
    FREETYPE_MOD_STATE(PyState_FindModule(&_freetypemodule))
#else
    extern _FreeTypeState _modstate;
#   define FREETYPE_MOD_STATE(mod) (&_modstate)
#   define FREETYPE_STATE FREETYPE_MOD_STATE(0)
#endif

#define ASSERT_GRAB_FREETYPE(ft_ptr, rvalue)                    \
    ft_ptr = FREETYPE_STATE->freetype;                          \
    if (!ft_ptr) {                                              \
        PyErr_SetString(PyExc_RuntimeError,                     \
            "The FreeType 2 library hasn't been initialized");  \
        return (rvalue);                                        \
    }




/**********************************************************
 * Internal API
 **********************************************************/

/**************************************** General functions ******************/
const char *_PGFT_GetError(FreeTypeInstance *);
void _PGFT_Quit(FreeTypeInstance *);
int _PGFT_Init(FreeTypeInstance **, int);
long _PGFT_Face_GetAscender(FreeTypeInstance *, PgFaceObject *);
long _PGFT_Face_GetAscenderSized(FreeTypeInstance *, PgFaceObject *,
                                 FT_UInt16);
long _PGFT_Face_GetDescender(FreeTypeInstance *, PgFaceObject *);
long _PGFT_Face_GetDescenderSized(FreeTypeInstance *, PgFaceObject *,
                                  FT_UInt16);
long _PGFT_Face_GetHeight(FreeTypeInstance *, PgFaceObject *);
long _PGFT_Face_GetHeightSized(FreeTypeInstance *, PgFaceObject *,
                               FT_UInt16);
long _PGFT_Face_GetGlyphHeightSized(FreeTypeInstance *, PgFaceObject *,
                                    FT_UInt16);
int _PGFT_Face_IsFixedWidth(FreeTypeInstance *, PgFaceObject *);
const char *_PGFT_Face_GetName(FreeTypeInstance *, PgFaceObject *);
int _PGFT_TryLoadFont_Filename(FreeTypeInstance *,
                               PgFaceObject *, const char *, int);
#ifdef HAVE_PYGAME_SDL_RWOPS
int _PGFT_TryLoadFont_RWops(FreeTypeInstance *,
                            PgFaceObject *, SDL_RWops *, int);
#endif
void _PGFT_UnloadFace(FreeTypeInstance *, PgFaceObject *);


/**************************************** Metrics management *****************/
int _PGFT_GetTextRect(FreeTypeInstance *, PgFaceObject *,
                      const FaceRenderMode *, PGFT_String *,
                      SDL_Rect *);
int _PGFT_GetMetrics(FreeTypeInstance *, PgFaceObject *,
                     PGFT_char, const FaceRenderMode *,
                     FT_UInt *, long *, long *, long *, long *,
                     double *, double *);
void _PGFT_GetRenderMetrics(const FaceRenderMode *, FaceText *,
                            unsigned *, unsigned *, FT_Vector *,
                            FT_Pos *, FT_Fixed *);


/**************************************** Rendering **************************/
PyObject *_PGFT_Render_PixelArray(FreeTypeInstance *, PgFaceObject *,
                                  const FaceRenderMode *,
                                  PGFT_String *, int *, int *);
SDL_Surface *_PGFT_Render_NewSurface(FreeTypeInstance *, PgFaceObject *,
                                     const FaceRenderMode *, PGFT_String *,
                                     FaceColor *, FaceColor *, SDL_Rect *);
int _PGFT_Render_ExistingSurface(FreeTypeInstance *, PgFaceObject *,
                                 const FaceRenderMode *, PGFT_String *,
                                 SDL_Surface *, int, int,
                                 FaceColor *, FaceColor *, SDL_Rect *);
int _PGFT_BuildRenderMode(FreeTypeInstance *, PgFaceObject *,
                          FaceRenderMode *, int, int, int);
int _PGFT_CheckStyle(FT_UInt32);


/**************************************** Render callbacks *******************/
void __fill_glyph_RGB1(int, int, int, int, FaceSurface *, FaceColor *);
void __fill_glyph_RGB2(int, int, int, int, FaceSurface *, FaceColor *);
void __fill_glyph_RGB3(int, int, int, int, FaceSurface *, FaceColor *);
void __fill_glyph_RGB4(int, int, int, int, FaceSurface *, FaceColor *);

void __fill_glyph_GRAY1(int, int, int, int, FaceSurface *, FaceColor *);

void __render_glyph_MONO1(int, int, FaceSurface *, FT_Bitmap *, FaceColor *);
void __render_glyph_MONO2(int, int, FaceSurface *, FT_Bitmap *, FaceColor *);
void __render_glyph_MONO3(int, int, FaceSurface *, FT_Bitmap *, FaceColor *);
void __render_glyph_MONO4(int, int, FaceSurface *, FT_Bitmap *, FaceColor *);

void __render_glyph_RGB1(int, int, FaceSurface *, FT_Bitmap *, FaceColor *);
void __render_glyph_RGB2(int, int, FaceSurface *, FT_Bitmap *, FaceColor *);
void __render_glyph_RGB3(int, int, FaceSurface *, FT_Bitmap *, FaceColor *);
void __render_glyph_RGB4(int, int, FaceSurface *, FT_Bitmap *, FaceColor *);

void __render_glyph_GRAY1(int, int, FaceSurface *, FT_Bitmap *, FaceColor *);
void __render_glyph_MONO_as_GRAY1(int, int, FaceSurface *,
                                  FT_Bitmap *, FaceColor *);
void __render_glyph_GRAY_as_MONO1(int, int, FaceSurface *,
                                  FT_Bitmap *, FaceColor *);


/**************************************** Face text management ***************/
int _PGFT_FaceTextInit(FreeTypeInstance *, PgFaceObject *);
void _PGFT_FaceTextFree(PgFaceObject *);
FaceText *_PGFT_LoadFaceText(FreeTypeInstance *, PgFaceObject *,
                            const FaceRenderMode *, PGFT_String *);
int _PGFT_LoadGlyph(FaceGlyph *, PGFT_char, const FaceRenderMode *, void *);


/**************************************** Glyph cache management *************/
int _PGFT_Cache_Init(FreeTypeInstance *, FaceCache *);
void _PGFT_Cache_Destroy(FaceCache *);
void _PGFT_Cache_Cleanup(FaceCache *);
FaceGlyph *_PGFT_Cache_FindGlyph(FT_UInt32, const FaceRenderMode *,
                                 FaceCache *, void *);


/**************************************** Unicode ****************************/
PGFT_String *_PGFT_EncodePyString(PyObject *, int);
#define PGFT_String_GET_DATA(s) ((s)->data)
#define PGFT_String_GET_LENGTH(s) ((s)->length)
#define _PGFT_FreeString _PGFT_free


/**************************************** Internals **************************/
void _PGFT_SetError(FreeTypeInstance *, const char *, FT_Error);
FT_Face _PGFT_GetFace(FreeTypeInstance *, PgFaceObject *);
FT_Face _PGFT_GetFaceSized(FreeTypeInstance *, PgFaceObject *, int);
void _PGFT_BuildScaler(PgFaceObject *, FTC_Scaler, int);
#define _PGFT_malloc PyMem_Malloc
#define _PGFT_free   PyMem_Free

#endif
