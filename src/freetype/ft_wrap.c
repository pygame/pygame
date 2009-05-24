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

#define PYGAME_FREETYPE_INTERNAL

#include "ft_mod.h"
#include "ft_wrap.h"
#include "pgfreetype.h"
#include "pgtypes.h"
#include "freetypebase_doc.h"

#include FT_MODULE_H

void    _PGFT_SetError(FreeTypeInstance *, const char *, FT_Error);
FT_Face _PGFT_GetFace(FreeTypeInstance *, PyFreeTypeFont *);
FT_Face _PGFT_GetFaceSized(FreeTypeInstance *, PyFreeTypeFont *, int);
void    _PGFT_BuildScaler(PyFreeTypeFont *, FTC_Scaler, int);
int     _PGFT_LoadGlyph(FreeTypeInstance *, PyFreeTypeFont *, FTC_Scaler, int, FT_Glyph *);
void    _PGFT_GetMetrics_INTERNAL(FT_Glyph, int *, int *, int *, int *, int *);


static FT_Error
_PGFT_face_request(FTC_FaceID face_id, 
    FT_Library library, 
    FT_Pointer request_data, 
    FT_Face *aface)
{
    FontId *id = GET_FONT_ID(face_id); 
    FT_Error error = 0;
    
    Py_BEGIN_ALLOW_THREADS;
        error = FT_Open_Face(library, &id->open_args, id->face_index, aface);
    Py_END_ALLOW_THREADS;

    return error;
}

void
_PGFT_SetError(FreeTypeInstance *ft, const char *error_msg, FT_Error error_id)
{
#undef __FTERRORS_H__
#define FT_ERRORDEF( e, v, s )  { e, s },
#define FT_ERROR_START_LIST     {
#define FT_ERROR_END_LIST       {0, 0}};
    static const struct
    {
        int          err_code;
        const char*  err_msg;
    } ft_errors[] = 
#include FT_ERRORS_H

          int i;
    const char *ft_msg;

    ft_msg = NULL;
    for (i = 0; ft_errors[i].err_msg != NULL; ++i)
    {
        if (error_id == ft_errors[i].err_code)
        {
            ft_msg = ft_errors[i].err_msg;
            break;
        }
    }

    if (ft_msg)
        sprintf(ft->_error_msg, "%s: %s", error_msg, ft_msg);
    else
        strcpy(ft->_error_msg, error_msg);
}


FT_UInt16 *
PGFT_BuildUnicodeString(PyObject *obj, int *must_free)
{
    FT_UInt16 *utf16_buffer = NULL;
    *must_free = 0;

#ifdef IS_PYTHON_3
    /* 
     * If this is Python 3 and we pass an unicode string,
     * we can access directly its internal contents, as
     * they are in UCS-2
     */
    if (PyUnicode_Check(obj))
    {
        utf16_buffer = (FT_UInt16 *)PyUnicode_AS_UNICODE(obj);
    } else
#endif

    /*
     * If we pass a Bytes array, we assume it's standard
     * text encoded in Latin1 (SDL_TTF does the same).
     * We need to expand each character into 2 bytes because
     * FreeType expects UTF16 encodings.
     *
     * TODO: What happens if the user passes a byte array
     * representing e.g. a UTF8 string? He would be mostly
     * stupid, yes, but we should probably handle it.
     */
    if (Bytes_Check(obj))
    {
        const char *latin1_buffer;
        size_t i, len;

        latin1_buffer = (const char *)Bytes_AsString(obj);
        len = strlen(latin1_buffer);

        utf16_buffer = malloc((len + 1) * sizeof(FT_UInt16));

        for (i = 0; i < len; ++i)
            utf16_buffer[i] = (FT_UInt16)latin1_buffer[i];

        utf16_buffer[i] = 0;
        *must_free = 1;
    }

    return utf16_buffer;
}

const char *
PGFT_GetError(FreeTypeInstance *ft)
{
    return ft->_error_msg;
}

int
PGFT_Face_IsFixedWidth(PyFreeTypeFont *font)
{
    return FT_IS_FIXED_WIDTH(font->face);
}

const char *
PGFT_Face_GetName(PyFreeTypeFont *font)
{
    return font->face->family_name;
}


const char *
PGFT_Face_GetFormat(PyFreeTypeFont *font)
{
#ifdef HAS_X11
    return FT_Get_X11_Font_Format(font->face);
#else
    return ""; /* FIXME: Find a portable solution for native Win32 freetype */
#endif
}

int
PGFT_Face_GetHeight(PyFreeTypeFont *font)
{
    return font->face->height;
}

FT_Face
_PGFT_GetFace(FreeTypeInstance *ft,
        PyFreeTypeFont *font)
{
    FT_Error error;
    FT_Face face;

    error = FTC_Manager_LookupFace(ft->cache_manager,
            (FTC_FaceID)font,
            &face);

    if (error)
    {
        _PGFT_SetError(ft, "Failed to load face", error);
        return NULL;
    }

    return face;
}

void
_PGFT_BuildScaler(PyFreeTypeFont *font, FTC_Scaler scale, int size)
{
    scale->face_id = (FTC_FaceID)font;
    scale->width = scale->height = (pguint32)(size * 64);
    scale->pixel = 0;
    scale->x_res = scale->y_res = 0;
}

FT_Face
_PGFT_GetFaceSized(FreeTypeInstance *ft,
        PyFreeTypeFont *font,
        int face_size)
{
    FT_Error error;
    FTC_ScalerRec scale;
    FT_Size _fts;

    _PGFT_BuildScaler(font, &scale, face_size);

    /*
     * TODO: Check if face has already been sized?
     */

    error = FTC_Manager_LookupSize(ft->cache_manager, 
            &scale, &_fts);

    if (error)
    {
        _PGFT_SetError(ft, "Failed to resize face", error);
        return NULL;
    }

    return _fts->face;
}

int
_PGFT_LoadGlyph(FreeTypeInstance *ft, 
        PyFreeTypeFont *font,
        FTC_Scaler scale, 
        int character, FT_Glyph *glyph)
{
    FT_Error error;
    FT_UInt32 char_index;

    char_index = FTC_CMapCache_Lookup(
            ft->cache_charmap, 
            (FTC_FaceID)font,
            -1, (FT_UInt32)character);

    if (char_index == 0)
        return -1;

    error = FTC_ImageCache_LookupScaler(
            ft->cache_img,
            scale,
            FT_LOAD_DEFAULT, /* TODO: proper load flags */
            char_index,
            glyph, NULL);

    return error;
}

void _PGFT_GetMetrics_INTERNAL(FT_Glyph glyph, 
        int *minx, int *maxx, int *miny, int *maxy, int *advance)
{
    FT_BBox box;

    /*
     * FIXME: We need to return pixel-based coordinates...
     * It would make sense to use FT_GLYPH_BBOX_TRUNCATE
     * to get the fixed point coordinates truncated, but
     * the results are usually off by 1 pixel from what we
     * get from SDL_TTF.
     *
     * Using FT_GLYPH_BBOX_PIXELS (truncated + grid fitted
     * coordinates) we get exactly the same results as
     * SDL_TTF, but does SDL actually do this properly?
     * 
     * Which are the *right* results? 
     */
    FT_Glyph_Get_CBox(glyph, FT_GLYPH_BBOX_PIXELS, &box);

    *minx = box.xMin;
    *maxx = box.xMax;
    *miny = box.yMin;
    *maxy = box.yMax;
    *advance = (glyph->advance.x >> 16);
}


int PGFT_GetMetrics(FreeTypeInstance *ft, PyFreeTypeFont *font,
        int character, int font_size, 
        int *minx, int *maxx, int *miny, int *maxy, int *advance)
{
    FT_Error error;
    FTC_ScalerRec scale;
    FT_Face face;
    FT_Glyph glyph;

    _PGFT_BuildScaler(font, &scale, font_size);
    face = font->face;

    error = _PGFT_LoadGlyph(ft, font, &scale, character, &glyph);

    if (error)
    {
        _PGFT_SetError(ft, "Failed to load glyph metrics", error);
        return error;
    }

    _PGFT_GetMetrics_INTERNAL(glyph, minx, maxx, miny, maxy, advance);
    return 0;
}

int
PGFT_GetTextSize(FreeTypeInstance *ft, PyFreeTypeFont *font,
        const FT_UInt16 *text, int font_size, int *w, int *h)
{
#define UNICODE_BOM_NATIVE	0xFEFF
#define UNICODE_BOM_SWAPPED	0xFFFE

    const FT_UInt16 *ch;
    int swapped;
    FTC_ScalerRec scale;
    FT_Face face;
    FT_Glyph glyph;

    int minx, maxx, miny, maxy, x, z;
    int gl_maxx, gl_maxy, gl_minx, gl_miny, gl_advance;

    _PGFT_BuildScaler(font, &scale, font_size);

    /* FIXME: Some way to set the system's default ? */
    swapped = 0;
    x = 0;
    face = font->face;

    minx = maxx = 0;
    miny = maxy = 0;

    for (ch = text; *ch; ++ch)
    {
        FT_UInt16 c = *ch;

        if (c == UNICODE_BOM_NATIVE || c == UNICODE_BOM_SWAPPED)
        {
            swapped = (c == UNICODE_BOM_SWAPPED);
            if (text == ch)
                ++text;

            continue;
        }

        if (swapped)
            c = (FT_UInt16)((c << 8) | (c >> 8));

        if (_PGFT_LoadGlyph(ft, font, &scale, c, &glyph) != 0)
            continue;

        /* TODO: Handle kerning */

        _PGFT_GetMetrics_INTERNAL(glyph, 
                &gl_minx, &gl_maxx, &gl_miny, &gl_maxy, &gl_advance);

        z = x + gl_minx;
		if (minx > z) 
            minx = z;

        z = x + (gl_advance > gl_maxx) ? gl_advance : gl_maxx;
		if (maxx < z) 
            maxx = z;

        if (gl_miny < miny)
            miny = gl_miny;

        if (gl_maxy > maxy)
            maxy = gl_maxy;

		x += gl_advance;
    }

    *w = (maxx - minx);
    *h = (maxy - miny);
    return 0;
}


int
PGFT_TryLoadFont(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    FT_Face face;
    face = _PGFT_GetFace(ft, font);

    if (!face)
        return -1;

    font->face = face;
    return 0;
}

void
PGFT_UnloadFont(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    FTC_Manager_RemoveFaceID(ft->cache_manager, (FTC_FaceID)font);
}

void
PGFT_Quit(FreeTypeInstance *ft)
{
    if (ft == NULL)
        return;

    /* TODO: Free caches */

    if (ft->cache_manager)
        FTC_Manager_Done(ft->cache_manager);

    if (ft->library)
        FT_Done_FreeType(ft->library);

    free(ft->_error_msg);
    free(ft);
}

int
PGFT_Init(FreeTypeInstance **_instance)
{
    FreeTypeInstance *inst = NULL;

    inst = malloc(sizeof(FreeTypeInstance));

    if (!inst)
        goto error_cleanup;

    memset(inst, 0, sizeof(FreeTypeInstance));
    inst->_error_msg = malloc(1024);
    
    if (FT_Init_FreeType(&inst->library) != 0)
        goto error_cleanup;

    if (FTC_Manager_New(inst->library, 0, 0, 0,
            &_PGFT_face_request, NULL,
            &inst->cache_manager) != 0)
        goto error_cleanup;

    if (FTC_CMapCache_New(inst->cache_manager, 
            &inst->cache_charmap) != 0)
        goto error_cleanup;

    if (FTC_SBitCache_New(inst->cache_manager,
            &inst->cache_sbit) != 0)
        goto error_cleanup;

    if (FTC_ImageCache_New(inst->cache_manager,
            &inst->cache_img) != 0)
        goto error_cleanup;

    *_instance = inst;
    return 0;

error_cleanup:
    PGFT_Quit(inst);
    *_instance = NULL;

    return -1;
}
