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

#define FP_1616_FLOAT(i)    ((float)((int)(i) / 65536.0f))
#define FP_248_FLOAT(i)     ((float)((int)(i) / 256.0f))
#define FP_266_FLOAT(i)     ((float)((int)(i) / 64.0f))

#define UNICODE_BOM_NATIVE	0xFEFF
#define UNICODE_BOM_SWAPPED	0xFFFE

#define METRICS_RETURN_AVERAGED_HEIGHT

void    _PGFT_SetError(FreeTypeInstance *, const char *, FT_Error);
FT_Face _PGFT_GetFace(FreeTypeInstance *, PyFreeTypeFont *);
FT_Face _PGFT_GetFaceSized(FreeTypeInstance *, PyFreeTypeFont *, int);
void    _PGFT_BuildScaler(PyFreeTypeFont *, FTC_Scaler, int);
int     _PGFT_LoadGlyph(FreeTypeInstance *, PyFreeTypeFont *, int,
    FTC_Scaler, int, FT_Glyph *, FT_UInt32 *);
void    _PGFT_GetMetrics_INTERNAL(FT_Glyph, FT_UInt, int *, int *, int *, int *, int *);
int     _PGFT_Render_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    const FT_UInt16 *text, int font_size, 
    FT_Byte *_buffer, int width, int height, int pitch);


static FT_Error
_PGFT_face_request(FTC_FaceID face_id, 
    FT_Library library, 
    FT_Pointer request_data, 
    FT_Face *aface)
{
    FontId *id = (FontId *)face_id; 
    FT_Error error;
    
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

    /* 
     * If this is Python 3 and we pass an unicode string,
     * we can access directly its internal contents, as
     * they are in UCS-2
     */
    if (PyUnicode_Check(obj))
    {
        utf16_buffer = (FT_UInt16 *)PyUnicode_AS_UNICODE(obj);
    }
    else if (Bytes_Check(obj))
    {
    
        const char *latin1_buffer;
        size_t i, len;

        latin1_buffer = (const char *)Bytes_AsString(obj);
        len = strlen(latin1_buffer);

        utf16_buffer = malloc((len + 1) * sizeof(FT_UInt16));
        if (!utf16_buffer)
            return NULL;

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
PGFT_Face_IsFixedWidth(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    FT_Face face;
    face = _PGFT_GetFace(ft, font);

    return face ? FT_IS_FIXED_WIDTH(face) : 0;
}

const char *
PGFT_Face_GetName(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    FT_Face face;
    face = _PGFT_GetFace(ft, font);

    return face ? face->family_name : ""; 
}

int
PGFT_Face_GetHeight(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    FT_Face face;
    face = _PGFT_GetFace(ft, font);

    return face ? face->height : 0;
}

FT_Face
_PGFT_GetFace(FreeTypeInstance *ft,
    PyFreeTypeFont *font)
{
    FT_Error error;
    FT_Face face;

    error = FTC_Manager_LookupFace(ft->cache_manager,
        (FTC_FaceID)(&font->id),
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
    scale->face_id = (FTC_FaceID)(&font->id);
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
    int do_render,
    FTC_Scaler scale, 
    int character, 
    FT_Glyph *glyph, 
    FT_UInt32 *_index)
{
    FT_Error error = 0;
    FT_UInt32 char_index;

    FT_ULong render_mode = do_render ? 
        (FT_ULong)FT_LOAD_RENDER : 
        (FT_ULong)FT_LOAD_DEFAULT;

    char_index = FTC_CMapCache_Lookup(
        ft->cache_charmap, 
            (FTC_FaceID)(&font->id),
            -1, (FT_UInt32)character);

    if (_index)
        *_index = char_index;

    if (char_index == 0)
        return -1;

    if (glyph)
    {
        error = FTC_ImageCache_LookupScaler(
            ft->cache_img,
                scale,
                render_mode,
                char_index,
                glyph, NULL);
    }

    return error;
}

void _PGFT_GetMetrics_INTERNAL(FT_Glyph glyph, FT_UInt bbmode,
    int *minx, int *maxx, int *miny, int *maxy, int *advance)
{
    FT_BBox box;
    FT_Glyph_Get_CBox(glyph, bbmode, &box);

    *minx = box.xMin;
    *maxx = box.xMax;
    *miny = box.yMin;
    *maxy = box.yMax;
    *advance = glyph->advance.x;

    if (bbmode == FT_GLYPH_BBOX_TRUNCATE ||
        bbmode == FT_GLYPH_BBOX_PIXELS)
        *advance >>= 16;
}


int PGFT_GetMetrics(FreeTypeInstance *ft, PyFreeTypeFont *font,
    int character, int font_size, int bbmode,
    void *minx, void *maxx, void *miny, void *maxy, void *advance)
{
    FT_Error error;
    FTC_ScalerRec scale;
    FT_Glyph glyph;

    _PGFT_BuildScaler(font, &scale, font_size);

    error = _PGFT_LoadGlyph(ft, font, 0, &scale, character, &glyph, NULL);

    if (error)
    {
        _PGFT_SetError(ft, "Failed to load glyph metrics", error);
        return error;
    }

    _PGFT_GetMetrics_INTERNAL(glyph, (FT_UInt)bbmode, minx, maxx, miny, maxy, advance);

    if (bbmode == FT_BBOX_EXACT || bbmode == FT_BBOX_EXACT_GRIDFIT)
    {
        *(float *)minx =    (FP_266_FLOAT(*(int *)minx));
        *(float *)miny =    (FP_266_FLOAT(*(int *)miny));
        *(float *)maxx =    (FP_266_FLOAT(*(int *)maxx));
        *(float *)maxy =    (FP_266_FLOAT(*(int *)maxy));
        *(float *)advance = (FP_1616_FLOAT(*(int *)advance));
    }

    return 0;
}

int
PGFT_GetTextSize(FreeTypeInstance *ft, PyFreeTypeFont *font,
    const FT_UInt16 *text, int font_size, int *w, int *h)
{
    const FT_UInt16 *ch;
    int swapped, use_kerning;
    FT_UInt32 prev_index, cur_index;

    FTC_ScalerRec scale;
    FT_Face face;
    FT_Glyph glyph;
    FT_Size fontsize;

    int minx, maxx, miny, maxy, x, z;
    int gl_maxx, gl_maxy, gl_minx, gl_miny, gl_advance;

    _PGFT_BuildScaler(font, &scale, font_size);

    /* FIXME: Some way to set the system's default ? */
    swapped = 0;
    x = 0;
    face = _PGFT_GetFace(ft, font);

    if (!face)
        return -1;

    minx = maxx = 0;
    miny = maxy = 0;
    prev_index = 0;

    use_kerning = FT_HAS_KERNING(face);

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

        if (_PGFT_LoadGlyph(ft, font, 0, &scale, c, &glyph, &cur_index) != 0)
            continue;

        _PGFT_GetMetrics_INTERNAL(glyph, FT_GLYPH_BBOX_PIXELS,
            &gl_minx, &gl_maxx, &gl_miny, &gl_maxy, &gl_advance);

        if (use_kerning && prev_index)
        {
            FT_Vector delta;
            FT_Get_Kerning(face, prev_index, cur_index, ft_kerning_default, &delta); 
            x += delta.x >> 6;
        }

        z = x + gl_minx;
        if (minx > z)
            minx = z;
		
        /* TODO: Handle bold fonts */

        z = x + MAX(gl_maxx, gl_advance);
        if (maxx < z)
            maxx = z;

#ifndef METRICS_RETURN_AVERAGED_HEIGHT
        miny = MIN(gl_miny, miny);
        maxy = MAX(gl_maxy, maxy);
#endif

        x += gl_advance;
        prev_index = cur_index;
    }

    *w = (maxx - minx);

#ifdef METRICS_RETURN_AVERAGED_HEIGHT
    if (FTC_Manager_LookupSize(ft->cache_manager, &scale, &fontsize) != 0)
        return -1;

    *h = (fontsize->metrics.height + 63) >> 6;
#else
    *h = (maxy - miny);
#endif

    return 0;
}

#ifdef HAVE_PYGAME_SDL_VIDEO
PyObject *PGFT_Render_NewSurface(FreeTypeInstance *ft, PyFreeTypeFont *font,
    const FT_UInt16 *text, int font_size, int *_width, int *_height)
{
    int width, height, locked = 0;
    SDL_Surface *surface = NULL;

    if (PGFT_GetTextSize(ft, font, text, font_size, &width, &height) != 0 ||
        width == 0)
    {
        _PGFT_SetError(ft, "Error when building text size", 0);
        return NULL;
    }

    surface = SDL_CreateRGBSurface(SDL_SWSURFACE,
        width, height, 8, 0, 0, 0, 0);

    if (!surface)
    {
        _PGFT_SetError(ft, SDL_GetError (), 0);
        return NULL;
    }

    if (SDL_MUSTLOCK (surface))
    {
        if (SDL_LockSurface (surface) == -1)
        {
            _PGFT_SetError(ft, SDL_GetError (), 0);
            SDL_FreeSurface (surface);
            return NULL;
        }
        locked = 1;
    }

    if (_PGFT_Render_INTERNAL(ft, font, text, font_size, 
            surface->pixels,
            surface->w, surface->h, surface->pitch) != 0)
    {
        _PGFT_SetError(ft, "Failed to render text", 0);
        SDL_FreeSurface(surface);
        return NULL;
    }

    *_width = width;
    *_height = height;

    if (locked)
        SDL_UnlockSurface (surface);

    return PySDLSurface_NewFromSDLSurface(surface);
}
#endif

PyObject *PGFT_Render_PixelArray(FreeTypeInstance *ft, PyFreeTypeFont *font,
    const FT_UInt16 *text, int font_size, int *_width, int *_height)
{
    int width, height;
    FT_Byte *buffer = NULL;
    PyObject *array = NULL;

    if (PGFT_GetTextSize(ft, font, text, font_size, &width, &height) != 0 ||
        width == 0)
    {
        _PGFT_SetError(ft, "Error when building text size", 0);
        goto cleanup;
    }

    buffer = calloc((size_t)(width * height), sizeof(FT_Byte));
    if (!buffer)
    {
        _PGFT_SetError(ft, "Could not allocate memory", 0);
        goto cleanup;
    }

    if (_PGFT_Render_INTERNAL(ft, font, text, 
            font_size, buffer, width, height, width) != 0)
    {
        _PGFT_SetError(ft, "Failed to render text", 0);
        goto cleanup;
    }

    *_width = width;
    *_height = height;

    array = Bytes_FromStringAndSize(buffer, width * height);

cleanup:
    if (buffer)
        free(buffer);

    return array;
}

int _PGFT_Render_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    const FT_UInt16 *text, int font_size, 
    FT_Byte *_buffer, int width, int height, int pitch)
{
    const FT_UInt16 *ch;

    FTC_ScalerRec scale;
    FT_Face face;
    FT_Glyph glyph;
    FT_Bitmap *bitmap;

    FT_UInt32 prev_index, cur_index;

    int swapped, use_kerning;
    int pen_x, pen_y;
    int x_advance;

    FT_Byte *_buffer_cap;

    _PGFT_BuildScaler(font, &scale, font_size);
    face = _PGFT_GetFace(ft, font);

    if (!face)
    {
        _PGFT_SetError(ft, "Failed to cache font face", 0);
        return -1;
    }

    _buffer_cap = _buffer + (width * height);
    use_kerning = FT_HAS_KERNING(face);
    prev_index = 0;

    /* FIXME: Some way to set the system's default ? */
    swapped = 0;

    pen_x = 0;
    pen_y = height;

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

        if (_PGFT_LoadGlyph(ft, font, 1 /* RENDER! */, &scale, c, &glyph, &cur_index) != 0)
            continue; /* FIXME: fail if we cannot find a char? */

        assert(glyph->format == FT_GLYPH_FORMAT_BITMAP);
        bitmap = &((FT_BitmapGlyph)glyph)->bitmap;

        if (use_kerning && prev_index)
        {
            FT_Vector delta;
            FT_Get_Kerning(face, prev_index, cur_index, ft_kerning_default, &delta); 
            pen_x += delta.x >> 6;
        }

        x_advance = (glyph->advance.x + 0x8000) >> 16;

        /*
         * Render bitmap on the surface at coords:
         *      pen_x + bitmap->left, pen_y - bitmap->top
         */
        {
            const int left = ((FT_BitmapGlyph)glyph)->left;
            const int top = ((FT_BitmapGlyph)glyph)->top;
            int j;

            FT_Byte *dst = _buffer + 
                pen_x + left +
                (pen_y - top) * pitch;

            FT_Byte *src = bitmap->buffer;

            for (j = 0; j < bitmap->rows; ++j)
            {
                if (dst < _buffer || dst + bitmap->width > _buffer_cap)
                {
                    fprintf(stderr, 
                        "HEAP CORRUPTION rendering '%c': "
                        "text size(%d, %d) | left %d | top %d | pen_x %d | pen_y %d |\n",
                        c, width, height, left, top, pen_x, pen_y);
                    continue;
                }

                memcpy(dst, src, (size_t)bitmap->width);

                dst += pitch;
                src += bitmap->pitch;
            }
        }


        pen_x += x_advance + 1;
        prev_index = cur_index;
    }

    return 0;
}


int
PGFT_TryLoadFont_Filename(FreeTypeInstance *ft, 
    PyFreeTypeFont *font, 
    const char *filename, 
    int face_index)
{
    char *filename_alloc;
    size_t file_len;

    file_len = strlen(filename);
    filename_alloc = malloc(file_len + 1);
    if (!filename_alloc)
    {
        _PGFT_SetError(ft, "Could not allocate memory", 0);
        return -1;
    }

    strcpy(filename_alloc, filename);
    filename_alloc[file_len] = 0;

    font->id.face_index = face_index;
    font->id.open_args.flags = FT_OPEN_PATHNAME;
    font->id.open_args.pathname = filename_alloc;

    return _PGFT_GetFace(ft, font) ? 0 : -1;
}

void
PGFT_UnloadFont(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    if (ft != NULL)
        FTC_Manager_RemoveFaceID(ft->cache_manager, (FTC_FaceID)(&font->id));

    free(font->id.open_args.pathname);
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

    if (ft->_error_msg)
        free (ft->_error_msg);
    
    free (ft);
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
