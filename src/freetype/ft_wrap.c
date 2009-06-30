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

#ifdef HAVE_PYGAME_SDL_VIDEO
#   include "surface.h"
#endif

#define FP_1616_FLOAT(i)    ((float)((int)(i) / 65536.0f))
#define FP_248_FLOAT(i)     ((float)((int)(i) / 256.0f))
#define FP_266_FLOAT(i)     ((float)((int)(i) / 64.0f))

#define PGFT_FLOOR(x)  (   (x)        & -64 )
#define PGFT_CEIL(x)   ( ( (x) + 63 ) & -64 )
#define PGFT_ROUND(x)  ( ( (x) + 32 ) & -64 )
#define PGFT_TRUNC(x)  (   (x) >> 6 )

#define UNICODE_BOM_NATIVE	0xFEFF
#define UNICODE_BOM_SWAPPED	0xFFFE

#define FONT_RENDER_HORIZONTAL  0
#define FONT_RENDER_VERTICAL    1

#define MAX_GLYPHS      64

typedef struct __fontsurface
{
    void *buffer;
    void *buffer_cap;

    int x_offset;
    int y_offset;

    int width;
    int height;
    int glyph_height;
    int pitch;

    SDL_PixelFormat *format;
    void (* render)(int, int, struct __fontsurface *, FT_Bitmap *, PyColor *);

} FontSurface;


typedef void (* FontRenderPtr)(int, int, FontSurface *, FT_Bitmap *, PyColor *);


void    _PGFT_SetError(FreeTypeInstance *, const char *, FT_Error);
FT_Face _PGFT_GetFace(FreeTypeInstance *, PyFreeTypeFont *);
FT_Face _PGFT_GetFaceSized(FreeTypeInstance *, PyFreeTypeFont *, int);
void    _PGFT_BuildScaler(PyFreeTypeFont *, FTC_Scaler, int);
int     _PGFT_LoadGlyph(FreeTypeInstance *, PyFreeTypeFont *, int,
    FTC_Scaler, int, FT_Glyph *, FT_UInt32 *);
void    _PGFT_GetMetrics_INTERNAL(FT_Glyph, FT_UInt, int *, int *, int *, int *, int *);
int _PGFT_RenderFontText(FreeTypeInstance *ft, PyFreeTypeFont *font, int pt_size, 
        FontRenderMode *render, FontText *text, FT_Vector *advances);


int     _PGFT_Render_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    const FT_UInt16 *text, int font_size, PyColor *fg_color, FontSurface *surf);

int _PGFT_Render_NEW(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    FontText *text, int font_size, PyColor *fg_color, FontSurface *surface, 
    FontRenderMode *render);


void _PGFT_BuildRenderMode(FontRenderMode *mode, float center, int vertical, int hinted, int rotation);

/* blitters */
void __render_glyph_SDL8(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);
void __render_glyph_SDL16(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);
void __render_glyph_SDL24(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);
void __render_glyph_SDL32(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);
void __render_glyph_ByteArray(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);


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

FontText *
PGFT_BuildFontText(FreeTypeInstance *ft, PyFreeTypeFont *font, PyObject *text, int pt_size, FontRenderMode *render)
{
    /*
     * TODO:
     * - Hash the text string and size
     * - store it in the ft instance
     * - automatically free the previous one
     * - return existing ones if available
     */

    FT_Int32 load_flags = FT_LOAD_DEFAULT; 

    int         must_free;
    int         swapped = 0;
    int         string_length = 0;

    FT_UInt16 * buffer = NULL;
    FT_UInt16 * orig_buffer;
    FT_UInt16 * ch;

    FT_Pos      prev_rsb_delta = 0;

    FontText  * ftext = NULL;
    FontGlyph * glyph = NULL;

    FT_Face     face;

    /* compute proper load flags */
    load_flags |= FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH;

    if (render->autohint)
        load_flags |= FT_LOAD_FORCE_AUTOHINT;

    if (render->hinted)
    {
        load_flags |=   render->antialias ?
                        FT_LOAD_TARGET_NORMAL :
                        FT_LOAD_TARGET_MONO;
    }
    else
    {
        load_flags |= FT_LOAD_NO_HINTING;
    }

    /* load our sized face */
    face = _PGFT_GetFaceSized(ft, font, pt_size);

    if (!face)
        return NULL;

    /* get the text as an unicode string */
    orig_buffer = buffer = PGFT_BuildUnicodeString(text, &must_free);

    /* get the length of the text */
    for (ch = buffer; *ch; ++ch)
        string_length++;


    /* create the text struct */
    ftext = malloc(sizeof(FontText));
    ftext->length = string_length;
    ftext->glyphs = calloc((size_t)string_length, sizeof(FontGlyph));


    /* fill it with the glyphs */
    glyph = &(ftext->glyphs[0]);

    for (ch = buffer; *ch; ++ch, ++glyph)
    {
        FT_UInt16 c = *ch;

        if (c == UNICODE_BOM_NATIVE || c == UNICODE_BOM_SWAPPED)
        {
            swapped = (c == UNICODE_BOM_SWAPPED);
            if (buffer == ch)
                ++buffer;

            continue;
        }

        if (swapped)
            c = (FT_UInt16)((c << 8) | (c >> 8));

        glyph->glyph_index = FTC_CMapCache_Lookup(ft->cache_charmap, 
            (FTC_FaceID)(&font->id),
            -1, (FT_UInt32)c);

        /* FIXME: leaks memory, needs to use the cache */
        if (!FT_Load_Glyph(face, glyph->glyph_index, load_flags)  &&
            !FT_Get_Glyph(face->glyph, &glyph->image))
        {
            FT_Glyph_Metrics *metrics = &face->glyph->metrics;

            /* note that in vertical layout, y-positive goes downwards */
            glyph->vvector.x  = metrics->vertBearingX - metrics->horiBearingX;
            glyph->vvector.y  = -metrics->vertBearingY - metrics->horiBearingY;

            glyph->vadvance.x = 0;
            glyph->vadvance.y = -metrics->vertAdvance;

            if ( prev_rsb_delta - face->glyph->lsb_delta >= 32 )
                glyph->delta = -1 << 6;
            else if ( prev_rsb_delta - face->glyph->lsb_delta < -32 )
                glyph->delta = 1 << 6;
            else
                glyph->delta = 0;
        }

    }

    if (must_free)
        free(orig_buffer);

    return ftext;
}

int
_PGFT_RenderFontText(FreeTypeInstance *ft, PyFreeTypeFont *font, int pt_size, 
        FontRenderMode *render, FontText *text, FT_Vector *advances)
{
    FT_Face     face;
    FontGlyph   *glyph;
    FT_Pos      track_kern   = 0;
    FT_UInt     prev_index   = 0;
    FT_Vector*  prev_advance = NULL;
    FT_Vector   extent       = {0, 0};
    FT_Int      i;

    face = _PGFT_GetFaceSized(ft, font, pt_size);

    if (!face)
        return -1;

    if (!render->vertical && render->kerning_degree)
    {
        FT_Fixed  ptsize;

        ptsize = FT_MulFix(face->units_per_EM, face->size->metrics.x_scale);

        if (FT_Get_Track_Kerning(face, ptsize << 10, 
                    -render->kerning_degree, &track_kern))
            track_kern = 0;
        else
            track_kern >>= 10;
    }

    for (i = 0; i < text->length; i++)
    {
        glyph = &(text->glyphs[i]);

        if (!glyph->image)
            continue;

        if (render->vertical)
            advances[i] = glyph->vadvance;
        else
        {
            advances[i] = glyph->image->advance;
            advances[i].x >>= 10;
            advances[i].y >>= 10;

            if (prev_advance)
            {
                prev_advance->x += track_kern;

                if (render->kerning_mode)
                {
                    FT_Vector  kern;

                    FT_Get_Kerning(face, prev_index, glyph->glyph_index,
                            FT_KERNING_UNFITTED, &kern);

                    prev_advance->x += kern.x;
                    prev_advance->y += kern.y;

                    if (render->kerning_mode > 1) /* KERNING_MODE_NORMAL */
                        prev_advance->x += glyph->delta;
                }
            }
        }

        if (prev_advance)
        {
            if (render->hinted)
            {
                prev_advance->x = PGFT_ROUND(prev_advance->x);
                prev_advance->y = PGFT_ROUND(prev_advance->y);
            }

            extent.x += prev_advance->x;
            extent.y += prev_advance->y;
        }

        prev_index   = glyph->glyph_index;
        prev_advance = advances + i;
    }

    if (prev_advance)
    {
        if (render->hinted)
        {
            prev_advance->x = PGFT_ROUND(prev_advance->x);
            prev_advance->y = PGFT_ROUND(prev_advance->y);
        }

        extent.x += prev_advance->x;
        extent.y += prev_advance->y;
    }

    /* store the extent in the last slot */
    i = text->length - 1;
    advances[i] = extent;

    fprintf(stderr, "Text size: %dx%d\n", PGFT_TRUNC(extent.x), PGFT_TRUNC(extent.y));

    return 0;
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
        char *latin1_buffer;
        int i, len;

        Bytes_AsStringAndSize(obj, &latin1_buffer, &len);

        utf16_buffer = malloc((size_t)(len + 1) * sizeof(FT_UInt16));
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

void 
_PGFT_BuildRenderMode(FontRenderMode *mode, float center, int vertical, int antialias, int rotation)
{
    double      radian;
    FT_Fixed    cosinus;
    FT_Fixed    sinus;
    int         angle;

    angle = rotation % 360;
    while (angle < 0) angle += 360;

    mode->kerning_mode = 1;
    mode->kerning_degree = 0;
    mode->center = (FT_Fixed)(center * (1 << 16));
    mode->vertical = (FT_Byte)vertical;
    mode->hinted = (FT_Byte)1;
    mode->autohint = 0;
    mode->antialias = (FT_Byte)antialias;
    mode->matrix = NULL;

    if (angle != 0)
    {
        radian  = angle * 3.14159 / 180.0;
        cosinus = (FT_Fixed)( cos( radian ) * 65536.0 );
        sinus   = (FT_Fixed)( sin( radian ) * 65536.0 );

        mode->_rotation_matrix.xx = cosinus;
        mode->_rotation_matrix.yx = sinus;
        mode->_rotation_matrix.xy = -sinus;
        mode->_rotation_matrix.yy = cosinus;

        mode->matrix = &mode->_rotation_matrix;
    }
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
    int load_flags,
    FTC_Scaler scale, 
    int character, 
    FT_Glyph *glyph, 
    FT_UInt32 *_index)
{
    FT_Error error = 0;
    FT_UInt32 char_index;

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
                load_flags,
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
    const FT_UInt16 *text, int font_size, int *w, int *h, int *h_avg)
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

        miny = MIN(gl_miny, miny);
        maxy = MAX(gl_maxy, maxy);

        x += gl_advance;
        prev_index = cur_index;
    }

    if (w) *w = (maxx - minx);

    if (h) *h = (maxy - miny);

    if (h_avg)
    {
        if (FTC_Manager_LookupSize(ft->cache_manager, &scale, &fontsize) != 0)
            return -1;

        *h_avg = (fontsize->metrics.height + 63) >> 6;
    }

    return 0;
}

#ifdef HAVE_PYGAME_SDL_VIDEO

void __render_glyph_SDL24(int rx, int ry, FontSurface *surface,
    FT_Bitmap *bitmap, PyColor *color)
{
    const int max_x = MIN(rx + bitmap->width, surface->width);
    const int max_y = MIN(ry + bitmap->rows, surface->height);

    FT_Byte *dst = ((FT_Byte *)surface->buffer) + 
        (rx * 3) + (ry * surface->pitch * 3);

    FT_Byte *dst_cpy;

    const FT_Byte *src = bitmap->buffer;
    const FT_Byte *src_cpy;

    FT_UInt32 bgR, bgG, bgB, bgA;
    int j, i;

    if (rx < 0 || ry < 0)
        return;

    for (j = ry; j < max_y; ++j)
    {
        src_cpy = src;
        dst_cpy = dst;

        for (i = rx; i < max_x; ++i, dst_cpy += 3)
        {
            FT_UInt32 alpha = (*src_cpy++); 
            alpha = (alpha * color->a) / 255;

            if (alpha > 0)
            {
                const FT_UInt32 pixel = (FT_UInt32)GET_PIXEL24(dst_cpy);

                GET_RGB_VALS(
                        pixel, surface->format,              
                        bgR, bgG, bgB, bgA);

                ALPHA_BLEND(
                        color->r, color->g, color->b, alpha,
                        bgR, bgG, bgB, bgA);

                SET_PIXEL24_RGB(
                        dst_cpy, surface->format,
                        bgR, bgG, bgB);
            }                                                   
        }                                                       

        dst += surface->pitch * 3;
        src += bitmap->pitch;
    }                                                           
} 

#define _CREATE_SDL_RENDER(bpp, T, _build_pixel)                    \
    void __render_glyph_SDL##bpp(int rx, int ry, FontSurface *surface,\
        FT_Bitmap *bitmap, PyColor *color)                          \
    {                                                               \
        const int max_x = MIN(rx + bitmap->width, surface->width);  \
        const int max_y = MIN(ry + bitmap->rows, surface->height);  \
                                                                    \
        T *dst = ((T*)surface->buffer) + rx + (ry * surface->pitch);\
        T *dst_cpy;                                                 \
                                                                    \
        const FT_Byte *src = bitmap->buffer;                        \
        const FT_Byte *src_cpy;                                     \
                                                                    \
        FT_UInt32 bgR, bgG, bgB, bgA;                               \
        int j, i;                                                   \
                                                                    \
        if (rx < 0 || ry < 0) return;                               \
                                                                    \
        for (j = ry; j < max_y; ++j)                                \
        {                                                           \
            src_cpy = src;                                          \
            dst_cpy = dst;                                          \
                                                                    \
            for (i = rx; i < max_x; ++i, ++dst_cpy)                 \
            {                                                       \
                FT_UInt32 alpha = (*src_cpy++);                     \
                alpha = (alpha * color->a) / 255;                   \
                                                                    \
                if (alpha > 0)                                      \
                {                                                   \
                    GET_RGB_VALS(                                   \
                            *dst_cpy, surface->format,              \
                            bgR, bgG, bgB, bgA);                    \
                                                                    \
                    ALPHA_BLEND(                                    \
                            color->r, color->g, color->b, alpha,    \
                            bgR, bgG, bgB, bgA);                    \
                                                                    \
                    *dst_cpy = (T)(_build_pixel);                   \
                }                                                   \
            }                                                       \
                                                                    \
            dst += surface->pitch;                                  \
            src += bitmap->pitch;                                   \
        }                                                           \
    }

#define BUILD_PIXEL_TRUECOLOR (                                     \
    ((bgR >> surface->format->Rloss) << surface->format->Rshift) |  \
    ((bgG >> surface->format->Gloss) << surface->format->Gshift) |  \
    ((bgB >> surface->format->Bloss) << surface->format->Bshift) |  \
    ((255 >> surface->format->Aloss) << surface->format->Ashift  &  \
     surface->format->Amask)                                        )

#define BUILD_PIXEL_GENERIC (                           \
        SDL_MapRGB(surface->format,                     \
            (FT_Byte)bgR, (FT_Byte)bgG, (FT_Byte)bgB)   )

_CREATE_SDL_RENDER(32,  FT_UInt32,  BUILD_PIXEL_TRUECOLOR)
_CREATE_SDL_RENDER(16,  FT_UInt16,  BUILD_PIXEL_TRUECOLOR)
_CREATE_SDL_RENDER(8,   FT_Byte,    BUILD_PIXEL_GENERIC)

int PGFT_Render_ExistingSurface(FreeTypeInstance *ft, PyFreeTypeFont *font,
    PyObject *text, int font_size, PySDLSurface *_surface,
    int *_width, int *_height, int x, int y,
    PyColor *fgcolor)
{
    static const FontRenderPtr __renderFuncs[] =
    {
        NULL,
        __render_glyph_SDL8,
        __render_glyph_SDL16,
        __render_glyph_SDL24,
        __render_glyph_SDL32
    };

    int width, glyph_height, height, locked = 0;

    SDL_Surface *surface = PySDLSurface_AsSDLSurface(_surface);
    FontSurface font_surf;
    FontRenderMode render_mode;
    FontText *font_text;

    if (SDL_MUSTLOCK(surface))
    {
        if (SDL_LockSurface(surface) == -1)
        {
            _PGFT_SetError(ft, SDL_GetError (), 0);
            SDL_FreeSurface(surface);
            return -1;
        }
        locked = 1;
    }

    /*
     * Setup render mode
     * TODO: get render mode from FT instance
     */
    _PGFT_BuildRenderMode(&render_mode, 0.0f, FONT_RENDER_HORIZONTAL, 1, 45);

    /* build font text */
    font_text = PGFT_BuildFontText(ft, font, text, font_size, &render_mode);

    if (!font_text)
    {
        return -1;
    }

    /*
     * Setup target surface struct
     */
    font_surf.buffer = surface->pixels;
    font_surf.buffer_cap = ((FT_Byte *)surface->pixels) + (surface->pitch * surface->h);
    font_surf.x_offset = x;
    font_surf.y_offset = y;

    font_surf.width = surface->w;
    font_surf.height = surface->h;
    font_surf.glyph_height = glyph_height;
    font_surf.pitch = surface->pitch / surface->format->BytesPerPixel;

    font_surf.format = surface->format;
    font_surf.render = __renderFuncs[surface->format->BytesPerPixel];

    /*
     * Render!
     */
    if (_PGFT_Render_NEW(ft, font, font_text, font_size, fgcolor, &font_surf, &render_mode) != 0)
    {
        _PGFT_SetError(ft, "Failed to render text", 0);
        return -1;
    }

    *_width = 0;
    *_height = 0;

    if (locked)
        SDL_UnlockSurface(surface);

    return 0;
}

PyObject *PGFT_Render_NewSurface(FreeTypeInstance *ft, PyFreeTypeFont *font,
    const FT_UInt16 *text, int font_size, int *_width, int *_height, 
    PyColor *fgcolor, PyColor *bgcolor)
{
    int width, glyph_height, height, locked = 0;
    FT_UInt32 fillcolor, rmask, gmask, bmask, amask;
    SDL_Surface *surface = NULL;

    FontSurface font_surf;

    if (PGFT_GetTextSize(ft, font, text, font_size, &width, &glyph_height, &height) != 0 ||
        width == 0)
    {
        _PGFT_SetError(ft, "Error when building text size", 0);
        return NULL;
    }

#if SDL_BYTEORDER == SDL_BIG_ENDIAN
    rmask = 0xff000000;
    gmask = 0x00ff0000;
    bmask = 0x0000ff00;
    amask = 0x000000ff;
#else
    rmask = 0x000000ff;
    gmask = 0x0000ff00;
    bmask = 0x00ff0000;
    amask = 0xff000000;
#endif

    surface = SDL_CreateRGBSurface(SDL_SWSURFACE, width, height, 
            32, rmask, gmask, bmask, amask);

    if (!surface)
    {
        _PGFT_SetError(ft, SDL_GetError (), 0);
        return NULL;
    }

    if (SDL_MUSTLOCK(surface))
    {
        if (SDL_LockSurface(surface) == -1)
        {
            _PGFT_SetError(ft, SDL_GetError (), 0);
            SDL_FreeSurface(surface);
            return NULL;
        }
        locked = 1;
    }

    if (bgcolor)
    {
        fillcolor = SDL_MapRGBA(surface->format, 
                bgcolor->r, bgcolor->g, bgcolor->b, 255);
    }
    else
    {
        fillcolor = SDL_MapRGBA(surface->format, 0, 0, 0, 0);
    }

    SDL_FillRect(surface, NULL, fillcolor);

    font_surf.buffer = surface->pixels;
    font_surf.buffer_cap = ((FT_Byte *)surface->pixels) + (surface->pitch * surface->h);
    font_surf.x_offset = font_surf.y_offset = 0;

    font_surf.width = surface->w;
    font_surf.height = surface->h;
    font_surf.glyph_height = glyph_height;
    font_surf.pitch = surface->pitch / sizeof(FT_UInt32);

    font_surf.format = surface->format;
    font_surf.render = __render_glyph_SDL32;

    if (_PGFT_Render_INTERNAL(ft, font, text, font_size, fgcolor, &font_surf) != 0)
    {
        _PGFT_SetError(ft, "Failed to render text", 0);
        SDL_FreeSurface(surface);
        return NULL;
    }

    *_width = width;
    *_height = height;

    if (locked)
        SDL_UnlockSurface(surface);

    return PySDLSurface_NewFromSDLSurface(surface);
}
#endif

void __render_glyph_ByteArray(int x, int y, FontSurface *surface,
    FT_Bitmap *bitmap, PyColor *fg_color)
{
    FT_Byte *dst = ((FT_Byte *)surface->buffer) + x + (y * surface->pitch);
    FT_Byte *dst_cpy;

    const FT_Byte *src = bitmap->buffer;
    const FT_Byte *src_cpy;

    int j, i;

    for (j = 0; j < bitmap->rows; ++j)
    {
        src_cpy = src;
        dst_cpy = dst;

        LOOP_UNROLLED4({ *dst_cpy++ = (FT_Byte)(~(*src_cpy++)); }, i, bitmap->width);

        dst += surface->pitch;
        src += bitmap->pitch;
    }
}

PyObject *PGFT_Render_PixelArray(FreeTypeInstance *ft, PyFreeTypeFont *font,
    const FT_UInt16 *text, int font_size, int *_width, int *_height)
{
    int width, height, glyph_height;
    FT_Byte *buffer = NULL;
    PyObject *array = NULL;
    FontSurface surf;

    if (PGFT_GetTextSize(ft, font, text, font_size, &width, &glyph_height, &height) != 0 ||
        width == 0)
    {
        _PGFT_SetError(ft, "Error when building text size", 0);
        goto cleanup;
    }

    buffer = malloc((size_t)(width * height));
    if (!buffer)
    {
        _PGFT_SetError(ft, "Could not allocate memory", 0);
        goto cleanup;
    }

    memset(buffer, 0xFF, (size_t)(width * height));

    surf.buffer = buffer;
    surf.width = surf.pitch = width;
    surf.height = height;
    surf.glyph_height = glyph_height;

    surf.format = NULL;
    surf.render = __render_glyph_ByteArray;

    if (_PGFT_Render_INTERNAL(ft, font, text, font_size, 0x0, &surf) != 0)
    {
        _PGFT_SetError(ft, "Failed to render text", 0);
        goto cleanup;
    }

    *_width = width;
    *_height = height;

    array = Bytes_FromStringAndSize((char *)buffer, width * height);

cleanup:
    if (buffer)
        free(buffer);

    return array;
}

int _PGFT_Render_NEW(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    FontText *text, int font_size, PyColor *fg_color, FontSurface *surface, 
    FontRenderMode *render)
{
    int n;
    FT_Vector pen, advances[MAX_GLYPHS];
    FT_Face face;
    FT_Error error;

    int x = surface->x_offset;
    int y = surface->y_offset;

    /* TODO: return if drawing outside surface */


    /******************************************************
     * Load scaler, size & face
     ******************************************************/
    face = _PGFT_GetFaceSized(ft, font, font_size);

    if (!face)
    {
        _PGFT_SetError(ft, "Failed to resize face", 0);
        return -1;
    }

    /******************************************************
     * Load advance information
     ******************************************************/

    error = _PGFT_RenderFontText(ft, font, font_size, render, text, advances);

    if (error)
    {
        _PGFT_SetError(ft, "Failed to load advance glyph advances", error);
        return error;
    }

    /******************************************************
     * Prepare pen for drawing 
     ******************************************************/

    /* change to Cartesian coordinates */
    y = surface->height - y;

    /* get the extent, which we store in the last slot */
    pen = advances[text->length - 1];

    pen.x = FT_MulFix(pen.x, render->center); /* TODO: What is sc->center? */
    pen.y = FT_MulFix(pen.y, render->center);

    /* get pen position */
    if (render->matrix && FT_IS_SCALABLE(face))
    {
        FT_Vector_Transform(&pen, render->matrix);
        pen.x = (x << 6) - pen.x;
        pen.y = (y << 6) - pen.y;
    }
    else
    {
        pen.x = PGFT_ROUND(( x << 6 ) - pen.x);
        pen.y = PGFT_ROUND(( y << 6 ) - pen.y);
    }


    /******************************************************
     * Draw text
     ******************************************************/

    for (n = 0; n < text->length; ++n)
    {
        FT_Glyph image;
        FT_BBox bbox;

        FontGlyph *glyph = &(text->glyphs[n]);

        if (!glyph->image)
            continue;

        /* copy image */
        error = FT_Glyph_Copy(glyph->image, &image);
        if (error)
            continue;

        if (image->format != FT_GLYPH_FORMAT_BITMAP)
        {
            if (render->vertical)
                error = FT_Glyph_Transform(image, NULL, &glyph->vvector);

            if (!error)
                error = FT_Glyph_Transform(image, render->matrix, &pen);

            if (error)
            {
                FT_Done_Glyph(image);
                continue;
            }
        }
        else
        {
            FT_BitmapGlyph  bitmap = (FT_BitmapGlyph)image;

            if (render->vertical)
            {
                bitmap->left += (glyph->vvector.x + pen.x) >> 6;
                bitmap->top  += (glyph->vvector.x + pen.y) >> 6;
            }
            else
            {
                bitmap->left += pen.x >> 6;
                bitmap->top  += pen.y >> 6;
            }
        }

        if (render->matrix)
            FT_Vector_Transform(advances + n, render->matrix);

        pen.x += advances[n].x;
        pen.y += advances[n].y;

        FT_Glyph_Get_CBox(image, FT_GLYPH_BBOX_PIXELS, &bbox);

        if (bbox.xMax > 0 && bbox.yMax > 0 &&
            bbox.xMin < surface->width &&
            bbox.yMin < surface->height)
        {
            int         left, top;
            FT_Bitmap*  source;
            FT_BitmapGlyph  bitmap;

            if (image->format == FT_GLYPH_FORMAT_OUTLINE)
            {
                FT_Render_Mode render_mode = 
                    render->antialias ?
                    FT_RENDER_MODE_NORMAL :
                    FT_RENDER_MODE_MONO;

                /* render the glyph to a bitmap, don't destroy original */
                error = FT_Glyph_To_Bitmap(&image, render_mode, NULL, 0);
                if (error)
                    return error;
            }

            if (image->format != FT_GLYPH_FORMAT_BITMAP)
                fprintf(stderr, "Format is not bitmap!\n");

            bitmap = (FT_BitmapGlyph)image;
            source = &bitmap->bitmap;

            left = bitmap->left;
            top = bitmap->top;

            top = surface->height - top;

            surface->render(left, top, surface, source, fg_color);
        }

        FT_Done_Glyph(image);
    } /* END OF RENDERING LOOP */

    return error;
}

int _PGFT_Render_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    const FT_UInt16 *text, int font_size, PyColor *fg_color, FontSurface *surface)
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

    _PGFT_BuildScaler(font, &scale, font_size);
    face = _PGFT_GetFace(ft, font);

    if (!face)
    {
        _PGFT_SetError(ft, "Failed to cache font face", 0);
        return -1;
    }

    use_kerning = FT_HAS_KERNING(face);
    prev_index = 0;

    /* FIXME: Some way to set the system's default ? */
    swapped = 0;

    pen_x = 0;
    pen_y = surface->glyph_height;

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
            const int left = pen_x + ((FT_BitmapGlyph)glyph)->left;
            const int top = pen_y - ((FT_BitmapGlyph)glyph)->top;

            surface->render(left, top, surface, bitmap, fg_color);
        }

        /* FIXME: Why the extra pixel? */
        pen_x += x_advance; /* + 1; */
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
    inst->_error_msg = calloc(1024, sizeof(char));
    
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
