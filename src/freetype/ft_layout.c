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

#include "ft_wrap.h"
#include FT_MODULE_H
#include FT_TRIGONOMETRY_H
#include FT_OUTLINE_H
#include FT_BITMAP_H
#include FT_CACHE_H

/* Multiply the font's x ppem by this factor to get the x strength
 * factor in 16.16 Fixed.
 */
#define FX16_WIDE_FACTOR (FX16_ONE / 12)

#define SLANT_FACTOR    0.22
static FT_Matrix slant_matrix = {
    FX16_ONE,  (FT_Fixed)(SLANT_FACTOR * FX16_ONE),
    0,         FX16_ONE
};

static FT_Matrix unit_matrix = {
    FX16_ONE,  0,
    0,         FX16_ONE
};

typedef struct textcontext_ {
    FT_Library lib;
    FTC_FaceID id;
    FT_Face font;
    FTC_CMapCache charmap;
    int do_transform;
    FT_Matrix transform;
} TextContext;

#if 0
#define BOLD_STRENGTH_D (0.65)
#define PIXEL_SIZE ((FT_Fixed)64)
#define BOLD_STRENGTH ((FT_Fixed)(BOLD_STRENGTH_D * PIXEL_SIZE))
#define BOLD_ADVANCE (BOLD_STRENGTH * (FT_Fixed)4)
#endif
#define FX16_BOLD_FACTOR (FX16_ONE / 36)
#define UNICODE_SPACE ((PGFT_char)' ')

static FT_UInt32 get_load_flags(const FontRenderMode *);
static void fill_metrics(FontMetrics *, FT_Pos, FT_Pos,
                         FT_Vector *, FT_Vector *);
static void fill_context(TextContext *,
                         const FreeTypeInstance *,
                         const PgFontObject *,
                         const FontRenderMode *,
                         const FT_Face);

int
_PGFT_FontTextInit(FreeTypeInstance *ft, PgFontObject *fontobj)
{
    FontText *ftext = &(PGFT_INTERNALS(fontobj)->active_text);

    ftext->buffer_size = 0;
    ftext->glyphs = 0;
    ftext->posns = 0;

    if (_PGFT_Cache_Init(ft, &ftext->glyph_cache)) {
        PyErr_NoMemory();
        return -1;
    }

    return 0;
}

void
_PGFT_FontTextFree(PgFontObject *fontobj)
{
    FontText *ftext = &(PGFT_INTERNALS(fontobj)->active_text);

    if (ftext->buffer_size > 0) {
        _PGFT_free(ftext->glyphs);
        _PGFT_free(ftext->posns);
    }
    _PGFT_Cache_Destroy(&ftext->glyph_cache);
}

FontText *
_PGFT_LoadFontText(FreeTypeInstance *ft, PgFontObject *fontobj,
                   const FontRenderMode *mode, PGFT_String *text)
{
    Py_ssize_t  string_length = PGFT_String_GET_LENGTH(text);

    PGFT_char * buffer = PGFT_String_GET_DATA(text);
    PGFT_char * buffer_end;
    PGFT_char * ch;

    FontText    *ftext = &(PGFT_INTERNALS(fontobj)->active_text);
    FontGlyph   *glyph = 0;
    FontGlyph   **glyph_array = 0;
    FontMetrics *metrics;
    TextContext context;

    FT_Face     font;
    FT_Size_Metrics *sz_metrics;

    FT_Vector   pen = {0, 0};                /* untransformed origin  */
    FT_Vector   pen1 = {0, 0};
    FT_Vector   pen2;

    FT_Vector   *next_pos;

    int         vertical = mode->render_flags & FT_RFLAG_VERTICAL;
    int         use_kerning = mode->render_flags & FT_RFLAG_KERNING;
    int         pad = mode->render_flags & FT_RFLAG_PAD;
    FT_UInt     prev_glyph_index = 0;

    /* All these are 16.16 precision */
    FT_Angle    rotation_angle = mode->rotation_angle;

    /* All these are 26.6 precision */
    FT_Vector   kerning;
    FT_Pos      min_x = FX6_MAX;
    FT_Pos      max_x = FX6_MIN;
    FT_Pos      min_y = FX6_MAX;
    FT_Pos      max_y = FX6_MIN;
    FT_Pos      glyph_width;
    FT_Pos      glyph_height;
    FT_Pos      top = FX6_MIN;
    FT_Fixed    y_scale;

    FT_Error    error = 0;

    /* load our sized font */
    font = _PGFT_GetFontSized(ft, fontobj, mode->face_size);
    if (!font) {
        PyErr_SetString(PyExc_SDLError, _PGFT_GetError(ft));
        return 0;
    }
    sz_metrics = &font->size->metrics;
    y_scale = sz_metrics->y_scale;

    /* cleanup the cache */
    _PGFT_Cache_Cleanup(&ftext->glyph_cache);

    /* create the text struct */
    if (string_length > ftext->buffer_size) {
        _PGFT_free(ftext->glyphs);
        ftext->glyphs = (FontGlyph **)
            _PGFT_malloc((size_t)string_length * sizeof(FontGlyph *));
        if (!ftext->glyphs) {
            PyErr_NoMemory();
            return 0;
        }

        _PGFT_free(ftext->posns);
        ftext->posns = (FT_Vector *)
        _PGFT_malloc((size_t)string_length * sizeof(FT_Vector));
        if (!ftext->posns) {
            PyErr_NoMemory();
            return 0;
        }
        ftext->buffer_size = string_length;
    }
    ftext->length = string_length;
    ftext->ascender = sz_metrics->ascender;
    ftext->underline_pos = -FT_MulFix(font->underline_position, y_scale);
    ftext->underline_size = FT_MulFix(font->underline_thickness, y_scale);
    if (mode->style & FT_STYLE_STRONG) {
        FT_Fixed bold_str = mode->strength * sz_metrics->x_ppem;

        ftext->underline_size = FT_MulFix(ftext->underline_size,
                                          FX16_ONE + bold_str / 4);
    }

    /* fill it with the glyphs */
    fill_context(&context, ft, fontobj, mode, font);
    glyph_array = ftext->glyphs;
    next_pos = ftext->posns;

    for (ch = buffer, buffer_end = ch + string_length; ch < buffer_end; ++ch) {
        pen2.x = pen1.x;
        pen2.y = pen1.y;
        pen1.x = pen.x;
        pen1.y = pen.y;
        /*
         * Load the corresponding glyph from the cache
         */
        glyph = _PGFT_Cache_FindGlyph(*((FT_UInt32 *)ch), mode,
                                     &ftext->glyph_cache, &context);

        if (!glyph) {
            --ftext->length;
            continue;
        }
        glyph_width = glyph->width;
        glyph_height = glyph->height;

        /*
         * Do size calculations for all the glyphs in the text
         */
        if (use_kerning && prev_glyph_index) {
            error = FT_Get_Kerning(font, prev_glyph_index,
                                   glyph->glyph_index,
                                   FT_KERNING_UNFITTED, &kerning);
            if (error) {
                _PGFT_SetError(ft, "Loading glyphs", error);
                PyErr_SetString(PyExc_SDLError, _PGFT_GetError(ft));
                return 0;
            }
            if (rotation_angle != 0) {
                FT_Vector_Rotate(&kerning, rotation_angle);
            }
            pen.x += FX6_ROUND(kerning.x);
            pen.y += FX6_ROUND(kerning.y);
            if (FT_Vector_Length(&pen2) > FT_Vector_Length(&pen)) {
                pen.x = pen2.x;
                pen.y = pen2.y;
            }
        }

        prev_glyph_index = glyph->glyph_index;
        metrics = vertical ? &glyph->v_metrics : &glyph->h_metrics;
        if (metrics->bearing_rotated.y > top) {
            top = metrics->bearing_rotated.y;
        }
        if (pen.x + metrics->bearing_rotated.x < min_x) {
            min_x = pen.x + metrics->bearing_rotated.x;
        }
        if (pen.x + metrics->bearing_rotated.x + glyph_width > max_x) {
            max_x = pen.x + metrics->bearing_rotated.x + glyph_width;
        }
        next_pos->x = pen.x + metrics->bearing_rotated.x;
        pen.x += metrics->advance_rotated.x;
        if (vertical) {
            if (pen.y + metrics->bearing_rotated.y < min_y) {
                min_y = pen.y + metrics->bearing_rotated.y;
            }
            if (pen.y + metrics->bearing_rotated.y + glyph_height > max_y) {
                max_y = pen.y + metrics->bearing_rotated.y + glyph_height;
            }
            next_pos->y = pen.y + metrics->bearing_rotated.y;
            pen.y += metrics->advance_rotated.y;
        }
        else {
            if (pen.y - metrics->bearing_rotated.y < min_y) {
                min_y = pen.y - metrics->bearing_rotated.y;
            }
            if (pen.y - metrics->bearing_rotated.y + glyph_height > max_y) {
                max_y = pen.y - metrics->bearing_rotated.y + glyph_height;
            }
            next_pos->y = pen.y - metrics->bearing_rotated.y;
            pen.y -= metrics->advance_rotated.y;
        }
        *glyph_array++ = glyph;
        ++next_pos;
    }

    if (ftext->length == 0) {
        min_x = 0;
        max_x = 0;
        if (vertical) {
            ftext->min_y = 0;
            max_y = sz_metrics->height;
        }
        else {
            FT_Size_Metrics *sz_metrics = &font->size->metrics;

            min_y = -sz_metrics->ascender;
            max_y = -sz_metrics->descender;
        }
    }

    if (pad) {
        FT_Size_Metrics *sz_metrics = &font->size->metrics;

        if (pen.x > max_x) {
            max_x = pen.x;
        }
        else if (pen.x < min_x) {
            min_x = pen.x;
        }
        if (pen.y > max_y) {
            max_y = pen.y;
        }
        else if (pen.y < min_y) {
            min_y = pen.y;
        }
        if (vertical) {
            FT_Fixed right = sz_metrics->max_advance / 2;

            if (max_x < right) {
                max_x = right;
            }
            if (min_x > -right) {
                min_x = -right;
            }
            if (min_y > 0) {
                min_y = 0;
            }
            else if (max_y < pen.y) {
                max_y = pen.y;
            }
        }
        else {
            FT_Fixed ascender = sz_metrics->ascender;
            FT_Fixed descender = sz_metrics->descender;

            if (min_x > 0) {
                min_x = 0;
            }
            if (max_x < pen.x) {
                max_x = pen.x;
            }
            if (min_y > -ascender) {
                min_y = -ascender;
            }
            if (max_y <= -descender) {
                max_y = -descender + /* baseline */ FX6_ONE;
            }
        }
    }

    ftext->left = FX6_TRUNC(FX6_FLOOR(min_x));
    ftext->top = FX6_TRUNC(FX6_CEIL(top));
    ftext->min_x = min_x;
    ftext->max_x = max_x;
    ftext->min_y = min_y;
    ftext->max_y = max_y;
    ftext->advance.x = pen.x;
    ftext->advance.y = pen.y;

    return ftext;
}

int _PGFT_GetMetrics(FreeTypeInstance *ft, PgFontObject *fontobj,
                    PGFT_char character, const FontRenderMode *mode,
                    FT_UInt *gindex, long *minx, long *maxx,
                    long *miny, long *maxy,
                    double *advance_x, double *advance_y)
{
    FontText    *ftext = &(PGFT_INTERNALS(fontobj)->active_text);
    FontGlyph *glyph = 0;
    TextContext context;
    FT_Face     font;

    /* load our sized font */
    font = _PGFT_GetFontSized(ft, fontobj, mode->face_size);
    if (!font) {
        return -1;
    }

    /* cleanup the cache */
    _PGFT_Cache_Cleanup(&ftext->glyph_cache);

    fill_context(&context, ft, fontobj, mode, font);
    glyph = _PGFT_Cache_FindGlyph(character, mode,
                                 &PGFT_INTERNALS(fontobj)->active_text.glyph_cache,
                                 &context);

    if (!glyph) {
        return -1;
    }

    *gindex = glyph->glyph_index;
    *minx = (long)glyph->image->left;
    *maxx = (long)(glyph->image->left + glyph->image->bitmap.width);
    *maxy = (long)glyph->image->top;
    *miny = (long)(glyph->image->top - glyph->image->bitmap.rows);
    *advance_x = (double)(glyph->h_metrics.advance_rotated.x / 64.0);
    *advance_y = (double)(glyph->h_metrics.advance_rotated.y / 64.0);

    return 0;
}

int
_PGFT_LoadGlyph(FontGlyph *glyph, PGFT_char character,
                const FontRenderMode *mode, void *internal)
{
    static FT_Vector delta = {0, 0};

    FT_Render_Mode rmode = (mode->render_flags & FT_RFLAG_ANTIALIAS ?
                            FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO);
    FT_Vector strong_delta = {0, 0};
    FT_Glyph image = 0;

    FT_Glyph_Metrics *ft_metrics;
    TextContext *context = (TextContext *)internal;

    FT_UInt32 load_flags;
    FT_UInt gindex;

    FT_Fixed rotation_angle = mode->rotation_angle;
    /* FT_Matrix transform; */
    FT_Vector h_bearing_rotated;
    FT_Vector v_bearing_rotated;
    FT_Vector h_advance_rotated;
    FT_Vector v_advance_rotated;

    FT_Error error = 0;

    /*
     * Calculate the corresponding glyph index for the char
     */
    gindex = FTC_CMapCache_Lookup(context->charmap, context->id,
                                  -1, (FT_UInt32)character);

    glyph->glyph_index = gindex;

    /*
     * Get loading information
     */
    load_flags = get_load_flags(mode);

    /*
     * Load the glyph into the glyph slot
     */
    if (FT_Load_Glyph(context->font, glyph->glyph_index, (FT_Int)load_flags) ||
        FT_Get_Glyph(context->font->glyph, &image))
        goto cleanup;

    /*
     * Perform any outline transformations
     */
    if (mode->style & FT_STYLE_STRONG) {
        FT_UShort x_ppem = context->font->size->metrics.x_ppem;
        FT_Fixed bold_str;
        FT_BBox before;
        FT_BBox after;

        bold_str = FX16_CEIL_TO_FX6(mode->strength * x_ppem);
        FT_Outline_Get_CBox(&((FT_OutlineGlyph)image)->outline, &before);
        if (FT_Outline_Embolden(&((FT_OutlineGlyph)image)->outline, bold_str))
            goto cleanup;
        FT_Outline_Get_CBox(&((FT_OutlineGlyph)image)->outline, &after);
        strong_delta.x += ((after.xMax - after.xMin) -
                           (before.xMax - before.xMin));
        strong_delta.y += ((after.yMax - after.yMin) -
                           (before.yMax - before.yMin));
    }

    if (context->do_transform) {
        if (FT_Glyph_Transform(image, &context->transform, &delta)) {
            goto cleanup;
        }
    }

    /*
     * Finished with outline transformations, now replace with a bitmap
     */
    error = FT_Glyph_To_Bitmap(&image, rmode, 0, 1);
    if (error) {
        goto cleanup;
    }

    if (mode->style & FT_STYLE_WIDE) {
        FT_Bitmap *bitmap = &((FT_BitmapGlyph)image)->bitmap;
        int w = bitmap->width;
        FT_UShort x_ppem = context->font->size->metrics.x_ppem;
        FT_Pos x_strength;

        x_strength = FX16_CEIL_TO_FX6(mode->strength * x_ppem);

        /* FT_Bitmap_Embolden returns an error for a zero width bitmap */
        if (w > 0) {
            error = FT_Bitmap_Embolden(context->lib, bitmap,
                                       x_strength, (FT_Pos)0);
            if (error) {
                goto cleanup;
            }
            strong_delta.x += INT_TO_FX6(bitmap->width - w);
        }
        else {
            strong_delta.x += x_strength;
        }
    }

    /* Fill the glyph */
    ft_metrics = &context->font->glyph->metrics;

    h_advance_rotated.x = ft_metrics->horiAdvance + strong_delta.x;
    h_advance_rotated.y = 0;
    v_advance_rotated.x = 0;
    v_advance_rotated.y = ft_metrics->vertAdvance + strong_delta.y;
    if (rotation_angle != 0) {
        FT_Angle counter_rotation = INT_TO_FX6(360) - rotation_angle;

        FT_Vector_Rotate(&h_advance_rotated, rotation_angle);
        FT_Vector_Rotate(&v_advance_rotated, counter_rotation);
    }

    glyph->image = (FT_BitmapGlyph)image;
    glyph->width = INT_TO_FX6(glyph->image->bitmap.width);
    glyph->height = INT_TO_FX6(glyph->image->bitmap.rows);
    h_bearing_rotated.x = INT_TO_FX6(glyph->image->left);
    h_bearing_rotated.y = INT_TO_FX6(glyph->image->top);
    fill_metrics(&glyph->h_metrics,
                 ft_metrics->horiBearingX,
                 ft_metrics->horiBearingY,
                 &h_bearing_rotated, &h_advance_rotated);

    if (rotation_angle == 0) {
        v_bearing_rotated.x = ft_metrics->vertBearingX - strong_delta.x / 2;
        v_bearing_rotated.y = ft_metrics->vertBearingY;
    }
    else {
        /*
         * Adjust the vertical metrics.
         */
        FT_Vector v_origin;

        v_origin.x = (glyph->h_metrics.bearing_x -
                      ft_metrics->vertBearingX + strong_delta.x / 2);
        v_origin.y = (glyph->h_metrics.bearing_y +
                      ft_metrics->vertBearingY);
        FT_Vector_Rotate(&v_origin, rotation_angle);
        v_bearing_rotated.x = glyph->h_metrics.bearing_rotated.x - v_origin.x;
        v_bearing_rotated.y = v_origin.y - glyph->h_metrics.bearing_rotated.y;
    }
    fill_metrics(&glyph->v_metrics,
                 ft_metrics->vertBearingX,
                 ft_metrics->vertBearingY,
                 &v_bearing_rotated, &v_advance_rotated);

    return 0;

    /*
     * Cleanup on error
     */
cleanup:
    if (image) {
        FT_Done_Glyph(image);
    }

    return -1;
}

static void
fill_context(TextContext *context,
             const FreeTypeInstance *ft,
             const PgFontObject *fontobj,
             const FontRenderMode *mode,
             const FT_Face font)
{
    context->lib = ft->library;
    context->id = (FTC_FaceID)&(fontobj->id);
    context->font = font;
    context->charmap = ft->cache_charmap;
    context->do_transform = 0;

    if (mode->style & FT_STYLE_OBLIQUE) {
        context->transform = slant_matrix;
        context->do_transform = 1;
    }
    else {
        context->transform = unit_matrix;
    }

    if (mode->render_flags & FT_RFLAG_TRANSFORM) {
        FT_Matrix_Multiply(&mode->transform, &context->transform);
        context->do_transform = 1;
    }

    if (mode->rotation_angle != 0) {
        FT_Vector unit;
        FT_Matrix rotate;

        FT_Vector_Unit(&unit, mode->rotation_angle);
        rotate.xx = unit.x;  /*  cos(angle) */
        rotate.xy = -unit.y; /* -sin(angle) */
        rotate.yx = unit.y;  /*  sin(angle) */
        rotate.yy = unit.x;  /*  cos(angle) */
        FT_Matrix_Multiply(&rotate, &context->transform);
        context->do_transform = 1;
    }
}

static void
fill_metrics(FontMetrics *metrics,
             FT_Pos bearing_x, FT_Pos bearing_y,
             FT_Vector *bearing_rotated,
             FT_Vector *advance_rotated)
{
    metrics->bearing_x = bearing_x;
    metrics->bearing_y = bearing_y;
    metrics->bearing_rotated.x = bearing_rotated->x;
    metrics->bearing_rotated.y = bearing_rotated->y;
    metrics->advance_rotated.x = advance_rotated->x;
    metrics->advance_rotated.y = advance_rotated->y;
}

static FT_UInt32
get_load_flags(const FontRenderMode *mode)
{
    FT_UInt32 load_flags = FT_LOAD_DEFAULT;

    load_flags |= FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH;

    if (mode->render_flags & FT_RFLAG_AUTOHINT) {
        load_flags |= FT_LOAD_FORCE_AUTOHINT;
    }

    if (mode->render_flags & FT_RFLAG_HINTED) {
        load_flags |= FT_LOAD_TARGET_NORMAL;
    }
    else {
        load_flags |= FT_LOAD_NO_HINTING;
    }

    if (!(mode->render_flags & FT_RFLAG_USE_BITMAP_STRIKES) ||
        (mode->render_flags & FT_RFLAG_TRANSFORM) ||
        (mode->rotation_angle != 0) ||
        (mode->style & (FT_STYLE_STRONG | FT_STYLE_OBLIQUE))) {
        load_flags |= FT_LOAD_NO_BITMAP;
    }

    return load_flags;
}
