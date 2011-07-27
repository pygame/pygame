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

#define FX6_ONE 64
#define FX16_ONE 65536

#define SLANT_FACTOR    0.22
static FT_Matrix PGFT_SlantMatrix = 
{
    FX16_ONE,  (FT_Fixed)(SLANT_FACTOR * FX16_ONE),
    0,         FX16_ONE 
};

static FT_Matrix PGFT_Unit =
{
    FX16_ONE,  0,
    0,         FX16_ONE 
};

typedef struct __fonttextcontext
{
    FT_Library lib;
    FTC_FaceID id;
    FT_Face face;
    FTC_CMapCache charmap;
    int do_transform;
    FT_Matrix transform;
} FontTextContext;

#define BOLD_STRENGTH_D (0.65)
#define PIXEL_SIZE ((FT_Fixed)64)
#define BOLD_STRENGTH ((FT_Fixed)(BOLD_STRENGTH_D * PIXEL_SIZE))
#define BOLD_ADVANCE (BOLD_STRENGTH * (FT_Fixed)4)
#define UNICODE_SPACE ((PGFT_char)' ')

static FT_UInt32 GetLoadFlags(const FontRenderMode *);
static void fill_metrics(FontMetrics *metrics,
                         FT_Pos bearing_x, FT_Pos bearing_y,
                         FT_Vector *bearing_rotated,
                         FT_Vector *advance_rotated);
static void fill_context(FontTextContext *context,
                         const FreeTypeInstance *ft,
                         const PyFreeTypeFont *font,
                         const FontRenderMode *render,
                         const FT_Face face);

int
PGFT_FontTextInit(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    FontText *ftext = &(PGFT_INTERNALS(font)->active_text);

    ftext->buffer_size = 0;
    ftext->glyphs = NULL;
    ftext->posns = NULL;

    if (PGFT_Cache_Init(ft, &ftext->glyph_cache))
    {
        PyErr_NoMemory();
        return -1;
    }

    return 0;
}

void
PGFT_FontTextFree(PyFreeTypeFont *font)
{
    FontText *ftext = &(PGFT_INTERNALS(font)->active_text);

    if (ftext->buffer_size > 0)
    {
        _PGFT_free(ftext->glyphs);
        _PGFT_free(ftext->posns);
    }
    PGFT_Cache_Destroy(&ftext->glyph_cache);
}

FontText *
PGFT_LoadFontText(FreeTypeInstance *ft, PyFreeTypeFont *font, 
                  const FontRenderMode *render, PGFT_String *text)
{
    Py_ssize_t  string_length = PGFT_String_GET_LENGTH(text);

    PGFT_char * buffer = PGFT_String_GET_DATA(text);
    PGFT_char * buffer_end;
    PGFT_char * ch;

    FontText    *ftext = &(PGFT_INTERNALS(font)->active_text);
    FontGlyph   *glyph = NULL;
    FontGlyph   **glyph_array = NULL;
    FontMetrics *metrics;
    FT_BitmapGlyph image;
    FontTextContext context;

    FT_Face     face;

    FT_Vector   pen = {0, 0};                /* untransformed origin  */
    FT_Vector   pen1 = {0, 0};
    FT_Vector   pen2;

    FT_Vector   *next_pos;

    int         vertical = render->render_flags & FT_RFLAG_VERTICAL;
    int         use_kerning = font->kerning;
    int         pad = render->render_flags & FT_RFLAG_PAD;
    FT_UInt     prev_glyph_index = 0;

    /* All these are 16.16 precision */
    FT_Angle    rotation_angle = render->rotation_angle;

    /* All these are 26.6 precision */
    FT_Vector   kerning;
    FT_Pos      min_x = PGFT_MAX_6;
    FT_Pos      max_x = PGFT_MIN_6;
    FT_Pos      min_y = PGFT_MAX_6;
    FT_Pos      max_y = PGFT_MIN_6;
    FT_Pos      glyph_width;
    FT_Pos      glyph_height;
    FT_Pos      text_width;
    FT_Pos      text_height;
    FT_Pos      top = PGFT_MIN_6;
    FT_Fixed    bold_str = render->style & FT_STYLE_BOLD ? BOLD_STRENGTH : 0;

    FT_Error    error = 0;

    /* load our sized face */
    face = _PGFT_GetFaceSized(ft, font, render->pt_size);

    if (!face)
    {
        PyErr_SetString(PyExc_SDLError, PGFT_GetError(ft));
        return NULL;
    }

    /* cleanup the cache */
    PGFT_Cache_Cleanup(&ftext->glyph_cache);

    /* create the text struct */
    if (string_length > ftext->buffer_size)
    {
        _PGFT_free(ftext->glyphs);
        ftext->glyphs = (FontGlyph **)
            _PGFT_malloc((size_t)string_length * sizeof(FontGlyph *));
        if (!ftext->glyphs)
        {
            PyErr_NoMemory();
            return NULL;
        }

        _PGFT_free(ftext->posns);
        ftext->posns = (FT_Vector *)
        _PGFT_malloc((size_t)string_length * sizeof(FT_Vector));
        if (!ftext->posns)
        {
            PyErr_NoMemory();
            return NULL;
        }
        ftext->buffer_size = string_length;
    }
    ftext->length = string_length;
    ftext->underline_pos = ftext->underline_size = 0;
    ftext->descender = face->size->metrics.descender;

    /* fill it with the glyphs */
    fill_context(&context, ft, font, render, face);
    glyph_array = ftext->glyphs;
    next_pos = ftext->posns;

    for (ch = buffer, buffer_end = ch + string_length; ch < buffer_end; ++ch)
    {
        pen2.x = pen1.x;
        pen2.y = pen1.y;
        pen1.x = pen.x;
        pen1.y = pen.y;
        /*
         * Load the corresponding glyph from the cache
         */
        glyph = PGFT_Cache_FindGlyph(*((FT_UInt32 *)ch), render,
                                     &ftext->glyph_cache, &context);

        if (!glyph)
        {
            --ftext->length;
            continue;
        }
        image = glyph->image;
        glyph_width = glyph->width;
        glyph_height = glyph->height;

        /*
         * Do size calculations for all the glyphs in the text
         */
        if (use_kerning && prev_glyph_index)
        {
            error = FT_Get_Kerning(face, prev_glyph_index,
                                   glyph->glyph_index,
                                   FT_KERNING_UNFITTED, &kerning);
            if (error)
            {
                _PGFT_SetError(ft, "Loading glyphs", error);
                PyErr_SetString(PyExc_SDLError, PGFT_GetError(ft));
                return NULL;
            }
            if (rotation_angle != 0)
            {
                FT_Vector_Rotate(&kerning, rotation_angle);
            }
            pen.x += PGFT_ROUND(kerning.x);
            pen.y += PGFT_ROUND(kerning.y);
            if (FT_Vector_Length(&pen2) > FT_Vector_Length(&pen))
            {
                pen.x = pen2.x;
                pen.y = pen2.y;
            }
        }

        prev_glyph_index = glyph->glyph_index;
	metrics = vertical ? &glyph->v_metrics : &glyph->h_metrics;
	if (metrics->bearing_rotated.y > top)
        {
            top = metrics->bearing_rotated.y;
        }
        if (pen.x + metrics->bearing_rotated.x < min_x)
        {
            min_x = pen.x + metrics->bearing_rotated.x;
        }
        if (pen.x + metrics->bearing_rotated.x + glyph_width > max_x)
        {
            max_x = pen.x + metrics->bearing_rotated.x + glyph_width;
        }
        next_pos->x = pen.x + metrics->bearing_rotated.x;
        pen.x += metrics->advance_rotated.x;
        if (vertical)
        {
            if (pen.y + metrics->bearing_rotated.y < min_y)
        {
                min_y = pen.y + metrics->bearing_rotated.y;
        }
            if (pen.y + metrics->bearing_rotated.y + glyph_height > max_y)
        {
                max_y = pen.y + metrics->bearing_rotated.y + glyph_height;
        }
            next_pos->y = pen.y + metrics->bearing_rotated.y;
            pen.y += metrics->advance_rotated.y;
        }
        else
        {
            if (pen.y - metrics->bearing_rotated.y < min_y)
        {
                min_y = pen.y - metrics->bearing_rotated.y;
        }
            if (pen.y - metrics->bearing_rotated.y + glyph_height > max_y)
        {
                max_y = pen.y - metrics->bearing_rotated.y + glyph_height;
        }
            next_pos->y = pen.y - metrics->bearing_rotated.y;
            pen.y -= metrics->advance_rotated.y;
        }
        *glyph_array++ = glyph;
        ++next_pos;
    }

    if (pad && rotation_angle == 0)
    {
        FT_Size_Metrics *sz_metrics = &face->size->metrics;

        if (pen.x > max_x)
            max_x = pen.x;
        else if (pen.x < min_x)
            min_x = pen.x;
        if (pen.y > max_y)
            max_y = pen.y;
        else if (pen.y < min_y)
            min_y = pen.y;
        if (vertical)
        {
            FT_Fixed right = sz_metrics->max_advance / 2;
            
            if (max_x < right)
                max_x = right;
            if (min_x > -right)
                min_x = -right;
            if (min_y > 0)
                min_y = 0;
            else if (max_y < pen.y)
                max_y = pen.y;
        }
        else
        {
            FT_Fixed ascender = sz_metrics->ascender;
            FT_Fixed descender = sz_metrics->descender;

            if (min_x > 0)
                min_x = 0;
            if (max_x < pen.x)
                max_x = pen.x;
            if (min_y > -ascender)
                min_y = -ascender;
            if (max_y <= -descender)
                max_y = -descender + /* underscore allowance */ FX6_ONE;
            else
            {
                ftext->descender = -max_y;
                max_y += FX6_ONE;
            }
        }
    }
    else if (render->style & FT_STYLE_UNDERSCORE && !vertical &&
             rotation_angle == 0)
    {
        if (-ftext->descender >= max_y)
            max_y = -ftext->descender + /* underscore allowance */ FX6_ONE;
        else
            ftext->descender = -max_y;
    }

    if (render->style & FT_STYLE_UNDERLINE && !vertical && rotation_angle == 0)
    {
        FT_Fixed scale = face->size->metrics.y_scale;
        FT_Fixed underline_pos;
        FT_Fixed underline_size;
        FT_Fixed max_y_underline;
        
        underline_pos = -FT_MulFix(face->underline_position, scale) / 4; /*(1)*/
        underline_size = FT_MulFix(face->underline_thickness, scale) + bold_str;
        max_y_underline = underline_pos + underline_size / 2;
        if (max_y_underline > max_y)
        {
            max_y = max_y_underline;
        }
        ftext->underline_pos = underline_pos;
        ftext->underline_size = underline_size;

        /*
         * (1) HACK HACK HACK
         *
         * According to the FT documentation, 'underline_pos' is the offset 
         * to draw the underline in 26.6 FP, based on the text's baseline 
         * (negative values mean below the baseline).
         *
         * However, after scaling the underline position, the values for all
         * fonts are WAY off (e.g. fonts with 32pt size get underline offsets
         * of -14 pixels).
         *
         * Dividing the offset by 4, somehow, returns very sane results for
         * all kind of fonts; the underline seems to fit perfectly between
         * the baseline and bottom of the glyphs.
         *
         * We'll leave it like this until we can figure out what's wrong
         * with it...
         *
         */
    }

    text_width = PGFT_CEIL(max_x) - PGFT_FLOOR(min_x);
    ftext->width = PGFT_TRUNC(text_width);
    ftext->offset.x = -min_x;
    ftext->advance.x = pen.x;
    ftext->left = PGFT_TRUNC(PGFT_FLOOR(min_x));
    text_height = PGFT_CEIL(max_y) - PGFT_FLOOR(min_y);
    ftext->height = PGFT_TRUNC(text_height);
    ftext->offset.y = -min_y;
    ftext->advance.y = pen.y;
    ftext->top = PGFT_TRUNC(PGFT_CEIL(top));
    
    return ftext;
}

int PGFT_GetMetrics(FreeTypeInstance *ft, PyFreeTypeFont *font,
                    PGFT_char character, const FontRenderMode *render,
                    FT_UInt *gindex, long *minx, long *maxx,
                    long *miny, long *maxy,
                    double *advance_x, double *advance_y)
{ 
    FontText    *ftext = &(PGFT_INTERNALS(font)->active_text);
    FontGlyph *glyph = NULL;
    FontTextContext context;
    FT_Face     face;

    /* load our sized face */
    face = _PGFT_GetFaceSized(ft, font, render->pt_size);

    if (!face)
    {
        return -1;
    }

    /* cleanup the cache */
    PGFT_Cache_Cleanup(&ftext->glyph_cache);

    fill_context(&context, ft, font, render, face);
    glyph = PGFT_Cache_FindGlyph(character, render,
                                 &PGFT_INTERNALS(font)->active_text.glyph_cache, 
                                 &context);
    
    if (!glyph)
    {
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
PGFT_GetSurfaceSize(FreeTypeInstance *ft, PyFreeTypeFont *font,
        const FontRenderMode *render, FontText *text, 
        int *width, int *height)
{
    *width = text->width;
    *height = text->height;
    return 0;
}

int
PGFT_GetTopLeft(FontText *text, int *top, int *left)
{
    *top = text->top;
    *left = text->left;
    return 0;
}

int
PGFT_GetTextRect(FreeTypeInstance *ft, PyFreeTypeFont *font,
    const FontRenderMode *render, PGFT_String *text, SDL_Rect *r)
{
    FontText *font_text;

    font_text = PGFT_LoadFontText(ft, font, render, text);

    if (!font_text)
        return -1;

    r->x = -(Sint16)PGFT_TRUNC(PGFT_FLOOR(font_text->offset.x));
    r->y = (Sint16)PGFT_TRUNC(PGFT_CEIL(font_text->offset.y));
    r->w = (Uint16)font_text->width;
    r->h = (Uint16)font_text->height;
    return 0;
}

int
PGFT_LoadGlyph(FontGlyph *glyph, PGFT_char character, const FontRenderMode *render,
               void *internal)
{
    static FT_Vector delta = {0, 0};

    int embolden = render->style & FT_STYLE_BOLD;
    FT_Render_Mode rmode = (render->render_flags & FT_RFLAG_ANTIALIAS ?
                            FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO);
    FT_Fixed bold_str = 0;
    FT_Vector bold_delta = {0, 0};
    FT_Glyph image = NULL;

    FT_Glyph_Metrics *ft_metrics;
    FontTextContext *context = (FontTextContext *)internal;

    FT_UInt32 load_flags;
    FT_UInt gindex;

    FT_Fixed rotation_angle = render->rotation_angle;
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
    load_flags = GetLoadFlags(render);

    /*
     * Load the glyph into the glyph slot
     */
    if (FT_Load_Glyph(context->face, glyph->glyph_index, (FT_Int)load_flags) ||
        FT_Get_Glyph(context->face->glyph, &image))
        goto cleanup;

    /*
     * Perform any outline transformations
     */
    if (embolden)
    {
        FT_BBox before;
        FT_BBox after;

        bold_str = BOLD_STRENGTH;
        FT_Outline_Get_CBox(&((FT_OutlineGlyph)image)->outline, &before);
        if (FT_Outline_Embolden(&((FT_OutlineGlyph)image)->outline, bold_str))
            goto cleanup;
        FT_Outline_Get_CBox(&((FT_OutlineGlyph)image)->outline, &after);
        bold_delta.x += (after.xMax - after.xMin) - (before.xMax - before.xMin);
        bold_delta.y += (after.yMax - after.yMin) - (before.yMax - before.yMin);
    }

    if (context->do_transform)
    {
        if (FT_Glyph_Transform(image, &context->transform, &delta))
        {
            goto cleanup;
        }
    }

    /*
     * Finished with outline transformations, now replace with a bitmap
     */
    error = FT_Glyph_To_Bitmap(&image, rmode, 0, 1);
    if (error)
    {
        goto cleanup;
    }

    /* if (wide) */
    /* { */
    /*     int w = ((FT_BitmapGlyph)image)->bitmap.width; */

    /*     if (w) */
    /*     { */
    /*         error = FT_Bitmap_Embolden(context->lib, */
    /*                                    &((FT_BitmapGlyph)image)->bitmap, */
    /*                                    (FT_Pos)0x80, (FT_Pos)0); */
    /*         if (error) */
    /*         { */
    /*             goto cleanup; */
    /*         } */
    /*         bold_delta.x += PGFT_INT_TO_6(((FT_BitmapGlyph)image)->bitmap.width */
    /*                                       - w); */
    /*     } */
    /*     else */
    /*     { */
    /*         bold_delta.x += BOLD_ADVANCE; */
    /*     } */
    /* } */

    /* Fill the glyph */
    ft_metrics = &context->face->glyph->metrics;

    h_advance_rotated.x = ft_metrics->horiAdvance + bold_delta.x;
    h_advance_rotated.y = 0;
    v_advance_rotated.x = 0;
    v_advance_rotated.y = ft_metrics->vertAdvance + bold_delta.y;
    if (rotation_angle != 0)
    {
        FT_Angle counter_rotation = PGFT_INT_TO_6(360) - rotation_angle;

        FT_Vector_Rotate(&h_advance_rotated, rotation_angle);
        FT_Vector_Rotate(&v_advance_rotated, counter_rotation);
    }

    glyph->image = (FT_BitmapGlyph)image;
    glyph->width = PGFT_INT_TO_6(glyph->image->bitmap.width);
    glyph->height = PGFT_INT_TO_6(glyph->image->bitmap.rows);
    glyph->bold_strength = bold_str;
    h_bearing_rotated.x = PGFT_INT_TO_6(glyph->image->left);
    h_bearing_rotated.y = PGFT_INT_TO_6(glyph->image->top);
    fill_metrics(&glyph->h_metrics,
                 ft_metrics->horiBearingX,
                 ft_metrics->horiBearingY,
                 &h_bearing_rotated, &h_advance_rotated);

    if (rotation_angle == 0)
    {
        v_bearing_rotated.x = ft_metrics->vertBearingX - bold_delta.x / 2;
        v_bearing_rotated.y = ft_metrics->vertBearingY;
    }
    else
    {
        /*
         * Adjust the vertical metrics.
         */
        FT_Vector v_origin;

        v_origin.x = (glyph->h_metrics.bearing_x -
                      ft_metrics->vertBearingX + bold_delta.x / 2);
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
    if (image)
        FT_Done_Glyph(image);

    return -1;
}

static void
fill_context(FontTextContext *context,
             const FreeTypeInstance *ft,
             const PyFreeTypeFont *font,
             const FontRenderMode *render,
             const FT_Face face)
{
    context->lib = ft->library;
    context->id = (FTC_FaceID)&(font->id);
    context->face = face;
    context->charmap = ft->cache_charmap;
    context->do_transform = 0;

    if (render->style & FT_STYLE_OBLIQUE)
    {
        context->transform = PGFT_SlantMatrix;
        context->do_transform = 1;
    }
    else
    {
        context->transform = PGFT_Unit;
    }

    if (render->render_flags & FT_RFLAG_TRANSFORM)
    {
        FT_Matrix_Multiply(&render->transform, &context->transform);
        context->do_transform = 1;
    }

    if (render->rotation_angle != 0)
    {
        FT_Vector unit;
        FT_Matrix rotate;

        FT_Vector_Unit(&unit, render->rotation_angle);
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
GetLoadFlags(const FontRenderMode *render)
{
    FT_UInt32 load_flags = FT_LOAD_DEFAULT;

    load_flags |= FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH;

    if (render->render_flags & FT_RFLAG_AUTOHINT)
        load_flags |= FT_LOAD_FORCE_AUTOHINT;

    if (render->render_flags & FT_RFLAG_HINTED)
    {
        load_flags |= FT_LOAD_TARGET_NORMAL;
        /* load_flags |= ((render->render_flags & FT_RFLAG_ANTIALIAS) ? */
        /*                FT_LOAD_TARGET_NORMAL : */
        /*                FT_LOAD_TARGET_MONO); */
    }
    else
    {
        load_flags |= FT_LOAD_NO_HINTING;
    }

    return load_flags;
}
