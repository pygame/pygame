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

#define SLANT_FACTOR    0.22
static FT_Matrix PGFT_SlantMatrix = 
{
    (1 << 16),  (FT_Fixed)(SLANT_FACTOR * (1 << 16)),
    0,          (1 << 16) 
};

static FT_UInt32 GetLoadFlags(const FontRenderMode *);
static void fill_metrics(FontMetrics *metrics,
                         FT_Pos bearing_x, FT_Pos bearing_y,
                         FT_Vector *bearing_rotated,
                         FT_Vector *advance_rotated);

FontText *
PGFT_LoadFontText(FreeTypeInstance *ft, PyFreeTypeFont *font, 
                  const FontRenderMode *render, PGFT_String *text)
{
    Py_ssize_t  string_length = PGFT_String_GET_LENGTH(text);

    PGFT_char * buffer = PGFT_String_GET_DATA(text);
    PGFT_char * buffer_end;
    PGFT_char * ch;

    FT_Fixed    y_scale;
    FT_Fixed    bold_str = 0;

    FontText    *ftext = NULL;
    FontGlyph   *glyph = NULL;
    FontGlyph   **glyph_array = NULL;
    FontMetrics *metrics;
    FT_BitmapGlyph image;

    FT_Face     face;

    FT_Vector   pen = {0, 0};                /* untransformed origin  */
    FT_Vector   pen1 = {0, 0};
    FT_Vector   pen2;

    FT_Vector   *next_pos;

    int         vertical = font->vertical;
    int         use_kerning = font->kerning;
    FT_UInt     prev_glyph_index = 0;

    /* All these are 16.16 precision */
    FT_Angle    angle = render->rotation_angle;

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

    FT_Error    error = 0;

    /* load our sized face */
    face = _PGFT_GetFaceSized(ft, font, render->pt_size);

    if (!face)
    {
        PyErr_SetString(PyExc_SDLError, PGFT_GetError(ft));
        return NULL;
    }

    /* cleanup the cache */
    PGFT_Cache_Cleanup(&PGFT_INTERNALS(font)->cache);

    /* create the text struct */
    ftext = &(PGFT_INTERNALS(font)->active_text);

    if (string_length > ftext->length)
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
    }

    ftext->length = string_length;
    ftext->underline_pos = ftext->underline_size = 0;

    y_scale = face->size->metrics.y_scale;

    /* fill it with the glyphs */
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
        glyph = PGFT_Cache_FindGlyph(ft, &PGFT_INTERNALS(font)->cache,
                                     *((FT_UInt32 *)ch), render);

        if (!glyph)
            continue;
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
            if (angle != 0)
            {
                FT_Vector_Rotate(&kerning, angle);
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
    if (pen.x > max_x)
        max_x = pen.x;
    if (pen.x < min_x)
        min_x = pen.x;
    if (pen.y > max_y)
        max_y = pen.y;
    if (pen.y < min_y)
        min_y = pen.y;

    if (render->style & FT_STYLE_UNDERLINE && !vertical && angle == 0)
    {
        FT_Fixed scale;
        FT_Fixed underline_pos;
        FT_Fixed underline_size;
        FT_Fixed max_y_underline;
        
        scale = face->size->metrics.y_scale;

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

int
PGFT_LoadGlyph(FontGlyph *glyph, FreeTypeInstance *ft, PyFreeTypeFont *font,
	       const FontRenderMode *render, PGFT_char character)
{
    static FT_Vector delta = {0, 0};

    int embolden = render->style & FT_STYLE_BOLD;
    FT_Glyph image = NULL;

    FT_Glyph_Metrics *ft_metrics;
    FT_Face face;

    FT_UInt32 load_flags;
    FT_Pos bold_str = 0;
    FT_Pos bold_advance = 0;
    FT_UInt gindex;

    FT_Fixed rotation_angle = render->rotation_angle;
    FT_Vector unit;
    FT_Matrix transform;
    FT_Vector h_bearing_rotated;
    FT_Vector v_bearing_rotated;
    FT_Vector h_advance_rotated;
    FT_Vector v_advance_rotated;

    FT_Error error = 0;

    /*
     * Grab face reference
     */
    face = _PGFT_GetFaceSized(ft, font, render->pt_size);

    if (!face)
    {
        _PGFT_SetError(ft, "Failed to resize face", 0);
        goto cleanup;
    }

    /*
     * Calculate the corresponding glyph index for the char
     */
    gindex = FTC_CMapCache_Lookup(ft->cache_charmap, 
                                  (FTC_FaceID)&(font->id), -1,
                                  (FT_UInt32)character);

    if (!gindex)
    {
        _PGFT_SetError(ft, "Glyph character not found in font", 0);
        goto cleanup;
    }
    glyph->glyph_index = gindex;

    /*
     * Get loading information
     */
    load_flags = GetLoadFlags(render);

    /*
     * Load the glyph into the glyph slot
     */
    if (FT_Load_Glyph(face, glyph->glyph_index, (FT_Int)load_flags) ||
        FT_Get_Glyph(face->glyph, &image))
        goto cleanup;

    if (embolden)
    {
        bold_str = PGFT_GetBoldStrength(face);
        /* bold_advance = (bold_str * 3) / 2; */
        bold_advance = 4 * bold_str;
        if (FT_Outline_Embolden(&((FT_OutlineGlyph)image)->outline, bold_str))
            goto cleanup;
    }

    /*
     * Collect useful metric values
     */
    ft_metrics = &face->glyph->metrics;
    h_advance_rotated.x = ft_metrics->horiAdvance + bold_advance;
    h_advance_rotated.y = 0;
    v_advance_rotated.x = 0;
    v_advance_rotated.y = ft_metrics->vertAdvance + bold_advance;

    /*
     * Perform any transformations
     */
    if (render->style & FT_STYLE_OBLIQUE)
    {
        FT_Outline_Transform(&(((FT_OutlineGlyph)image)->outline),
                             &PGFT_SlantMatrix);
    }

    if (rotation_angle != 0)
    {
        FT_Angle counter_rotation =
            rotation_angle ? PGFT_INT_TO_6(360) - rotation_angle : 0;

        FT_Vector_Unit(&unit, rotation_angle);
        transform.xx = unit.x;  /*  cos(angle) */
        transform.xy = -unit.y; /* -sin(angle) */
        transform.yx = unit.y;  /*  sin(angle) */
        transform.yy = unit.x;  /*  cos(angle) */
        if (FT_Glyph_Transform(image, &transform, &delta))
        {
            goto cleanup;
        }
        FT_Vector_Rotate(&h_advance_rotated, rotation_angle);
        FT_Vector_Rotate(&v_advance_rotated, counter_rotation);
    }

    /*
     * Finished with transformations, now replace with a bitmap
     */
    error = FT_Glyph_To_Bitmap(&image, FT_RENDER_MODE_NORMAL, 0, 1);
    if (error)
    {
        _PGFT_SetError(ft, "Rendering glyphs", error);
        RAISE(PyExc_SDLError, PGFT_GetError(ft));
        goto cleanup;
    }

    /* Fill the glyph */
    glyph->image = (FT_BitmapGlyph)image;
    glyph->width = PGFT_INT_TO_6(glyph->image->bitmap.width);
    glyph->height = PGFT_INT_TO_6(glyph->image->bitmap.rows);
    glyph->bold_strength = bold_str;
    h_bearing_rotated.x = PGFT_INT_TO_6(glyph->image->left);
    h_bearing_rotated.y = PGFT_INT_TO_6(glyph->image->top);
    fill_metrics(&glyph->h_metrics,
                 ft_metrics->horiBearingX + bold_advance,
                 ft_metrics->horiBearingY + bold_advance,
                 &h_bearing_rotated, &h_advance_rotated);

    if (rotation_angle == 0)
    {
        v_bearing_rotated.x = ft_metrics->vertBearingX - bold_advance / 2;
        v_bearing_rotated.y = ft_metrics->vertBearingY;
    }
    else
    {
        /*
         * Adjust the vertical metrics.
         */
        FT_Vector v_origin;

        v_origin.x = (glyph->h_metrics.bearing_x -
                      ft_metrics->vertBearingX + bold_advance / 2);
        v_origin.y = (glyph->h_metrics.bearing_y +
                      ft_metrics->vertBearingY);
        FT_Vector_Rotate(&v_origin, rotation_angle);
        v_bearing_rotated.x = glyph->h_metrics.bearing_rotated.x - v_origin.x;
        v_bearing_rotated.y = v_origin.y - glyph->h_metrics.bearing_rotated.y;
    }
    fill_metrics(&glyph->v_metrics,
                 ft_metrics->vertBearingX + bold_advance,
                 ft_metrics->vertBearingY + bold_advance,
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
        load_flags |= ((render->render_flags & FT_RFLAG_ANTIALIAS) ?
                       FT_LOAD_TARGET_NORMAL :
                       FT_LOAD_TARGET_MONO);
    }
    else
    {
        load_flags |= FT_LOAD_NO_HINTING;
    }

    return load_flags;
}
