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
    FT_BitmapGlyph image;

    FT_Face     face;

    FT_Vector   pen = {0, 0};                /* untransformed origin  */
    FT_Vector   pen1 = {0, 0};
    FT_Vector   pen2;

    FT_Vector   *next_posn;

    int         vertical = font->vertical;
    int         use_kerning = 0;
    FT_Angle    angle = render->rotation_angle;
    FT_Vector   kerning;
    FT_UInt     prev_glyph_index = 0;

    FT_Pos      min_x = PGFT_MAX_6;  /* 26.6 */
    FT_Pos      max_x = PGFT_MIN_6;  /* 26.6 */
    FT_Pos      min_y = PGFT_MAX_6;  /* 26.6 */
    FT_Pos      max_y = PGFT_MIN_6;  /* 26.6 */
    int         glyph_width;
    int         glyph_height;

    FT_Error    error = 0;
    int         i;

    /* load our sized face */
    face = _PGFT_GetFaceSized(ft, font, render->pt_size);

    if (!face)
    {
        RAISE(PyExc_SDLError, PGFT_GetError(ft));
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

    next_posn = ftext->posns;

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
                RAISE(PyExc_SDLError, PGFT_GetError(ft));
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

	glyph_width = image->bitmap.width;
	glyph_height = image->bitmap.rows;
        prev_glyph_index = glyph->glyph_index;
	if (vertical)
        {
            if (pen.x + glyph->v_bearings.x < min_x)
                min_x = pen.x + glyph->v_bearings.x;
            if (min_x + PGFT_INT_TO_6(glyph_width) > max_x)
                max_x = min_x + PGFT_INT_TO_6(glyph_width);
            if (pen.y + glyph->v_bearings.y > max_y)
                max_y = pen.y + glyph->v_bearings.y;
            if (max_y - PGFT_INT_TO_6(glyph_height) < min_y)
                min_y = max_y - PGFT_INT_TO_6(glyph_height);
            next_posn->x = pen.x + glyph->v_bearings.x;
            next_posn->y = pen.y + glyph->v_bearings.y;
            pen.x += glyph->v_advances.x;
            pen.y += glyph->v_advances.y;
        }
        else
        {
            if (pen.x + glyph->h_bearings.x < min_x)
                min_x = pen.x + glyph->h_bearings.x;
            if (min_x + PGFT_INT_TO_6(glyph_width) > max_x)
                max_x = min_x + PGFT_INT_TO_6(glyph_width);
            if (pen.y + glyph->h_bearings.y > max_y)
                max_y = pen.y + glyph->h_bearings.y;
            if (max_y - PGFT_INT_TO_6(glyph_height) < min_y)
                min_y = max_y - PGFT_INT_TO_6(glyph_height);
            next_posn->x = pen.x + glyph->h_bearings.x;
            next_posn->y = pen.y + glyph->h_bearings.y;
            pen.x += glyph->h_advances.x;
            pen.y += glyph->h_advances.y;
        }
        ++next_posn;
        *glyph_array++ = glyph;
    }
    if (pen.x > max_x)
        max_x = pen.x;
    else if (pen.x < min_x)
        min_x = pen.x;
    if (pen.y > max_y)
        max_y = pen.y;
    else if (pen.y < min_y)
        min_y = pen.y;

    if (render->style & FT_STYLE_UNDERLINE && !vertical && angle == 0)
    {
        FT_Fixed scale;
        FT_Fixed underline_pos;
        FT_Fixed underline_size;
        FT_Fixed min_y_underline;
        
        scale = face->size->metrics.y_scale;

        underline_pos = FT_MulFix(face->underline_position, scale) / 4; /*(1)*/
        underline_size = FT_MulFix(face->underline_thickness, scale) + bold_str;
        min_y_underline = (underline_pos -
                           PGFT_CEIL(ftext->underline_size / 2));
	if (min_y_underline < min_y)
        {
            min_y = min_y_underline;
        }

	ftext->underline_pos = (max_y - PGFT_FLOOR(underline_pos) -
                                PGFT_CEIL(ftext->underline_size / 2));
	ftext->underline_size = PGFT_CEIL(underline_size);

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

    if (min_x < 0)
    {
        ftext->width = PGFT_TRUNC(PGFT_CEIL(max_x) - PGFT_FLOOR(min_x));
        ftext->left = -PGFT_TRUNC(PGFT_FLOOR(min_x));
    }
    else
    {
        ftext->width = PGFT_TRUNC(PGFT_CEIL(max_x) + PGFT_FLOOR(min_x));
        ftext->left = 0;
    }
    ftext->height = PGFT_TRUNC(PGFT_CEIL(max_y) - PGFT_FLOOR(min_y));
    ftext->top = max_y;

    glyph_array = ftext->glyphs;
    next_posn = ftext->posns;
    if (vertical)
    {
        for (i = 0; i < string_length; ++i)
        {
            next_posn->x -= min_x;
            ++next_posn;
        }
    }
    else
    {
        for (i = 0; i < string_length; ++i)
        {
            next_posn->y = max_y - next_posn->y;
            ++next_posn;
        } 
    }

    return ftext;
}
