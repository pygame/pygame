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
PGFT_GetTextSize_NEW(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    int pt_size, FontRenderMode *render, FontText *text, int *w, int *h)
{
    FT_Vector   extent, advances[MAX_GLYPHS];
    FT_Error    error;

    error = PGFT_GetTextAdvances(ft, font, pt_size, render, text, advances);

    if (error)
    {
        _PGFT_SetError(ft, "Failed to load glyph advances", error);
        return error;
    }

    extent = advances[text->length - 1];

    if (render->vertical)
    {
        *w = text->max_w;
        *h = PGFT_TRUNC(extent.y);
    }
    else
    {
        *w = PGFT_TRUNC(extent.x);
        *h = text->max_h;
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
