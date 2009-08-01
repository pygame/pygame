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

extern FT_Matrix PGFT_SlantMatrix;

/* Declarations */
void _PGFT_GetMetrics_INTERNAL(FT_Glyph, FT_UInt, int *, int *, int *, int *, int *);
int  _PGFT_GetTextSize_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font, 
        const FontRenderMode *render, FontText *text);


/* Real text metrics */
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
    int character, const FontRenderMode *render, int bbmode,
    void *minx, void *maxx, void *miny, void *maxy, void *advance)
{
    FontGlyph *     glyph = NULL;

    glyph = PGFT_Cache_FindGlyph(ft, &PGFT_INTERNALS(font)->cache, 
            (FT_UInt)character, render);

    if (!glyph)
        return -1;

    _PGFT_GetMetrics_INTERNAL(glyph->image, (FT_UInt)bbmode, 
            minx, maxx, miny, maxy, advance);

    if (bbmode == FT_BBOX_EXACT || bbmode == FT_BBOX_EXACT_GRIDFIT)
    {
#       define FP16_16(i)   ((float)((int)(i) / 65536.0f))
#       define FP26_6(i)    ((float)((int)(i) / 64.0f))

        *(float *)minx =    FP26_6(*(int *)minx);
        *(float *)miny =    FP26_6(*(int *)miny);
        *(float *)maxx =    FP26_6(*(int *)maxx);
        *(float *)maxy =    FP26_6(*(int *)maxy);
        *(float *)advance = FP16_16(*(int *)advance);

#       undef FP16_16
#       undef FP26_6
    }

    return 0;

}


int
_PGFT_GetTextSize_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    const FontRenderMode *render, FontText *text)
{
    FT_Vector   extent, advances[PGFT_MAX_GLYPHS];
    FT_Error    error;
    FT_Vector   size;

    error = PGFT_GetTextAdvances(ft, font, render, text, advances);

    if (error)
        return error;

    extent = advances[text->length - 1];

    if (render->render_flags & FT_RFLAG_VERTICAL)
    {
        size.x = text->glyph_size.x;
        size.y = ABS(extent.y);
    }
    else
    {
        size.x = extent.x;
        size.y = text->glyph_size.y;
    }

    if (render->rotation_angle)
    {   
        size.x = MAX(size.x, size.y);
        size.y = MAX(size.x, size.y);
    }

    text->text_size.x = ABS(size.x);
    text->text_size.y = ABS(size.y);

    return 0;
}

int
PGFT_GetSurfaceSize(FreeTypeInstance *ft, PyFreeTypeFont *font,
        const FontRenderMode *render, FontText *text, 
        int *width, int *height)
{
    int w, h;

    if (text == NULL || 
        _PGFT_GetTextSize_INTERNAL(ft, font, render, text) != 0)
        return -1;

    w = text->text_size.x;
    h = text->text_size.y;

    if (text->underline_size > 0)
    {
        h = MAX(h, text->underline_pos + text->underline_size);
    }

    if (render->style & FT_STYLE_ITALIC)
    {
        FT_Vector s = {w, h};

        FT_Vector_Transform(&s, &PGFT_SlantMatrix);
        w = s.x; h = s.y;
    }

    *width = PGFT_TRUNC(PGFT_CEIL(w));
    *height = PGFT_TRUNC(PGFT_CEIL(h));
    return 0;
}

int
PGFT_GetTextSize(FreeTypeInstance *ft, PyFreeTypeFont *font,
    const FontRenderMode *render, PyObject *text, int *w, int *h)
{
    FontText *font_text;

    font_text = PGFT_LoadFontText(ft, font, render, text);

    if (!font_text)
        return -1;

    return PGFT_GetSurfaceSize(ft, font, render, font_text, w, h);
}
