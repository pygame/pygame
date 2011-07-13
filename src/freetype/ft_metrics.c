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

extern FT_Matrix PGFT_SlantMatrix;

int PGFT_GetMetrics(FreeTypeInstance *ft, PyFreeTypeFont *font,
                    PGFT_char character, const FontRenderMode *render,
                    long *minx, long *maxx, long *miny, long *maxy,
                    double *advance_x, double *advance_y)
{ 
    FontGlyph *     glyph = NULL;

    glyph = PGFT_Cache_FindGlyph(ft, &PGFT_INTERNALS(font)->cache, 
                                 character, render);
    
    if (!glyph)
    {
        return -1;
    }

    *minx = (long)glyph->image->left;
    *maxx = (long)(glyph->image->left + glyph->image->bitmap.width);
    *maxy = (long)glyph->image->top;
    *miny = (long)(glyph->image->top - glyph->image->bitmap.rows);
    *advance_x = (double)(glyph->h_metrics.advance.x / 64.0);
    *advance_y = (double)(glyph->h_metrics.advance.y / 64.0);

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
PGFT_GetTextSize(FreeTypeInstance *ft, PyFreeTypeFont *font,
    const FontRenderMode *render, PGFT_String *text, int *w, int *h)
{
    FontText *font_text;

    font_text = PGFT_LoadFontText(ft, font, render, text);

    if (!font_text)
        return -1;

    return PGFT_GetSurfaceSize(ft, font, render, font_text, w, h);
}
