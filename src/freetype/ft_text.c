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

FontText *
PGFT_LoadFontText(FreeTypeInstance *ft, PyFreeTypeFont *font, 
        const FontRenderMode *render, PGFT_String *text)
{
    Py_ssize_t  string_length = PGFT_String_GET_LENGTH(text);

    PGFT_char * buffer = PGFT_String_GET_DATA(text);
    PGFT_char * ch;

    FT_Fixed    y_scale;
    FT_Fixed    bold_str = 0;

    FontText    *ftext = NULL;
    FontGlyph   *glyph = NULL;
    FontGlyph   **glyph_array = NULL;

    FT_Face     face;

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
        ftext->glyphs =
            _PGFT_malloc((size_t)string_length * sizeof(FontGlyph *));
        if (!ftext->glyphs)
        {
            PyErr_NoMemory();
            return NULL;
        }

        _PGFT_free(ftext->advances);
        ftext->advances =
            _PGFT_malloc((size_t)string_length * sizeof(FT_Vector));
        if (!ftext->advances)
        {
            _PGFT_free(ftext->glyphs);
            ftext->glyphs = NULL;
            PyErr_NoMemory();
            return NULL;
        }

    }

    ftext->length = string_length;
    ftext->glyph_size.x = ftext->glyph_size.y = 0;
    ftext->text_size.x = ftext->text_size.y = 0;
    ftext->baseline_offset.x = ftext->baseline_offset.y = 0;
    ftext->underline_pos = ftext->underline_size = 0;

    y_scale = face->size->metrics.y_scale;

    /* fill it with the glyphs */
    glyph_array = ftext->glyphs;

    for (ch = buffer; *ch; ++ch)
    {
        /*
         * Load the corresponding glyph from the cache
         */
        glyph = PGFT_Cache_FindGlyph(ft, &PGFT_INTERNALS(font)->cache,
                                     *((FT_UInt32 *)ch), render);

        if (!glyph)
            continue;
           
        /*
         * Do size calculations for all the glyphs in the text
         */
        if (glyph->baseline > ftext->baseline_offset.y)
            ftext->baseline_offset.y = glyph->baseline;

        if (glyph->size.x > ftext->glyph_size.x)
            ftext->glyph_size.x = glyph->size.x;

        if (glyph->size.y > ftext->glyph_size.y)
            ftext->glyph_size.y = glyph->size.y;

        *glyph_array++ = glyph;
    }

    if (render->style & FT_STYLE_UNDERLINE &&
        (render->render_flags & FT_RFLAG_VERTICAL) == 0 &&
        (render->rotation_angle == 0))
    {
        FT_Fixed scale;
        FT_Fixed underline_pos;
        FT_Fixed underline_size;
        
        scale = face->size->metrics.y_scale;

        underline_pos = FT_MulFix(face->underline_position, scale);
        underline_size = FT_MulFix(face->underline_thickness, scale) + bold_str;

        /*
         * HACK HACK HACK
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

        ftext->underline_pos =
            ftext->glyph_size.y - ftext->baseline_offset.y - (underline_pos / 4);

        ftext->underline_size = underline_size;
    }

    return ftext;
}

int
PGFT_LoadTextAdvances(FreeTypeInstance *ft, PyFreeTypeFont *font, 
        const FontRenderMode *render, FontText *text)
{
    /* Default kerning mode for all text */
    const int FT_KERNING_MODE = 1;

    FT_Face     face;
    FontGlyph   *glyph;
    FT_Pos      track_kern   = 0;
    FT_UInt     prev_index   = 0;
    FT_Vector*  prev_advance = NULL;
    FT_Vector   extent       = {0, 0};
    FT_Int      i;
    FT_Fixed    bold_str    = 0;
    FT_Vector   *advances   = NULL;

    face = _PGFT_GetFaceSized(ft, font, render->pt_size);

    if (!face)
    {
        RAISE(PyExc_SDLError, PGFT_GetError(ft));
        return -1;
    }

    advances = text->advances;

    if (render->style & FT_STYLE_BOLD)
        bold_str = PGFT_GetBoldStrength(face);

#if 0
    if (!(render->render_flags & FT_RFLAG_VERTICAL) && render->kerning_degree)
    {
        FT_Fixed  ptsize;

        ptsize = FT_MulFix(face->units_per_EM, face->size->metrics.x_scale);

        if (FT_Get_Track_Kerning(face, ptsize << 10, 
                    -render->kerning_degree, &track_kern))
            track_kern = 0;
        else
            track_kern >>= 10;
    }
#endif

    for (i = 0; i < text->length; i++)
    {
        glyph = text->glyphs[i];

        if (!glyph || !glyph->image)
            continue;

        if (render->render_flags & FT_RFLAG_VERTICAL)
            advances[i] = glyph->vadvance;
        else
        {
            advances[i] = glyph->image->advance;

            /* Convert to 26.6 */
            advances[i].x >>= 10;
            advances[i].y >>= 10;

            /* Apply BOLD transformation */
            if (advances[i].x)
                advances[i].x += bold_str;

            if (advances[i].y)
                advances[i].y += bold_str;

            if (prev_advance)
            {
                prev_advance->x += track_kern;

                if (FT_KERNING_MODE > 0)
                {
                    FT_Vector  kern;

                    FT_Get_Kerning(face, prev_index, glyph->glyph_index,
                            FT_KERNING_UNFITTED, &kern);

                    prev_advance->x += kern.x;
                    prev_advance->y += kern.y;

                    if (FT_KERNING_MODE > 1) /* KERNING_MODE_NORMAL */
                        prev_advance->x += glyph->delta;
                }
            }
        }

        if (prev_advance)
        {
            if (render->render_flags & FT_RFLAG_HINTED)
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
        if (render->render_flags & FT_RFLAG_HINTED)
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

    return 0;
}

