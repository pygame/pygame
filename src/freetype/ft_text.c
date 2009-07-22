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

FontText *
PGFT_LoadFontText(FreeTypeInstance *ft, PyFreeTypeFont *font, 
        const FontRenderMode *render, PyObject *text)
{
    FT_Int32 load_flags = FT_LOAD_DEFAULT; 

    int         swapped = 0;
    int         string_length = 0;

    FT_UInt16 * buffer = NULL;
    FT_UInt16 * orig_buffer;
    FT_UInt16 * ch;

    FT_Pos      prev_rsb_delta = 0;
    FT_Fixed    baseline;
    FT_Fixed    bold_str = 0;
    FT_Fixed    y_scale;

    FontText  * ftext = NULL;
    FontGlyph * glyph = NULL;

    FT_Face     face;

    /* compute proper load flags */
    load_flags |= FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH;

    if (render->render_flags & FT_RFLAG_AUTOHINT)
        load_flags |= FT_LOAD_FORCE_AUTOHINT;

    if (render->render_flags & FT_RFLAG_HINTED)
    {
        load_flags |=   (render->render_flags & FT_RFLAG_ANTIALIAS) ?
                        FT_LOAD_TARGET_NORMAL :
                        FT_LOAD_TARGET_MONO;
    }
    else
    {
        load_flags |= FT_LOAD_NO_HINTING;
    }

    /* load our sized face */
    face = _PGFT_GetFaceSized(ft, font, render->pt_size);

    if (!face)
    {
        _PGFT_SetError(ft, "Failed to scale the given face", 0);
        return NULL;
    }

    /* get the text as an unicode string */
    orig_buffer = buffer = PGFT_BuildUnicodeString(text);

    if (!buffer)
    {
        _PGFT_SetError(ft, "Invalid text string specified", 0);
        return NULL;
    }

    /* get the length of the text */
    for (ch = buffer; *ch; ++ch)
        string_length++;

    if (render->style & FT_STYLE_BOLD)
        bold_str = PGFT_GetBoldStrength(face);

    /* create the text struct */
    ftext = malloc(sizeof(FontText));
    ftext->length = string_length;
    ftext->glyphs = calloc((size_t)string_length, sizeof(FontGlyph));
    ftext->glyph_size.x = ftext->glyph_size.y = 0;
    ftext->text_size.x = ftext->text_size.y = 0;
    ftext->baseline_offset.x = ftext->baseline_offset.y = 0;

    y_scale = face->size->metrics.y_scale;

    /*
     * Calculate the position for the underline.
     * underline_pos is the offset of the underline based on top of the
     * font's baseline.
     *
     * HACK: When emboldening the font, the baseline nor the offset change,
     * however the font is indeed thicker and the type overlays the underline
     * on most cases. We manually adjust for this by increasing the offset
     * a proportional amount to the bold strength.
     */
    ftext->underline_pos = (FT_Int16)(
            PGFT_FLOOR(FT_MulFix(face->underline_position + bold_str / 2, y_scale)) >> 6);

    ftext->underline_h = (FT_Int16)(
            (bold_str + PGFT_FLOOR(FT_MulFix(face->underline_thickness, y_scale))) >> 6);

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
            glyph->vvector.x  = (metrics->vertBearingX - bold_str / 2) - metrics->horiBearingX;
            glyph->vvector.y  = -(metrics->vertBearingY + bold_str) - (metrics->horiBearingY + bold_str);

            glyph->vadvance.x = 0;
            glyph->vadvance.y = -(metrics->vertAdvance + bold_str);

            baseline = metrics->height - metrics->horiBearingY;

            if (baseline > ftext->baseline_offset.y)
                ftext->baseline_offset.y = baseline;

            if (metrics->width + bold_str > ftext->glyph_size.x)
                ftext->glyph_size.x = metrics->width + bold_str;

            if (metrics->height + bold_str > ftext->glyph_size.y)
                ftext->glyph_size.y = metrics->height + bold_str;

            if (prev_rsb_delta - face->glyph->lsb_delta >= 32)
                glyph->delta = -1 << 6;
            else if (prev_rsb_delta - face->glyph->lsb_delta < -32)
                glyph->delta = 1 << 6;
            else
                glyph->delta = 0;
        }

    }

    /* 
     * FIXME:
     * Take into account the extra height added by underlines
     */

    free(orig_buffer);
    return ftext;
}

int
PGFT_GetTextAdvances(FreeTypeInstance *ft, PyFreeTypeFont *font, 
        const FontRenderMode *render, FontText *text, FT_Vector *advances)
{
    FT_Face     face;
    FontGlyph   *glyph;
    FT_Pos      track_kern   = 0;
    FT_UInt     prev_index   = 0;
    FT_Vector*  prev_advance = NULL;
    FT_Vector   extent       = {0, 0};
    FT_Int      i;
    FT_Fixed    bold_str    = 0;

    face = _PGFT_GetFaceSized(ft, font, render->pt_size);

    if (!face)
        return -1;

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
        glyph = &(text->glyphs[i]);

        if (!glyph->image)
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

FT_UInt16 *
PGFT_BuildUnicodeString(PyObject *obj)
{
    size_t len;
    FT_UInt16 *utf16_buffer = NULL;
    char *tmp_buffer;

    if (PyUnicode_Check(obj))
    {
        /*
         * For unicode objects, create a new Bytes object
         * with the unicode contents as UTF16 and copy
         * the raw contents of that object.
         */
        PyObject *utf_bytes;

        utf_bytes = PyUnicode_AsUTF16String(obj);
        Bytes_AsStringAndSize(utf_bytes, &tmp_buffer, (int *)&len);
        utf16_buffer = malloc(len + 2);

        memcpy(utf16_buffer, tmp_buffer, len);
        utf16_buffer[len / sizeof(FT_UInt16)] = 0;

        Py_DECREF(utf_bytes);
    }
    else if (Bytes_Check(obj))
    {
        /*
         * For bytes objects, assume the bytes are
         * Latin1 text (who would manually enter bytes as
         * UTF8 anyway?), so manually copy the raw contents
         * of the object expanding each byte to 16 bits.
         */
        size_t i;

        Bytes_AsStringAndSize(obj, &tmp_buffer, (int *)&len);
        utf16_buffer = malloc((size_t)(len + 1) * sizeof(FT_UInt16));

        for (i = 0; i < len; ++i)
            utf16_buffer[i] = (FT_UInt16)tmp_buffer[i];

        utf16_buffer[len] = 0;
    }

    return utf16_buffer;
}
