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
#include FT_OUTLINE_H 
 

#define BOLD_FACTOR     0.08
#define SLANT_FACTOR    0.22

int
PGFT_Style_MakeBold(FT_Face face)
{
    FT_GlyphSlot    slot = face->glyph;
    FT_Pos          xstr, ystr;
    FT_Error        error;

    xstr = FT_MulFix(face->units_per_EM,
            face->size->metrics.y_scale);
    xstr = (FT_Fixed)(xstr * BOLD_FACTOR);
    ystr = xstr;

    if (slot->format == FT_GLYPH_FORMAT_OUTLINE)
    {
        FT_Outline_Embolden(&slot->outline, xstr);
        xstr = xstr * 2;
        ystr = xstr;
    }
    else if (slot->format == FT_GLYPH_FORMAT_BITMAP)
    {
        FT_Library    library = slot->library;

        xstr &= ~63;
        ystr &= ~63;

        error = FT_GlyphSlot_Own_Bitmap(slot);
        if (error)
            return error;

        error = FT_Bitmap_Embolden(library, &slot->bitmap, xstr, ystr);
        if (error)
            return error;
    }

    if (slot->advance.x)
        slot->advance.x += xstr;

    if (slot->advance.y)
        slot->advance.y += ystr;

    slot->metrics.width        += xstr;
    slot->metrics.height       += ystr;
    slot->metrics.horiBearingY += ystr;
    slot->metrics.horiAdvance  += xstr;
    slot->metrics.vertBearingX -= xstr / 2;
    slot->metrics.vertBearingY += ystr;
    slot->metrics.vertAdvance  += ystr;
    return 0;
}


