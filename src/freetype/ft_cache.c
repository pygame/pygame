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

#define PGFT_DEBUG_CACHE

FT_UInt32 _PGFT_Cache_Hash(const FontRenderMode *, FT_UInt);
FT_UInt32 _PGFT_GetLoadFlags(const FontRenderMode *);

FontGlyph *_PGFT_Cache_AllocateGlyph(FreeTypeInstance *, 
        PGFT_Cache *, const FontRenderMode *, FT_UInt);
void _PGFT_Cache_FreeGlyph(FontGlyph *);


FT_UInt32 _PGFT_GetLoadFlags(const FontRenderMode *render)
{
    FT_UInt32 load_flags = FT_LOAD_DEFAULT;

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

    return load_flags;
}

FT_UInt32 _PGFT_Cache_Hash(const FontRenderMode *render, FT_UInt glyph_index)
{
	const FT_UInt32 m = 0x5bd1e995;
	const int r = 24;

	FT_UInt32 h, k; 

    /* 
     * Quick hashing algorithm, based off MurmurHash2.
     * Assumes sizeof(FontRenderMode) == 8
     */

    h = (glyph_index << 12) ^ glyph_index;

    k = *(const FT_UInt32 *)render;
    k *= m; k ^= k >> r; 
    k *= m; h *= m; h ^= k;

    k = *(((const FT_UInt32 *)render) + 1);
    k *= m; k ^= k >> r; 
    k *= m; h *= m; h ^= k;

	h ^= h >> 13;
	h *= m;
	h ^= h >> 15;

	return h;
} 

void PGFT_Cache_Init(PGFT_Cache *cache, PyFreeTypeFont *parent)
{
    /* 
     * TODO: Let user specify the desired
     * size for the cache?
     */
    const int cache_size = 64;

    cache->font = parent;
    cache->nodes = calloc(cache_size, sizeof(FontGlyph *));
    cache->size_mask = (cache_size - 1);
    cache->lru_counter = 0;
}

void PGFT_Cache_Destroy(PGFT_Cache *cache)
{
    FT_UInt i;

    if (cache == NULL)
        return;

    for (i = 0; i <= cache->size_mask; ++i)
        _PGFT_Cache_FreeGlyph(cache->nodes[i]);

    free(cache->nodes);
}

FontGlyph *PGFT_Cache_FindGlyph(FreeTypeInstance *ft, PGFT_Cache *cache, 
        FT_UInt character, const FontRenderMode *render)
{
    FontGlyph **nodes = cache->nodes;
    FT_UInt32 lowest, current, first, i;

    const FT_UInt32 hash = _PGFT_Cache_Hash(render, character);
    FT_UInt32 perturb;
    
    i = hash;
    current = first = lowest = hash & cache->size_mask;
    perturb = hash;

    /*
     * Browse the whole cache with linear probing until either:
     *
     *  a) we find an empty spot for the glyph
     *      => we load our glyph and store it there
     *
     *  b) we find the glyph already on the cache
     *      => we return the reference to that glyph
     *
     *  c) we find no glyphs, and no empty spots
     *      => we kick the LRU glyph from the cache,
     *      and store the new one there
     */
    do
    {
        if (nodes[current] == NULL)
        {
            /* A: found empty spot */
            return (nodes[current] = 
                   _PGFT_Cache_AllocateGlyph(ft, cache, render, character));
        }
        
        if (nodes[current]->hash == hash)
        {
            /* B: found glyph on cache */
            nodes[current]->lru = ++cache->lru_counter;
            return nodes[current];
        }

        if (nodes[current]->lru < nodes[lowest]->lru)
            lowest = current;

        i = (5 * i) + 1 + perturb;
        perturb <<= 5;

        current = i & cache->size_mask;

    } while (current != first);

    /* C: kick glyph from cache */
    _PGFT_Cache_FreeGlyph(nodes[lowest]);

    return (nodes[lowest] = 
            _PGFT_Cache_AllocateGlyph(ft, cache, render, character));
}

void _PGFT_Cache_FreeGlyph(FontGlyph *glyph)
{
    if (glyph == NULL)
        return;

    FT_Done_Glyph(glyph->image);
    free(glyph);
}

FontGlyph *_PGFT_Cache_AllocateGlyph(FreeTypeInstance *ft, 
        PGFT_Cache *cache, const FontRenderMode *render, FT_UInt character)
{
    FT_Glyph_Metrics *metrics;
    FontGlyph *glyph = NULL;
    FT_UInt32 load_flags;
    FT_Fixed bold_str = 0;
    int gindex;

    FT_Face face;

    /*
     * Grab face reference
     */
    face = _PGFT_GetFaceSized(ft, cache->font, render->pt_size);

    if (!face)
    {
        _PGFT_SetError(ft, "Failed to resize face", 0);
        goto cleanup;
    }

    /* 
     * Allocate cache node 
     */
    glyph = malloc(sizeof(FontGlyph));


    /*
     * Calculate the corresponding glyph index for the char
     */
    gindex = FTC_CMapCache_Lookup(ft->cache_charmap, 
            (FTC_FaceID)&(cache->font->id), -1, character);

    if (gindex < 0)
    {
        _PGFT_SetError(ft, "Glyph character not found in font", 0);
        goto cleanup;
    }

    glyph->glyph_index = (FT_UInt)gindex;


    /*
     * Get loading information
     */
    load_flags = _PGFT_GetLoadFlags(render);

    if (render->style & FT_STYLE_BOLD)
        bold_str = PGFT_GetBoldStrength(face);

    /*
     * Load the glyph into the glyph slot
     * TODO: error handling
     */
    if (FT_Load_Glyph(face, glyph->glyph_index, (FT_Int)load_flags) != 0 ||
        FT_Get_Glyph(face->glyph, &(glyph->image)) != 0)
        goto cleanup;

    /*
     * Precalculate useful metric values
     */
    metrics = &face->glyph->metrics;

    glyph->vvector.x  = (metrics->vertBearingX - bold_str / 2) - metrics->horiBearingX;
    glyph->vvector.y  = -(metrics->vertBearingY + bold_str) - (metrics->horiBearingY + bold_str);

    glyph->vadvance.x = 0;
    glyph->vadvance.y = -(metrics->vertAdvance + bold_str);

    glyph->baseline = metrics->height - metrics->horiBearingY;

    glyph->size.x = metrics->width + bold_str;
    glyph->size.y = metrics->height + bold_str;


    /*
     * Update cache internals
     */
    glyph->lru = ++cache->lru_counter;
    glyph->hash = _PGFT_Cache_Hash(render, character);

    return glyph;

    /*
     * Cleanup on error
     */
cleanup:
    if (glyph && glyph->image)
        FT_Done_Glyph(glyph->image);

    free(glyph);
    return NULL;
}
