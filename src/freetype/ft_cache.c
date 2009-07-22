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

FT_UInt32 _PGFT_Cache_Hash(const FontRenderMode *, FT_UInt, int);
FT_UInt32 _PGFT_GetLoadFlags(const FontRenderMode *);

FontGlyph *_PGFT_Cache_AllocateGlyph(PGFT_Cache *, const FontRenderMode *, FT_UInt);
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

FT_UInt32 _PGFT_Cache_Hash(const FontRenderMode *render, FT_UInt glyph_index, int hash_mode)
{
    static const FT_UInt32 _CUCKOO_HASHES[] =
    {
        0xDEADC0DE,
        0xBEEF1234
    };

	const FT_UInt32 m = 0x5bd1e995;
	const int r = 24;

	FT_UInt32 h, k; 

    /* 
     * Quick hashing algorithm, based off MurmurHash2.
     * Assumes sizeof(FontRenderMode) == 8
     */

    h = (glyph_index << 8) | _CUCKOO_HASHES[hash_mode];

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

PGFT_Cache *PGFT_Cache_Create(FT_Face parent)
{
    /* 
     * TODO: Let user specify the desired
     * size for the cache?
     */
    const int cache_size = 64;
    PGFT_Cache *cache = NULL;

    cache = malloc(sizeof(PGFT_Cache));

    if (!cache)
        return NULL;

    cache->face = parent;
    cache->nodes = calloc(cache_size, sizeof(FontGlyph *));
    cache->size_mask = (cache_size - 1);

    return cache;
}

void PGFT_Cache_Destroy(PGFT_Cache *cache)
{
    FT_UInt i;

    if (cache == NULL)
        return;

    for (i = 0; i <= cache->size_mask; ++i)
        _PGFT_Cache_FreeGlyph(cache->nodes[i]);

    free(cache->nodes);
    free(cache);
}

FontGlyph *PGFT_Cache_FindGlyph(PGFT_Cache *cache, 
        FT_UInt glyph_index, const FontRenderMode *render)
{
    const int MAX_CUCKOO_ITER = 6;
    FontGlyph **glyph = NULL, *next_glyph, *aux, *alloc;
    int cuckoo_hash = 0;
    FT_UInt32 hashes[2];

    do
    {
        FT_UInt32 hash = _PGFT_Cache_Hash(render, glyph_index, cuckoo_hash & 1);
        glyph = &cache->nodes[hash & cache->size_mask];

        if (*glyph == NULL)
        {
           *glyph = _PGFT_Cache_AllocateGlyph(cache, render, glyph_index);
           return *glyph;
        }
        
        if ((*glyph)->glyph_index == glyph_index)
        {
            return *glyph;
        }

    } while (cuckoo_hash++ < 2);

    /*
     * Glyph is not cached.
     * Place it on the cache!
     */
    cuckoo_hash = 0;

    next_glyph = *glyph;
    *glyph = _PGFT_Cache_AllocateGlyph(cache, render, glyph_index);
    alloc = *glyph;

    while (next_glyph != NULL && cuckoo_hash++ < MAX_CUCKOO_ITER)
    {
        hashes[0] = next_glyph->hashes[0];
        hashes[1] = next_glyph->hashes[1];

        if (glyph == &cache->nodes[hashes[0] & cache->size_mask])
            glyph = &cache->nodes[hashes[1] & cache->size_mask];
        else
            glyph = &cache->nodes[hashes[0] & cache->size_mask];

        aux = *glyph;
        *glyph = next_glyph;
        next_glyph = aux;
    }

    if (next_glyph)
        _PGFT_Cache_FreeGlyph(next_glyph);

    return alloc;
}

void _PGFT_Cache_FreeGlyph(FontGlyph *glyph)
{
    if (glyph == NULL)
        return;

    FT_Done_Glyph(glyph->image);
    free(glyph);
}

FontGlyph *_PGFT_Cache_AllocateGlyph(PGFT_Cache *cache, 
        const FontRenderMode *render, FT_UInt glyph_index)
{
    FT_Glyph_Metrics *metrics;
    FontGlyph *glyph = NULL;
    FT_UInt32 load_flags;
    FT_Fixed bold_str = 0;

    /* 
     * Allocate cache node 
     */
    glyph = malloc(sizeof(FontGlyph));

    glyph->glyph_index = glyph_index;
    glyph->hashes[0] = _PGFT_Cache_Hash(render, glyph_index, 0);
    glyph->hashes[1] = _PGFT_Cache_Hash(render, glyph_index, 1);

    /*
     * Loading information
     */
    load_flags = _PGFT_GetLoadFlags(render);

    if (render->style & FT_STYLE_BOLD)
        bold_str = PGFT_GetBoldStrength(cache->face);

    /*
     * Load the glyph into the glyph slot
     * TODO: error handling
     */
    if (FT_Load_Glyph(cache->face, glyph_index, (FT_Int)load_flags) != 0 ||
        FT_Get_Glyph(cache->face->glyph, &(glyph->image)) != 0)
        return NULL;

    /*
     * Precalculate useful metric values
     */
    metrics = &cache->face->glyph->metrics;

    glyph->vvector.x  = (metrics->vertBearingX - bold_str / 2) - metrics->horiBearingX;
    glyph->vvector.y  = -(metrics->vertBearingY + bold_str) - (metrics->horiBearingY + bold_str);

    glyph->vadvance.x = 0;
    glyph->vadvance.y = -(metrics->vertAdvance + bold_str);

    glyph->baseline = metrics->height - metrics->horiBearingY;

    glyph->size.x = metrics->width + bold_str;
    glyph->size.y = metrics->height + bold_str;

    glyph->lsb_delta = cache->face->glyph->lsb_delta;

    return glyph;
}
