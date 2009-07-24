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


FT_UInt32 _PGFT_Cache_Hash(const FontRenderMode *, FT_UInt);
FT_UInt32 _PGFT_GetLoadFlags(const FontRenderMode *);

PGFT_CacheNode *_PGFT_Cache_AllocateNode(FreeTypeInstance *, 
        PGFT_Cache *, const FontRenderMode *, FT_UInt);
void _PGFT_Cache_FreeNode(PGFT_Cache *, PGFT_CacheNode *);


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

void PGFT_Cache_Init(FreeTypeInstance *ft, 
        PGFT_Cache *cache, PyFreeTypeFont *parent)
{
    int cache_size = MAX(ft->cache_size - 1, PGFT_MIN_CACHE_SIZE - 1);

    /*
     * Make sure this is a power of 2.
     */
    cache_size = cache_size | (cache_size >> 1);
    cache_size = cache_size | (cache_size >> 2);
    cache_size = cache_size | (cache_size >> 4);
    cache_size = cache_size | (cache_size >> 8);
    cache_size = cache_size | (cache_size >>16);

    cache_size = cache_size + 1;

    cache->nodes = calloc((size_t)cache_size, sizeof(FontGlyph *));
    cache->depths = calloc((size_t)cache_size, sizeof(FT_Byte));
    cache->font = parent;
    cache->free_nodes = NULL;
    cache->size_mask = (FT_UInt32)(cache_size - 1);

#ifdef PGFT_DEBUG_CACHE
    cache->count = 0;
    cache->_debug_delete_count = 0;
    cache->_debug_access = 0;
    cache->_debug_hit = 0;
    cache->_debug_miss = 0;
#endif
}

void PGFT_Cache_Destroy(PGFT_Cache *cache)
{
    FT_UInt i;
    PGFT_CacheNode *node, *next;

    if (cache == NULL)
        return;

#ifdef PGFT_DEBUG_CACHE
    fprintf(stderr, "Cache stats:\n");
    fprintf(stderr, "\t%d accesses in total\n", cache->_debug_access);
    fprintf(stderr, "\t%d hits / %d misses\n", cache->_debug_hit, cache->_debug_miss);
    fprintf(stderr, "\t%f hit ratio\n", (float)cache->_debug_hit/(float)cache->_debug_access);
    fprintf(stderr, "\t%d nodes kicked\n", cache->_debug_delete_count);
#endif

    for (i = 0; i <= cache->size_mask; ++i)
    {
        node = cache->nodes[i];

        while (node)
        {
            next = node->next;
            _PGFT_Cache_FreeNode(cache, node);
            node = next;
        }
    }

    free(cache->nodes);
    free(cache->depths);
}

void PGFT_Cache_Cleanup(PGFT_Cache *cache)
{
    const FT_Byte MAX_BUCKET_DEPTH = 2;
    PGFT_CacheNode *node, *prev;
    FT_UInt32 i;

    for (i = 0; i <= cache->size_mask; ++i)
    {
        if (cache->depths[i] > MAX_BUCKET_DEPTH)
        {
            node = cache->nodes[i];
            prev = NULL;

            for (;;)
            {
                if (!node->next)
                {
#ifdef PGFT_DEBUG_CACHE
                    cache->_debug_delete_count++;
#endif

                    prev->next = NULL; 
                    _PGFT_Cache_FreeNode(cache, node);
                    break;
                }

                prev = node;
                node = node->next;
            }
        }
    }

}

FontGlyph *PGFT_Cache_FindGlyph(FreeTypeInstance *ft, PGFT_Cache *cache, 
        FT_UInt character, const FontRenderMode *render)
{
    PGFT_CacheNode **nodes = cache->nodes;
    PGFT_CacheNode *node, *prev;

    FT_UInt32 hash = _PGFT_Cache_Hash(render, character);
    FT_UInt32 bucket = hash & cache->size_mask;
    
    node = nodes[bucket];
    prev = NULL;

#ifdef PGFT_DEBUG_CACHE
    cache->_debug_access++;
#endif
        
    while (node)
    {
        if (node->hash == hash)
        {
            if (prev)
            {
                prev->next = node->next;
                node->next = nodes[bucket];
                nodes[bucket] = node;
            }

#ifdef PGFT_DEBUG_CACHE
            cache->_debug_hit++;
#endif

            return &node->glyph;
        }

        prev = node;
        node = node->next;
    }

    node = _PGFT_Cache_AllocateNode(ft, cache, render, character);

#ifdef PGFT_DEBUG_CACHE
    cache->_debug_miss++;
#endif

    return &node->glyph;
}

void _PGFT_Cache_FreeNode(PGFT_Cache *cache, PGFT_CacheNode *node)
{
    if (node == NULL)
        return;

#ifdef PGFT_DEBUG_CACHE
    cache->count--;
#endif

    cache->depths[node->hash & cache->size_mask]--;

    FT_Done_Glyph(node->glyph.image);
    free(node);
}

PGFT_CacheNode *_PGFT_Cache_AllocateNode(FreeTypeInstance *ft, 
        PGFT_Cache *cache, const FontRenderMode *render, FT_UInt character)
{
    PGFT_CacheNode *node = NULL;
    FontGlyph *glyph = NULL;

    FT_Glyph_Metrics *metrics;
    FT_Face face;

    FT_UInt32 load_flags;
    FT_Fixed bold_str = 0;
    int gindex;
    FT_UInt32 bucket;

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
    node = malloc(sizeof(PGFT_CacheNode));
    glyph = &node->glyph;

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
    node->hash = _PGFT_Cache_Hash(render, character);
    bucket = node->hash & cache->size_mask;
    node->next = cache->nodes[bucket];
    cache->nodes[bucket] = node;

    cache->depths[bucket]++;

#ifdef PGFT_DEBUG_CACHE
    cache->count++;
#endif

    return node;

    /*
     * Cleanup on error
     */
cleanup:
    if (glyph && glyph->image)
        FT_Done_Glyph(glyph->image);

    free(node);
    return NULL;
}
