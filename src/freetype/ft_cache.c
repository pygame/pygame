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
#include FT_OUTLINE_H

#define SLANT_FACTOR    0.22
static FT_Matrix PGFT_SlantMatrix = 
{
    (1 << 16),  (FT_Fixed)(SLANT_FACTOR * (1 << 16)),
    0,          (1 << 16) 
};

FT_UInt32 _PGFT_Cache_Hash(const FontRenderMode *, FT_UInt);
FT_UInt32 _PGFT_GetLoadFlags(const FontRenderMode *);

FontCacheNode *_PGFT_Cache_AllocateNode(FreeTypeInstance *, 
        FontCache *, const FontRenderMode *, FT_UInt);
void _PGFT_Cache_FreeNode(FontCache *, FontCacheNode *);

const int render_flags_mask = (FT_RFLAG_ANTIALIAS |
                               FT_RFLAG_HINTED |
                               FT_RFLAG_AUTOHINT);

static int
equal_node_keys(CacheNodeKey *a, CacheNodeKey *b)
{
    return (a->ch == b->ch &&
            a->mode.pt_size == b->mode.pt_size &&
            a->mode.rotation_angle == b->mode.rotation_angle &&
            (a->mode.render_flags & render_flags_mask) ==
            (b->mode.render_flags & render_flags_mask) &&
            a->mode.style == b->mode.style);
}

FT_UInt32
_PGFT_GetLoadFlags(const FontRenderMode *render)
{
    FT_UInt32 load_flags = FT_LOAD_DEFAULT;

    load_flags |= FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH;

    if (render->render_flags & FT_RFLAG_AUTOHINT)
        load_flags |= FT_LOAD_FORCE_AUTOHINT;

    if (render->render_flags & FT_RFLAG_HINTED)
    {
        load_flags |= ((render->render_flags & FT_RFLAG_ANTIALIAS) ?
                       FT_LOAD_TARGET_NORMAL :
                       FT_LOAD_TARGET_MONO);
    }
    else
    {
        load_flags |= FT_LOAD_NO_HINTING;
    }

    return load_flags;
}

FT_UInt32 
_PGFT_Cache_Hash(const FontRenderMode *render, FT_UInt glyph_index)
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

int
PGFT_Cache_Init(FreeTypeInstance *ft, 
                FontCache *cache, PyFreeTypeFont *parent)
{
    int cache_size = MAX(ft->cache_size - 1, PGFT_MIN_CACHE_SIZE - 1);
    int i;

    /*
     * Make sure this is a power of 2.
     */
    cache_size = cache_size | (cache_size >> 1);
    cache_size = cache_size | (cache_size >> 2);
    cache_size = cache_size | (cache_size >> 4);
    cache_size = cache_size | (cache_size >> 8);
    cache_size = cache_size | (cache_size >>16);

    cache_size = cache_size + 1;

    cache->nodes = _PGFT_malloc((size_t)cache_size * sizeof(FontGlyph *));
    if (!cache->nodes)
        return -1;
    for (i=0; i < cache_size; ++i)
        cache->nodes[i] = NULL;
    cache->depths = _PGFT_malloc((size_t)cache_size);
    if (!cache->depths)
    {
        _PGFT_free(cache->nodes);
        cache->nodes = NULL;
        return -1;
    }
    memset(cache->depths, 0, cache_size);
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
    return 0;
}

void 
PGFT_Cache_Destroy(FontCache *cache)
{
    FT_UInt i;
    FontCacheNode *node, *next;

    if (cache == NULL)
        return;

#ifdef PGFT_DEBUG_CACHE
    fprintf(stderr, "Cache stats:\n");
    fprintf(stderr, "\t%d accesses in total\n", cache->_debug_access);
    fprintf(stderr, "\t%d hits / %d misses\n", cache->_debug_hit, cache->_debug_miss);
    fprintf(stderr, "\t%f hit ratio\n", (float)cache->_debug_hit/(float)cache->_debug_access);
    fprintf(stderr, "\t%d nodes kicked\n", cache->_debug_delete_count);
#endif

    if (cache->nodes != NULL)
    {
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
        _PGFT_free(cache->nodes);
        cache->nodes = NULL;
    }
    _PGFT_free(cache->depths);
    cache->depths = NULL;
}

void
PGFT_Cache_Cleanup(FontCache *cache)
{
    const FT_Byte MAX_BUCKET_DEPTH = 2;
    FontCacheNode *node, *prev;
    FT_UInt32 i;

    for (i = 0; i <= cache->size_mask; ++i)
    {
        while (cache->depths[i] > MAX_BUCKET_DEPTH)
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

FontGlyph *
PGFT_Cache_FindGlyph(FreeTypeInstance *ft, FontCache *cache, 
        FT_UInt32 character, const FontRenderMode *render)
{
    FontCacheNode **nodes = cache->nodes;
    FontCacheNode *node, *prev;
    CacheNodeKey key = { *render, character };

    FT_UInt32 hash = _PGFT_Cache_Hash(render, character);
    FT_UInt32 bucket = hash & cache->size_mask;
    
    node = nodes[bucket];
    prev = NULL;

#ifdef PGFT_DEBUG_CACHE
    cache->_debug_access++;
#endif
    
    while (node)
    {
        if (equal_node_keys(&node->key, &key))
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

void
_PGFT_Cache_FreeNode(FontCache *cache, FontCacheNode *node)
{
    if (node == NULL)
        return;

#ifdef PGFT_DEBUG_CACHE
    cache->count--;
#endif

    cache->depths[node->hash & cache->size_mask]--;

    FT_Done_Glyph((FT_Glyph)(node->glyph.image));
    _PGFT_free(node);
}

static void
_PGFT_Metrics_Rotate(FontMetrics *metrics,
                     FT_BitmapGlyph image,
                     FT_Angle angle)
{
    FT_Pos    min_x = metrics->bearing_x;
    FT_Pos    max_x = min_x + PGFT_INT_TO_6(image->bitmap.width);
    FT_Pos    max_y = metrics->bearing_y;
    FT_Pos    min_y = max_y - PGFT_INT_TO_6(image->bitmap.rows);
    FT_Vector topleft = {min_x, max_y};
    FT_Vector topright = {max_x, max_y};
    FT_Vector bottomleft = {min_x, min_y};
    FT_Vector bottomright = {max_x, min_y};
    FT_Vector bearing_x = {metrics->bearing_x, 0};
    FT_Vector bearing_y = {0, metrics->bearing_y};

    FT_Vector_Rotate(&metrics->advance, angle);

    metrics->bearing_x = PGFT_INT_TO_6(image->left);
    metrics->bearing_y = PGFT_INT_TO_6(image->top);
#if 0
    FT_Vector_Rotate(&bearing, angle);
    metrics->bearing_x = bearing.x;
    metrics->bearing_y = bearing.y;
#endif
#if 0
    FT_Vector_Rotate(&topleft, angle);
    FT_Vector_Rotate(&topright, angle);
    FT_Vector_Rotate(&bottomleft, angle);
    FT_Vector_Rotate(&bottomright, angle);

    min_x = bottomleft.x;
    if (bottomright.x < min_x)
    {
        min_x = bottomright.x;
    }
    if (topleft.x < min_x)
    {
        min_x = topleft.x;
    }
    if (topright.x < min_x)
    {
        min_x = topright.x;
    }
    max_y = topright.y;
    if (topleft.y > max_y)
    {
        max_y = bottomright.y;
    }
    if (bottomright.y > max_y)
    {
        max_y = topleft.y;
    }
    if (bottomleft.y > max_y)
    {
        max_y = topright.y;
    }
    metrics->bearing_x = min_x;
    metrics->bearing_y = max_y;
#endif
}

FontCacheNode *
_PGFT_Cache_AllocateNode(FreeTypeInstance *ft, 
        FontCache *cache, const FontRenderMode *render, FT_UInt character)
{
    static FT_Vector delta = {0, 0};

    int embolden = render->style & FT_STYLE_BOLD;
    FontCacheNode *node = NULL;
    FontGlyph *glyph = NULL;
    FontMetrics *metrics;
    FT_Glyph image;

    FT_Glyph_Metrics *ft_metrics;
    FT_Face face;

    FT_UInt32 load_flags;
    FT_Pos bold_str = 0;
    int gindex;
    FT_UInt32 bucket;

    FT_Fixed rotation_angle = render->rotation_angle;
    FT_Vector unit;
    FT_Matrix transform;

    FT_Error error = 0;

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
    node = _PGFT_malloc(sizeof(FontCacheNode));
    if (!node) {
        return 0;
    }
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
     */
    if (FT_Load_Glyph(face, glyph->glyph_index, (FT_Int)load_flags) ||
        FT_Get_Glyph(face->glyph, &image))
        goto cleanup;

    if (embolden)
    {
        bold_str = PGFT_GetBoldStrength(face);
        if (FT_Outline_Embolden(&((FT_OutlineGlyph)image)->outline, bold_str))
            goto cleanup;
    }

    /*
     * Precalculate useful metric values
     */
    ft_metrics = &face->glyph->metrics;
    glyph->bold_strength = bold_str;
    metrics = &glyph->h_metrics;
    metrics->bearing_x = ft_metrics->horiBearingX;
    metrics->bearing_y = ft_metrics->horiBearingY;
    metrics->advance.x = ft_metrics->horiAdvance + bold_str;
    metrics->advance.y = 0;
    metrics = &glyph->v_metrics;
    metrics->bearing_x = ft_metrics->vertBearingX;
    metrics->bearing_y = ft_metrics->vertBearingY;
    metrics->advance.x = 0;
    metrics->advance.y = ft_metrics->vertAdvance + bold_str;

    /*
     * Perform any transformations
     */
    if (rotation_angle != 0)
    {
        FT_Vector_Unit(&unit, rotation_angle);
        transform.xx = unit.x;  /*  cos(angle) */
        transform.xy = -unit.y; /* -sin(angle) */
        transform.yx = unit.y;  /*  sin(angle) */
        transform.yy = unit.x;  /*  cos(angle) */
        if (FT_Glyph_Transform(image, &transform, &delta))
        {
            goto cleanup;
        }
    }

    if (render->style & FT_STYLE_ITALIC)
    {
        FT_Outline_Transform(&(((FT_OutlineGlyph)image)->outline),
                             &PGFT_SlantMatrix);
    }

    /*
     * Finished with transformations, now replace with a bitmap
     */
    error = FT_Glyph_To_Bitmap(&image, FT_RENDER_MODE_NORMAL, 0, 1);
    if (error)
    {
        _PGFT_SetError(ft, "Rendering glyphs", error);
        RAISE(PyExc_SDLError, PGFT_GetError(ft));
        goto cleanup;
    }
    glyph->image = (FT_BitmapGlyph)image;

    /*
     * Adjust the metrics.
     */
    if (rotation_angle != 0)
    {
        _PGFT_Metrics_Rotate(&glyph->h_metrics, glyph->image, rotation_angle);
        _PGFT_Metrics_Rotate(&glyph->v_metrics, glyph->image, rotation_angle);
    }

    /*
     * Update cache internals
     */
    node->key.mode = *render;
    node->key.ch = character;
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
        FT_Done_Glyph((FT_Glyph)glyph->image);

    _PGFT_free(node);
    return NULL;
}
