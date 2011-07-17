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

static FT_UInt32 Cache_Hash(const FontRenderMode *, FT_UInt);
static FT_UInt32 GetLoadFlags(const FontRenderMode *);
static void fill_metrics(FontMetrics *metrics,
                         FT_Pos bearing_x, FT_Pos bearing_y,
                         FT_Vector *bearing_rotated,
                         FT_Vector *advance_rotated);

static FontCacheNode *Cache_AllocateNode(FreeTypeInstance *,
                                         FontCache *,
                                         const FontRenderMode *, FT_UInt);
static void Cache_FreeNode(FontCache *, FontCacheNode *);

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

static FT_UInt32
GetLoadFlags(const FontRenderMode *render)
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

static FT_UInt32 
Cache_Hash(const FontRenderMode *render, FT_UInt glyph_index)
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
                Cache_FreeNode(cache, node);
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
                    Cache_FreeNode(cache, node);
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
        PGFT_char character, const FontRenderMode *render)
{
    FontCacheNode **nodes = cache->nodes;
    FontCacheNode *node, *prev;
    CacheNodeKey key;

    FT_UInt32 hash = Cache_Hash(render, character);
    FT_UInt32 bucket = hash & cache->size_mask;
    
    key.mode = *render;
    key.ch = character;
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

    node = Cache_AllocateNode(ft, cache, render, character);

#ifdef PGFT_DEBUG_CACHE
    cache->_debug_miss++;
#endif

    return node ? &node->glyph : NULL;
}

static void
Cache_FreeNode(FontCache *cache, FontCacheNode *node)
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

static FontCacheNode *
Cache_AllocateNode(FreeTypeInstance *ft, 
        FontCache *cache, const FontRenderMode *render, PGFT_char character)
{
    static FT_Vector delta = {0, 0};

    int embolden = render->style & FT_STYLE_BOLD;
    FontCacheNode *node = NULL;
    FontGlyph *glyph = NULL;
    FT_Glyph image = NULL;

    FT_Glyph_Metrics *ft_metrics;
    FT_Face face;

    FT_UInt32 load_flags;
    FT_Pos bold_str = 0;
    FT_Pos bold_advance = 0;
    FT_UInt gindex;
    FT_UInt32 bucket;

    FT_Fixed rotation_angle = render->rotation_angle;
    FT_Vector unit;
    FT_Matrix transform;
    FT_Vector h_bearing_rotated;
    FT_Vector v_bearing_rotated;
    FT_Vector h_advance_rotated;
    FT_Vector v_advance_rotated;

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
    memset(node, 0, sizeof(FontCacheNode));
    if (!node) {
        return 0;
    }
    glyph = &node->glyph;

    /*
     * Calculate the corresponding glyph index for the char
     */
    gindex = FTC_CMapCache_Lookup(ft->cache_charmap, 
                                  (FTC_FaceID)&(cache->font->id), -1,
                                  (FT_UInt32)character);

    if (!gindex)
    {
        _PGFT_SetError(ft, "Glyph character not found in font", 0);
        goto cleanup;
    }

    glyph->glyph_index = gindex;

    /*
     * Get loading information
     */
    load_flags = GetLoadFlags(render);

    /*
     * Load the glyph into the glyph slot
     */
    if (FT_Load_Glyph(face, glyph->glyph_index, (FT_Int)load_flags) ||
        FT_Get_Glyph(face->glyph, &image))
        goto cleanup;

    if (embolden)
    {
        bold_str = PGFT_GetBoldStrength(face);
        /* bold_advance = (bold_str * 3) / 2; */
        bold_advance = 4 * bold_str;
        if (FT_Outline_Embolden(&((FT_OutlineGlyph)image)->outline, bold_str))
            goto cleanup;
    }

    /*
     * Collect useful metric values
     */
    ft_metrics = &face->glyph->metrics;
    h_advance_rotated.x = ft_metrics->horiAdvance + bold_advance;
    h_advance_rotated.y = 0;
    v_advance_rotated.x = 0;
    v_advance_rotated.y = ft_metrics->vertAdvance + bold_advance;

    /*
     * Perform any transformations
     */
    if (render->style & FT_STYLE_ITALIC)
    {
        FT_Outline_Transform(&(((FT_OutlineGlyph)image)->outline),
                             &PGFT_SlantMatrix);
    }

    if (rotation_angle != 0)
    {
        FT_Angle counter_rotation =
            rotation_angle ? PGFT_INT_TO_6(360) - rotation_angle : 0;

        FT_Vector_Unit(&unit, rotation_angle);
        transform.xx = unit.x;  /*  cos(angle) */
        transform.xy = -unit.y; /* -sin(angle) */
        transform.yx = unit.y;  /*  sin(angle) */
        transform.yy = unit.x;  /*  cos(angle) */
        if (FT_Glyph_Transform(image, &transform, &delta))
        {
            goto cleanup;
        }
        FT_Vector_Rotate(&h_advance_rotated, rotation_angle);
        FT_Vector_Rotate(&v_advance_rotated, counter_rotation);
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

    /* Fill the glyph */
    glyph->image = (FT_BitmapGlyph)image;
    glyph->width = PGFT_INT_TO_6(glyph->image->bitmap.width);
    glyph->height = PGFT_INT_TO_6(glyph->image->bitmap.rows);
    glyph->bold_strength = bold_str;
    h_bearing_rotated.x = PGFT_INT_TO_6(glyph->image->left);
    h_bearing_rotated.y = PGFT_INT_TO_6(glyph->image->top);
    fill_metrics(&glyph->h_metrics,
                 ft_metrics->horiBearingX + bold_advance,
                 ft_metrics->horiBearingY + bold_advance,
                 &h_bearing_rotated, &h_advance_rotated);

    if (rotation_angle == 0)
    {
        v_bearing_rotated.x = ft_metrics->vertBearingX - bold_advance / 2;
        v_bearing_rotated.y = ft_metrics->vertBearingY;
    }
    else
    {
        /*
         * Adjust the vertical metrics.
         */
        FT_Vector v_origin;

        v_origin.x = (glyph->h_metrics.bearing_x -
                      ft_metrics->vertBearingX + bold_advance / 2);
        v_origin.y = (glyph->h_metrics.bearing_y +
                      ft_metrics->vertBearingY);
        FT_Vector_Rotate(&v_origin, rotation_angle);
        v_bearing_rotated.x = glyph->h_metrics.bearing_rotated.x - v_origin.x;
        v_bearing_rotated.y = v_origin.y - glyph->h_metrics.bearing_rotated.y;
    }
    fill_metrics(&glyph->v_metrics,
                 ft_metrics->vertBearingX + bold_advance,
                 ft_metrics->vertBearingY + bold_advance,
                 &v_bearing_rotated, &v_advance_rotated);

    /*
     * Update cache internals
     */
    node->key.mode = *render;
    node->key.ch = character;
    node->hash = Cache_Hash(render, character);
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
    if (image)
        FT_Done_Glyph(image);

    _PGFT_free(node);
    return NULL;
}

static void
fill_metrics(FontMetrics *metrics, 
             FT_Pos bearing_x, FT_Pos bearing_y,
             FT_Vector *bearing_rotated,
             FT_Vector *advance_rotated)
{
    metrics->bearing_x = bearing_x;
    metrics->bearing_y = bearing_y;
    metrics->bearing_rotated.x = bearing_rotated->x;
    metrics->bearing_rotated.y = bearing_rotated->y;
    metrics->advance_rotated.x = advance_rotated->x;
    metrics->advance_rotated.y = advance_rotated->y;
}
