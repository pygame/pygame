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
#define NO_PYGAME_C_API

#include "ft_wrap.h"
#include FT_MODULE_H

typedef struct __cachenodekey
{
    FontRenderMode mode;
    PGFT_char ch;
} CacheNodeKey;

typedef struct __cachenode
{
    FontGlyph glyph;
    struct __cachenode *next;
    CacheNodeKey key;
    FT_UInt32 hash;
} FontCacheNode;

static FT_UInt32 Cache_Hash(const FontRenderMode *, FT_UInt);

static FontCacheNode *Cache_AllocateNode(FontCache *,
                                         const FontRenderMode *, FT_UInt, void *);
static void Cache_FreeNode(FontCache *, FontCacheNode *);
static int equal_node_keys(CacheNodeKey *a, CacheNodeKey *b);

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
PGFT_Cache_Init(FreeTypeInstance *ft, FontCache *cache)
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
PGFT_Cache_FindGlyph(PGFT_char character, const FontRenderMode *render,
                     FontCache *cache, void *internal)
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

    node = Cache_AllocateNode(cache, render, character, internal);

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
Cache_AllocateNode(FontCache *cache, const FontRenderMode *render, PGFT_char character, void *internal)
{
    FontCacheNode *node = _PGFT_malloc(sizeof(FontCacheNode));
    FT_UInt32 bucket;

    if (!node)
    {
        return NULL;
    }
    memset(node, 0, sizeof(FontCacheNode));

    if (PGFT_LoadGlyph(&node->glyph, character, render, internal))
    {
        goto cleanup;
    }

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
    _PGFT_free(node);
    return NULL;
}
