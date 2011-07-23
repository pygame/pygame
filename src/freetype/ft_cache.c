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

/* The key is the UTF character, point size (unsigned short),
 * style flags (unsigned short), render flags (unsigned short),
 * and rotation (in whole degrees, unsigned short). Byte order
 * is little-endian.
 */
#define MINKEYLEN (sizeof(PGFT_char) + 2 + 2 + 2 + 2)
#define KEYLEN ((MINKEYLEN + 3) & 0xFFFC)

typedef union __cachenodekey
{
    FT_Byte bytes[KEYLEN];
    FT_UInt32 dwords[KEYLEN / 4];
} CacheNodeKey;

typedef struct __cachenode
{
    FontGlyph glyph;
    struct __cachenode *next;
    CacheNodeKey key;
    FT_UInt32 hash;
} FontCacheNode;

static FT_UInt32 cache_hash(const CacheNodeKey *key);

static FontCacheNode *Cache_AllocateNode(FontCache *,
                                         const FontRenderMode *, FT_UInt, void *);
static void Cache_FreeNode(FontCache *, FontCacheNode *);
static void set_node_key(CacheNodeKey *key, PGFT_char ch,
                         const FontRenderMode *render);
static int equal_node_keys(const CacheNodeKey *a, const CacheNodeKey *b);

const int render_flags_mask = (FT_RFLAG_ANTIALIAS |
                               FT_RFLAG_HINTED |
                               FT_RFLAG_AUTOHINT);

static void
set_node_key(CacheNodeKey *key, PGFT_char ch, const FontRenderMode *render)
{
    const FT_UInt16 style_mask = ~(FT_STYLE_UNDERLINE);
    const FT_UInt16 rflag_mask = ~(FT_RFLAG_VERTICAL | FT_RFLAG_KERNING);
    int i = 0;
    unsigned short rot = (unsigned short)PGFT_TRUNC(render->rotation_angle);

    key->dwords[sizeof(key->dwords) / 4 - 1] = 0;
    key->bytes[i++] = (FT_Byte)ch;
    ch >>= 8;
    key->bytes[i++] = (FT_Byte)ch;
    ch >>= 8;
    key->bytes[i++] = (FT_Byte)ch;
    ch >>= 8;
    key->bytes[i++] = (FT_Byte)ch;
    key->bytes[i++] = (FT_Byte)render->pt_size;
    key->bytes[i++] = (FT_Byte)(render->pt_size >> 8);
    key->bytes[i++] = (FT_Byte)(render->style & style_mask);
    key->bytes[i++] = (FT_Byte)((render->style & style_mask) >> 8);
    key->bytes[i++] = (FT_Byte)(render->render_flags & rflag_mask);
    key->bytes[i++] = (FT_Byte)((render->render_flags & rflag_mask) >> 8);
    key->bytes[i++] = (FT_Byte)rot;
    key->bytes[i++] = (FT_Byte)(rot >> 8);
}

static int
equal_node_keys(const CacheNodeKey *a, const CacheNodeKey *b)
{
    int i;

    for (i = 0; i < sizeof(a->dwords) / 4; ++i)
    {
        if (a->dwords[i] != b->dwords[i])
        {
            return 0;
        }
    }
    return 1;
}

static FT_UInt32
cache_hash(const CacheNodeKey *key)
{
    /*
     * Based on the 32 bit x86 MurmurHash3, with the key size a multiple of 4.
     */

    FT_UInt32 h1 = 712189651; /* Set to the seed, a prime in this case */

    FT_UInt32 c1 = 0xCC9E2D51;
    FT_UInt32 c2 = 0x1B873593;

    FT_UInt32 k1;
    const FT_UInt32 *blocks = key->dwords - 1;

    int i;

    for (i = (sizeof(key->dwords) / 4); i; --i)
    {
        k1 = blocks[i];

        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> 17);
        k1 *= c2;

        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> 19);
        h1 = h1 * 5 + 0xE6546B64;
    }

    h1 ^= sizeof(key->dwords);
    
    h1 ^= h1 >> 16;
    h1 *= 0x85EBCA6B;
    h1 ^= h1 >> 13;
    h1 *= 0xC2B2AE35;
    h1 ^= h1 >> 16;

    return h1;
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
    cache->_debug_count = 0;
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

    /* PGFT_DEBUG_CACHE - Here is a good place to set a breakpoint
     * to examine _debug fields.
     */

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

    FT_UInt32 hash;
    FT_UInt32 bucket;
    
    set_node_key(&key, character, render);
    hash = cache_hash(&key);
    bucket = hash & cache->size_mask;
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
    cache->_debug_count--;
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

    set_node_key(&node->key, character, render);
    node->hash = cache_hash(&node->key);
    bucket = node->hash & cache->size_mask;
    node->next = cache->nodes[bucket];
    cache->nodes[bucket] = node;

    cache->depths[bucket]++;

#ifdef PGFT_DEBUG_CACHE
    cache->_debug_count++;
#endif

    return node;

    /*
     * Cleanup on error
     */
cleanup:
    _PGFT_free(node);
    return NULL;
}
