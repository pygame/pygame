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

typedef struct keyfields_ {
    GlyphIndex_t id;
    Scale_t face_size;
    unsigned short style;
    unsigned short render_flags;
    unsigned short rotation;
    FT_Fixed strength;
} KeyFields;

typedef union cachenodekey_ {
    KeyFields fields;
    FT_UInt32 dwords[(sizeof(KeyFields) + 3) / 4];
} NodeKey;

typedef struct cachenode_ {
    FontGlyph glyph;
    struct cachenode_ *next;
    NodeKey key;
    FT_UInt32 hash;
} CacheNode;

static FT_UInt32 get_hash(const NodeKey *);
static CacheNode *allocate_node(FontCache *,
                                const FontRenderMode *,
                                GlyphIndex_t, void *);
static void free_node(FontCache *, CacheNode *);
static void set_node_key(NodeKey *, GlyphIndex_t, const FontRenderMode *);
static int equal_node_keys(const NodeKey *, const NodeKey *);

const int render_flags_mask = (FT_RFLAG_ANTIALIAS |
                               FT_RFLAG_HINTED |
                               FT_RFLAG_AUTOHINT);

static void
set_node_key(NodeKey *key, GlyphIndex_t id, const FontRenderMode *mode)
{
    KeyFields *fields = &key->fields;
    const FT_UInt16 style_mask = ~(FT_STYLE_UNDERLINE);
    const FT_UInt16 rflag_mask = ~(FT_RFLAG_VERTICAL | FT_RFLAG_KERNING);
    unsigned short rot = (unsigned short)(((unsigned int)(mode->rotation_angle))>>16);

    memset(key, 0, sizeof(*key));
    fields->id = id;
    fields->face_size = mode->face_size;
    fields->style = mode->style & style_mask;
    fields->render_flags = mode->render_flags & rflag_mask;
    fields->rotation = rot;
    fields->strength = mode->strength;
}

static int
equal_node_keys(const NodeKey *a, const NodeKey *b)
{
    int i;

    for (i = 0; i < sizeof(a->dwords) / sizeof(a->dwords[0]); ++i) {
        if (a->dwords[i] != b->dwords[i]) {
            return 0;
        }
    }
    return 1;
}

static FT_UInt32
get_hash(const NodeKey *key)
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

    for (i = (sizeof(key->dwords) / 4); i; --i) {
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
_PGFT_Cache_Init(FreeTypeInstance *ft, FontCache *cache)
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
        cache->nodes[i] = 0;
    cache->depths = _PGFT_malloc((size_t)cache_size);
    if (!cache->depths) {
        _PGFT_free(cache->nodes);
        cache->nodes = 0;
        return -1;
    }
    memset(cache->depths, 0, cache_size);
    cache->free_nodes = 0;
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
_PGFT_Cache_Destroy(FontCache *cache)
{
    FT_UInt i;
    CacheNode *node, *next;

    if (!cache) {
        return;
    }

    /* PGFT_DEBUG_CACHE - Here is a good place to set a breakpoint
     * to examine _debug fields.
     */

    if (cache->nodes) {
        for (i = 0; i <= cache->size_mask; ++i) {
            node = cache->nodes[i];

            while (node) {
                next = node->next;
                free_node(cache, node);
                node = next;
            }
        }
        _PGFT_free(cache->nodes);
        cache->nodes = 0;
    }
    _PGFT_free(cache->depths);
    cache->depths = 0;
}

void
_PGFT_Cache_Cleanup(FontCache *cache)
{
    const FT_Byte MAX_BUCKET_DEPTH = 2;
    CacheNode *node, *prev;
    FT_UInt32 i;

    for (i = 0; i <= cache->size_mask; ++i) {
        while (cache->depths[i] > MAX_BUCKET_DEPTH) {
            node = cache->nodes[i];
            prev = 0;

            for (;;) {
                if (!node->next) {
#ifdef PGFT_DEBUG_CACHE
                    cache->_debug_delete_count++;
#endif

                    prev->next = 0;
                    free_node(cache, node);
                    break;
                }

                prev = node;
                node = node->next;
            }
        }
    }
}

FontGlyph *
_PGFT_Cache_FindGlyph(GlyphIndex_t id, const FontRenderMode *render,
                      FontCache *cache, void *internal)
{
    CacheNode **nodes = cache->nodes;
    CacheNode *node, *prev;
    NodeKey key;

    FT_UInt32 hash;
    FT_UInt32 bucket;

    set_node_key(&key, id, render);
    hash = get_hash(&key);
    bucket = hash & cache->size_mask;
    node = nodes[bucket];
    prev = 0;

#ifdef PGFT_DEBUG_CACHE
    cache->_debug_access++;
#endif

    while (node) {
        if (equal_node_keys(&node->key, &key)) {
            if (prev) {
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

    node = allocate_node(cache, render, id, internal);

#ifdef PGFT_DEBUG_CACHE
    cache->_debug_miss++;
#endif

    return node ? &node->glyph : 0;
}

static void
free_node(FontCache *cache, CacheNode *node)
{
    if (!node) {
        return;
    }

#ifdef PGFT_DEBUG_CACHE
    cache->_debug_count--;
#endif

    cache->depths[node->hash & cache->size_mask]--;

    FT_Done_Glyph((FT_Glyph)(node->glyph.image));
    _PGFT_free(node);
}

static CacheNode *
allocate_node(FontCache *cache, const FontRenderMode *render,
              GlyphIndex_t id, void *internal)
{
    CacheNode *node = _PGFT_malloc(sizeof(CacheNode));
    FT_UInt32 bucket;

    if (!node) {
        return 0;
    }
    memset(node, 0, sizeof(CacheNode));

    if (_PGFT_LoadGlyph(&node->glyph, id, render, internal)) {
        goto cleanup;
    }

    set_node_key(&node->key, id, render);
    node->hash = get_hash(&node->key);
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
    return 0;
}
