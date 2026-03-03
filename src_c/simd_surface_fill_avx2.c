#include "simd_fill.h"

#if defined(HAVE_IMMINTRIN_H) && !defined(SDL_DISABLE_IMMINTRIN_H)
#include <immintrin.h>
#endif /* defined(HAVE_IMMINTRIN_H) && !defined(SDL_DISABLE_IMMINTRIN_H) */

#define BAD_AVX2_FUNCTION_CALL                                               \
    printf(                                                                  \
        "Fatal Error: Attempted calling an AVX2 function when both compile " \
        "time and runtime support is missing. If you are seeing this "       \
        "message, you have stumbled across a pygame bug, please report it "  \
        "to the devs!");                                                     \
    PG_EXIT(1)

/* helper function that does a runtime check for AVX2. It has the added
 * functionality of also returning 0 if compile time support is missing */
int
_pg_has_avx2()
{
#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
    return SDL_HasAVX2();
#else
    return 0;
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */
}

#define SETUP_AVX2_FILLER(COLOR_PROCESS_CODE)                                 \
    /* initialize surface data */                                             \
    int width = rect->w, height = rect->h;                                    \
    int skip = surface->pitch / 4 - width;                                    \
    /* indicates the number of pixels that can't be processed in 8-pixel      \
     * blocks */                                                              \
    int pxl_excess = width % 8;                                               \
    /* indicates the number of 8-pixel blocks that can be processed */        \
    int n_iters_8 = width / 8;                                                \
    int i;                                                                    \
    /* load pixel data */                                                     \
    Uint32 *pixels =                                                          \
        (Uint32 *)surface->pixels + rect->y * (surface->pitch / 4) + rect->x; \
                                                                              \
    __m256i mm256_dst;                                                        \
    __m256i mask =                                                            \
        _mm256_set_epi32(0, pxl_excess > 6 ? -1 : 0, pxl_excess > 5 ? -1 : 0, \
                         pxl_excess > 4 ? -1 : 0, pxl_excess > 3 ? -1 : 0,    \
                         pxl_excess > 2 ? -1 : 0, pxl_excess > 1 ? -1 : 0,    \
                         pxl_excess > 0 ? -1 : 0);                            \
    /* prep and load the color */                                             \
    Uint32 amask = surface_format->Amask;                                     \
    if (amask) {                                                              \
        {                                                                     \
            COLOR_PROCESS_CODE                                                \
        }                                                                     \
    }                                                                         \
    __m256i mm256_color = _mm256_set1_epi32(color);

#define RUN_AVX2_FILLER(FILL_CODE)                                  \
    while (height--) {                                              \
        for (i = 0; i < n_iters_8; i++) {                           \
            /* load 8 pixels */                                     \
            mm256_dst = _mm256_loadu_si256((__m256i *)pixels);      \
                                                                    \
            {FILL_CODE}                                             \
                                                                    \
            /* store 8 pixels */                                    \
            _mm256_storeu_si256((__m256i *)pixels, mm256_dst);      \
                                                                    \
            pixels += 8;                                            \
        }                                                           \
                                                                    \
        if (pxl_excess) {                                           \
            /* load up to 7 pixels */                               \
            mm256_dst = _mm256_maskload_epi32((int *)pixels, mask); \
                                                                    \
            {FILL_CODE}                                             \
                                                                    \
            /* store up to 7 pixels */                              \
            _mm256_maskstore_epi32((int *)pixels, mask, mm256_dst); \
                                                                    \
            pixels += pxl_excess;                                   \
        }                                                           \
                                                                    \
        pixels += skip;                                             \
    }

/* Setup for RUN_16BIT_SHUFFLE_OUT */
#define SETUP_SHUFFLE                                                      \
    __m256i shuff_dst, _shuff16_temp, mm256_zero = _mm256_setzero_si256(); \
    mm256_color = _mm256_unpacklo_epi8(mm256_color, mm256_zero);

#define RUN_16BIT_SHUFFLE_OUT(FILL_CODE)                       \
    /* ==== shuffle pixels out into two registers each, src */ \
    /* and dst set up for 16 bit math, like 0A0R0G0B ==== */   \
    shuff_dst = _mm256_unpacklo_epi8(mm256_dst, mm256_zero);   \
                                                               \
    {FILL_CODE}                                                \
                                                               \
    _shuff16_temp = shuff_dst;                                 \
                                                               \
    shuff_dst = _mm256_unpackhi_epi8(mm256_dst, mm256_zero);   \
                                                               \
    {FILL_CODE}                                                \
                                                               \
    /* ==== recombine A and B pixels ==== */                   \
    mm256_dst = _mm256_packus_epi16(_shuff16_temp, shuff_dst);

#define FILLERS(NAME, COLOR_PROCESS_CODE, FILL_CODE)                          \
    int surface_fill_blend_##NAME##_avx2(SDL_Surface *surface,                \
                                         PG_PixelFormat *surface_format,      \
                                         SDL_Rect *rect, Uint32 color)        \
    {                                                                         \
        SETUP_AVX2_FILLER(COLOR_PROCESS_CODE)                                 \
        RUN_AVX2_FILLER(FILL_CODE)                                            \
        return 0;                                                             \
    }                                                                         \
    int surface_fill_blend_rgba_##NAME##_avx2(SDL_Surface *surface,           \
                                              PG_PixelFormat *surface_format, \
                                              SDL_Rect *rect, Uint32 color)   \
    {                                                                         \
        SETUP_AVX2_FILLER({})                                                 \
        RUN_AVX2_FILLER(FILL_CODE)                                            \
        return 0;                                                             \
    }

#define FILLERS_SHUFF(NAME, COLOR_PROCESS_CODE, FILL_CODE)                    \
    int surface_fill_blend_##NAME##_avx2(SDL_Surface *surface,                \
                                         PG_PixelFormat *surface_format,      \
                                         SDL_Rect *rect, Uint32 color)        \
    {                                                                         \
        SETUP_AVX2_FILLER(COLOR_PROCESS_CODE)                                 \
        SETUP_SHUFFLE                                                         \
        RUN_AVX2_FILLER(RUN_16BIT_SHUFFLE_OUT(FILL_CODE))                     \
        return 0;                                                             \
    }                                                                         \
    int surface_fill_blend_rgba_##NAME##_avx2(SDL_Surface *surface,           \
                                              PG_PixelFormat *surface_format, \
                                              SDL_Rect *rect, Uint32 color)   \
    {                                                                         \
        SETUP_AVX2_FILLER({})                                                 \
        SETUP_SHUFFLE                                                         \
        RUN_AVX2_FILLER(RUN_16BIT_SHUFFLE_OUT(FILL_CODE))                     \
        return 0;                                                             \
    }

#define INVALID_DEFS(NAME)                                                    \
    int surface_fill_blend_##NAME##_avx2(SDL_Surface *surface,                \
                                         PG_PixelFormat *surface_format,      \
                                         SDL_Rect *rect, Uint32 color)        \
    {                                                                         \
        BAD_AVX2_FUNCTION_CALL;                                               \
        return -1;                                                            \
    }                                                                         \
    int surface_fill_blend_rgba_##NAME##_avx2(SDL_Surface *surface,           \
                                              PG_PixelFormat *surface_format, \
                                              SDL_Rect *rect, Uint32 color)   \
    {                                                                         \
        BAD_AVX2_FUNCTION_CALL;                                               \
        return -1;                                                            \
    }

#define ADD_CODE mm256_dst = _mm256_adds_epu8(mm256_dst, mm256_color);
#define SUB_CODE mm256_dst = _mm256_subs_epu8(mm256_dst, mm256_color);
#define MIN_CODE mm256_dst = _mm256_min_epu8(mm256_dst, mm256_color);
#define MAX_CODE mm256_dst = _mm256_max_epu8(mm256_dst, mm256_color);
#define MULT_CODE                                                         \
    {                                                                     \
        shuff_dst = _mm256_mullo_epi16(shuff_dst, mm256_color);           \
        shuff_dst = _mm256_adds_epu16(shuff_dst, _mm256_set1_epi16(255)); \
        shuff_dst = _mm256_srli_epi16(shuff_dst, 8);                      \
    }

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
FILLERS(add, color &= ~amask;, ADD_CODE)
FILLERS(sub, color &= ~amask;, SUB_CODE)
FILLERS(min, color |= amask;, MIN_CODE)
FILLERS(max, color &= ~amask;, MAX_CODE)
FILLERS_SHUFF(mult, color |= amask;, MULT_CODE)
#else
INVALID_DEFS(add)
INVALID_DEFS(sub)
INVALID_DEFS(min)
INVALID_DEFS(max)
INVALID_DEFS(mult)
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
 !defined(SDL_DISABLE_IMMINTRIN_H) */
