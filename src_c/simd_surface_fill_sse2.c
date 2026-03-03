#include "simd_fill.h"

#if PG_ENABLE_ARM_NEON
// sse2neon.h is from here: https://github.com/DLTcollab/sse2neon
#include "include/sse2neon.h"
#endif /* PG_ENABLE_ARM_NEON */

#define BAD_SSE2_FUNCTION_CALL                                               \
    printf(                                                                  \
        "Fatal Error: Attempted calling an SSE2 function when both compile " \
        "time and runtime support is missing. If you are seeing this "       \
        "message, you have stumbled across a pygame bug, please report it "  \
        "to the devs!");                                                     \
    PG_EXIT(1)

int
_pg_HasSSE_NEON()
{
#if defined(__SSE2__)
    return SDL_HasSSE2();
#elif PG_ENABLE_ARM_NEON
    return SDL_HasNEON();
#else
    return 0;
#endif
}

#define SETUP_SSE2_FILLER(COLOR_PROCESS_CODE)                                 \
    /* initialize surface data */                                             \
    int width = rect->w, height = rect->h;                                    \
    int skip = surface->pitch / 4 - width;                                    \
    /* indicates the number of pixels that can't be processed in 4-pixel      \
     * blocks */                                                              \
    int pxl_excess = width % 4;                                               \
    /* indicates the number of 4-pixel blocks that can be processed */        \
    int n_iters_4 = width / 4;                                                \
    int i, j;                                                                 \
    /* load pixel data */                                                     \
    Uint32 *pixels =                                                          \
        (Uint32 *)surface->pixels + rect->y * (surface->pitch / 4) + rect->x; \
                                                                              \
    __m128i mm128_dst;                                                        \
    /* prep and load the color */                                             \
    Uint32 amask = surface_format->Amask;                                     \
    if (amask) {                                                              \
        {                                                                     \
            COLOR_PROCESS_CODE                                                \
        }                                                                     \
    }                                                                         \
    __m128i mm128_color = _mm_set1_epi32(color);

#define RUN_SSE2_FILLER(FILL_CODE)                          \
    while (height--) {                                      \
        for (i = 0; i < n_iters_4; i++) {                   \
            /* load 4 pixels */                             \
            mm128_dst = _mm_loadu_si128((__m128i *)pixels); \
                                                            \
            {FILL_CODE}                                     \
                                                            \
            /* store 4 pixels */                            \
            _mm_storeu_si128((__m128i *)pixels, mm128_dst); \
                                                            \
            pixels += 4;                                    \
        }                                                   \
                                                            \
        if (pxl_excess) {                                   \
            for (j = 0; j < pxl_excess; j++, pixels++) {    \
                mm128_dst = _mm_cvtsi32_si128(*pixels);     \
                                                            \
                {FILL_CODE}                                 \
                                                            \
                    *pixels = _mm_cvtsi128_si32(mm128_dst); \
            }                                               \
        }                                                   \
        pixels += skip;                                     \
    }

/* Setup for RUN_16BIT_SHUFFLE_OUT */
#define SETUP_SHUFFLE                                                   \
    __m128i shuff_dst, _shuff16_temp, mm128_zero = _mm_setzero_si128(); \
    mm128_color = _mm_unpacklo_epi8(mm128_color, mm128_zero);

#define RUN_16BIT_SHUFFLE_OUT(FILL_CODE)                       \
    /* ==== shuffle pixels out into two registers each, src */ \
    /* and dst set up for 16 bit math, like 0A0R0G0B ==== */   \
    shuff_dst = _mm_unpacklo_epi8(mm128_dst, mm128_zero);      \
                                                               \
    {FILL_CODE}                                                \
                                                               \
    _shuff16_temp = shuff_dst;                                 \
                                                               \
    shuff_dst = _mm_unpackhi_epi8(mm128_dst, mm128_zero);      \
                                                               \
    {FILL_CODE}                                                \
                                                               \
    /* ==== recombine A and B pixels ==== */                   \
    mm128_dst = _mm_packus_epi16(_shuff16_temp, shuff_dst);

#define FILLERS(NAME, COLOR_PROCESS_CODE, FILL_CODE)                          \
    int surface_fill_blend_##NAME##_sse2(SDL_Surface *surface,                \
                                         PG_PixelFormat *surface_format,      \
                                         SDL_Rect *rect, Uint32 color)        \
    {                                                                         \
        SETUP_SSE2_FILLER(COLOR_PROCESS_CODE)                                 \
        RUN_SSE2_FILLER(FILL_CODE)                                            \
        return 0;                                                             \
    }                                                                         \
    int surface_fill_blend_rgba_##NAME##_sse2(SDL_Surface *surface,           \
                                              PG_PixelFormat *surface_format, \
                                              SDL_Rect *rect, Uint32 color)   \
    {                                                                         \
        SETUP_SSE2_FILLER({})                                                 \
        RUN_SSE2_FILLER(FILL_CODE)                                            \
        return 0;                                                             \
    }

#define FILLERS_SHUFF(NAME, COLOR_PROCESS_CODE, FILL_CODE)                    \
    int surface_fill_blend_##NAME##_sse2(SDL_Surface *surface,                \
                                         PG_PixelFormat *surface_format,      \
                                         SDL_Rect *rect, Uint32 color)        \
    {                                                                         \
        SETUP_SSE2_FILLER(COLOR_PROCESS_CODE)                                 \
        SETUP_SHUFFLE                                                         \
        RUN_SSE2_FILLER(RUN_16BIT_SHUFFLE_OUT(FILL_CODE))                     \
        return 0;                                                             \
    }                                                                         \
    int surface_fill_blend_rgba_##NAME##_sse2(SDL_Surface *surface,           \
                                              PG_PixelFormat *surface_format, \
                                              SDL_Rect *rect, Uint32 color)   \
    {                                                                         \
        SETUP_SSE2_FILLER({})                                                 \
        SETUP_SHUFFLE                                                         \
        RUN_SSE2_FILLER(RUN_16BIT_SHUFFLE_OUT(FILL_CODE))                     \
        return 0;                                                             \
    }

#define INVALID_DEFS(NAME)                                                    \
    int surface_fill_blend_##NAME##_sse2(SDL_Surface *surface,                \
                                         PG_PixelFormat *surface_format,      \
                                         SDL_Rect *rect, Uint32 color)        \
    {                                                                         \
        BAD_SSE2_FUNCTION_CALL;                                               \
        return -1;                                                            \
    }                                                                         \
    int surface_fill_blend_rgba_##NAME##_sse2(SDL_Surface *surface,           \
                                              PG_PixelFormat *surface_format, \
                                              SDL_Rect *rect, Uint32 color)   \
    {                                                                         \
        BAD_SSE2_FUNCTION_CALL;                                               \
        return -1;                                                            \
    }

#define ADD_CODE mm128_dst = _mm_adds_epu8(mm128_dst, mm128_color);
#define SUB_CODE mm128_dst = _mm_subs_epu8(mm128_dst, mm128_color);
#define MIN_CODE mm128_dst = _mm_min_epu8(mm128_dst, mm128_color);
#define MAX_CODE mm128_dst = _mm_max_epu8(mm128_dst, mm128_color);
#define MULT_CODE                                                   \
    {                                                               \
        shuff_dst = _mm_mullo_epi16(shuff_dst, mm128_color);        \
        shuff_dst = _mm_adds_epu16(shuff_dst, _mm_set1_epi16(255)); \
        shuff_dst = _mm_srli_epi16(shuff_dst, 8);                   \
    }

#if defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)
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
#endif /* defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON) */
