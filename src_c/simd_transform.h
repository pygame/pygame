#define NO_PYGAME_C_API
#include "_surface.h"

/**
 * MACRO borrowed from SSE2NEON - useful for making the shuffling family of
 * intrinsics easier to understand by indicating clearly what will go where.
 *
 * SSE2Neon description follows...
 * MACRO for shuffle parameter for _mm_shuffle_ps().
 * Argument fp3 is a digit[0123] that represents the fp from argument "b"
 * of mm_shuffle_ps that will be placed in fp3 of result. fp2 is the same
 * for fp2 in result. fp1 is a digit[0123] that represents the fp from
 * argument "a" of mm_shuffle_ps that will be places in fp1 of result.
 * fp0 is the same for fp0 of result.
 */
#define _PG_SIMD_SHUFFLE(fp3, fp2, fp1, fp0) \
    (((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | ((fp0)))

#if !defined(PG_ENABLE_ARM_NEON) && defined(__aarch64__)
// arm64 has neon optimisations enabled by default, even when fpu=neon is not
// passed
#define PG_ENABLE_ARM_NEON 1
#endif

// SSE2 functions
#if defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)

void
grayscale_sse2(SDL_Surface *src, PG_PixelFormat *src_fmt,
               SDL_Surface *newsurf);
// smoothscale filters
void
filter_shrink_X_SSE2(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch,
                     int dstpitch, int srcwidth, int dstwidth);
void
filter_shrink_Y_SSE2(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch,
                     int dstpitch, int srcheight, int dstheight);
void
filter_expand_X_SSE2(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch,
                     int dstpitch, int srcwidth, int dstwidth);
void
filter_expand_Y_SSE2(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch,
                     int dstpitch, int srcheight, int dstheight);
void
invert_sse2(SDL_Surface *src, PG_PixelFormat *src_fmt, SDL_Surface *newsurf);

#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */

// AVX2 functions
void
grayscale_avx2(SDL_Surface *src, PG_PixelFormat *src_fmt,
               SDL_Surface *newsurf);
void
invert_avx2(SDL_Surface *src, PG_PixelFormat *src_fmt, SDL_Surface *newsurf);
