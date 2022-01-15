#define NO_PYGAME_C_API
#include "_surface.h"

#if !defined(PG_ENABLE_ARM_NEON) && defined(__aarch64__)
// arm64 has neon optimisations enabled by default, even when fpu=neon is not
// passed
#define PG_ENABLE_ARM_NEON 1
#endif

/* See if we are compiled 64 bit on GCC or MSVC */
#if _WIN32 || _WIN64
#if _WIN64
#define ENV64BIT
#endif
#endif

// Check GCC
#if __GNUC__
#if __x86_64__ || __ppc64__
#define ENV64BIT
#endif
#endif

#ifdef PG_ENABLE_ARM_NEON
// sse2neon.h is from here: https://github.com/DLTcollab/sse2neon
#include "include/sse2neon.h"
#endif /* PG_ENABLE_ARM_NEON */

#if (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON))
static void
blit_blend_rgba_mul_simd(SDL_BlitInfo *info);
#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */

static void
blit_blend_rgba_mul_simd(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;
    Uint8 *src = info->s_pixels;
    int srcpxskip = info->s_pxskip;
    int srcskip = info->s_skip;
    Uint8 *dst = info->d_pixels;
    int dstpxskip = info->d_pxskip;
    int dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int srcbpp = srcfmt->BytesPerPixel;
    int dstbpp = dstfmt->BytesPerPixel;
    Uint8 dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32 pixel;
    Uint32 tmp;
    int srcppa = info->src_blend != SDL_BLENDMODE_NONE && srcfmt->Amask;
    int dstppa = info->dst_blend != SDL_BLENDMODE_NONE && dstfmt->Amask;

    if (srcbpp == 4 && dstbpp == 4 && srcfmt->Rmask == dstfmt->Rmask &&
        srcfmt->Gmask == dstfmt->Gmask && srcfmt->Bmask == dstfmt->Bmask &&
        srcfmt->Amask == dstfmt->Amask &&
        info->src_blend != SDL_BLENDMODE_NONE) {
        int incr = srcpxskip > 0 ? 1 : -1;
        if (incr < 0) {
            src += 3;
            dst += 3;
        }
        while (height--) {
            LOOP_UNROLLED4(
                {
                    REPEAT_4({
                        tmp = ((*dst) && (*src)) ? ((*dst) * (*src)) >> 8 : 0;
                        (*dst) = (tmp <= 255 ? tmp : 255);
                        src += incr;
                        dst += incr;
                    });
                },
                n, width);
            src += srcskip;
            dst += dstskip;
        }
        return;
    }
}
