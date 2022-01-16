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


#if (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON))
static void
blit_blend_rgba_mul_simd(SDL_BlitInfo * info)
{
    int             n;
    int             width = info->width;
    int             height = info->height;

    Uint32          *srcp = (Uint32 *)info->s_pixels;
    int             srcskip = info->s_skip >> 2;
    int             srcpxskip = info->s_pxskip >> 2;

    Uint32          *dstp = (Uint32 *)info->d_pixels;
    int             dstskip = info->d_skip >> 2;
    int             dstpxskip = info->d_pxskip >> 2;

    Uint32          two_five_fives;

    __m128i mm_src, mm_dst, mm_zero, mm_two_five_fives;

    mm_zero = _mm_setzero_si128();
    two_five_fives = 0xFFFFFFFF;
    mm_two_five_fives = _mm_cvtsi32_si128(two_five_fives);
    mm_two_five_fives = _mm_unpacklo_epi8(mm_two_five_fives, mm_zero);

    while (height--)
    {
        LOOP_UNROLLED4(
        {
            mm_src = _mm_cvtsi32_si128(*srcp);
            /*mm_src = 0x000000000000000000000000AARRGGBB*/
            mm_src = _mm_unpacklo_epi8(mm_src, mm_zero);
            /*mm_src = 0x000000000000000000AA00RR00GG00BB*/
            mm_dst = _mm_cvtsi32_si128(*dstp);
            /*mm_dst = 0x000000000000000000000000AARRGGBB*/
            mm_dst = _mm_unpacklo_epi8(mm_dst, mm_zero);
            /*mm_dst = 0x000000000000000000AA00RR00GG00BB*/

            mm_dst = _mm_mullo_epi16(mm_src, mm_dst);
            /*mm_dst = 0x0000000000000000AAAARRRRGGGGBBBB*/
            mm_dst = _mm_add_epi16(mm_dst, mm_two_five_fives);
            /*mm_dst = 0x0000000000000000AAAARRRRGGGGBBBB*/
            mm_dst = _mm_srli_epi16(mm_dst, 8);
            /*mm_dst = 0x000000000000000000AA00RR00GG00BB*/
            mm_dst = _mm_packus_epi16(mm_dst, mm_dst);
            /*mm_dst = 0x00000000AARRGGBB00000000AARRGGBB*/
            *dstp = _mm_cvtsi128_si32(mm_dst);
            /*dstp = 0xAARRGGBB*/
            srcp += srcpxskip;
            dstp += dstpxskip;
        }, n, width);
        srcp += srcskip;
        dstp += dstskip;
    }

}
#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */
