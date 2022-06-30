#include "simd_blitters.h"

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
pg_has_avx2()
{
#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
    return SDL_HasAVX2();
#else
    return 0;
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */
}

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
blit_blend_rgba_mul_avx2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int pre_8_width = width % 8;
    int post_8_width = (width - pre_8_width) / 8;

    __m256i *srcp256 = (__m256i *)info->s_pixels;
    __m256i *dstp256 = (__m256i *)info->d_pixels;

    __m128i mm_src, mm_dst, mm_zero, mm_two_five_fives;
    __m256i mm256_src, mm256_srcA, mm256_srcB, mm256_dst, mm256_dstA,
        mm256_dstB, mm256_shuff_mask_A, mm256_shuff_mask_B,
        mm256_two_five_fives;

    mm256_shuff_mask_A =
        _mm256_set_epi8(0x80, 23, 0x80, 22, 0x80, 21, 0x80, 20, 0x80, 19, 0x80,
                        18, 0x80, 17, 0x80, 16, 0x80, 7, 0x80, 6, 0x80, 5,
                        0x80, 4, 0x80, 3, 0x80, 2, 0x80, 1, 0x80, 0);
    mm256_shuff_mask_B =
        _mm256_set_epi8(0x80, 31, 0x80, 30, 0x80, 29, 0x80, 28, 0x80, 27, 0x80,
                        26, 0x80, 25, 0x80, 24, 0x80, 15, 0x80, 14, 0x80, 13,
                        0x80, 12, 0x80, 11, 0x80, 10, 0x80, 9, 0x80, 8);

    mm_zero = _mm_setzero_si128();
    mm_two_five_fives = _mm_set_epi64x(0x00FF00FF00FF00FF, 0x00FF00FF00FF00FF);

    mm256_two_five_fives = _mm256_set_epi8(
        0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
        0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
        0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF);

    while (height--) {
        if (pre_8_width > 0) {
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
                },
                n, pre_8_width);
        }
        srcp256 = (__m256i *)srcp;
        dstp256 = (__m256i *)dstp;
        if (post_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm256_src = _mm256_loadu_si256(srcp256);
                    mm256_dst = _mm256_loadu_si256(dstp256);

                    mm256_srcA =
                        _mm256_shuffle_epi8(mm256_src, mm256_shuff_mask_A);
                    mm256_srcB =
                        _mm256_shuffle_epi8(mm256_src, mm256_shuff_mask_B);

                    mm256_dstA =
                        _mm256_shuffle_epi8(mm256_dst, mm256_shuff_mask_A);
                    mm256_dstB =
                        _mm256_shuffle_epi8(mm256_dst, mm256_shuff_mask_B);

                    mm256_dstA = _mm256_mullo_epi16(mm256_srcA, mm256_dstA);
                    mm256_dstA =
                        _mm256_add_epi16(mm256_dstA, mm256_two_five_fives);
                    mm256_dstA = _mm256_srli_epi16(mm256_dstA, 8);

                    mm256_dstB = _mm256_mullo_epi16(mm256_srcB, mm256_dstB);
                    mm256_dstB =
                        _mm256_add_epi16(mm256_dstB, mm256_two_five_fives);
                    mm256_dstB = _mm256_srli_epi16(mm256_dstB, 8);

                    mm256_dst = _mm256_packus_epi16(mm256_dstA, mm256_dstB);
                    _mm256_storeu_si256(dstp256, mm256_dst);

                    srcp256++;
                    dstp256++;
                },
                n, post_8_width);
        }
        srcp = (Uint32 *)srcp256 + srcskip;
        dstp = (Uint32 *)dstp256 + dstskip;
    }
}
#else
void
blit_blend_rgba_mul_avx2(SDL_BlitInfo *info)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
blit_blend_rgb_mul_avx2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int pre_8_width = width % 8;
    int post_8_width = (width - pre_8_width) / 8;

    /* if either surface has a non-zero alpha mask use that as our mask */
    Uint32 amask = info->src->Amask | info->dst->Amask;

    __m256i *srcp256 = (__m256i *)info->s_pixels;
    __m256i *dstp256 = (__m256i *)info->d_pixels;

    __m128i mm_src, mm_dst, mm_zero, mm_two_five_fives, mm_alpha_mask;
    __m256i mm256_src, mm256_srcA, mm256_srcB, mm256_dst, mm256_dstA,
        mm256_dstB, mm256_shuff_mask_A, mm256_shuff_mask_B,
        mm256_two_five_fives, mm256_alpha_mask;

    mm256_shuff_mask_A =
        _mm256_set_epi8(0x80, 23, 0x80, 22, 0x80, 21, 0x80, 20, 0x80, 19, 0x80,
                        18, 0x80, 17, 0x80, 16, 0x80, 7, 0x80, 6, 0x80, 5,
                        0x80, 4, 0x80, 3, 0x80, 2, 0x80, 1, 0x80, 0);
    mm256_shuff_mask_B =
        _mm256_set_epi8(0x80, 31, 0x80, 30, 0x80, 29, 0x80, 28, 0x80, 27, 0x80,
                        26, 0x80, 25, 0x80, 24, 0x80, 15, 0x80, 14, 0x80, 13,
                        0x80, 12, 0x80, 11, 0x80, 10, 0x80, 9, 0x80, 8);

    mm_zero = _mm_setzero_si128();
    mm_two_five_fives = _mm_set_epi64x(0x00FF00FF00FF00FF, 0x00FF00FF00FF00FF);
    mm_alpha_mask = _mm_cvtsi32_si128(amask);

    mm256_two_five_fives = _mm256_set_epi8(
        0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
        0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
        0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF);

    mm256_alpha_mask = _mm256_set_epi32(amask, amask, amask, amask, amask,
                                        amask, amask, amask);

    while (height--) {
        if (pre_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    /*mm_src = 0x000000000000000000000000AARRGGBB*/
                    mm_src = _mm_or_si128(mm_src, mm_alpha_mask);
                    /* ensure source alpha is 255 */
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
                },
                n, pre_8_width);
        }
        srcp256 = (__m256i *)srcp;
        dstp256 = (__m256i *)dstp;
        if (post_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm256_src = _mm256_loadu_si256(srcp256);
                    mm256_src = _mm256_or_si256(mm256_src, mm256_alpha_mask);
                    /* ensure source alpha is 255 */
                    mm256_dst = _mm256_loadu_si256(dstp256);

                    mm256_srcA =
                        _mm256_shuffle_epi8(mm256_src, mm256_shuff_mask_A);
                    mm256_srcB =
                        _mm256_shuffle_epi8(mm256_src, mm256_shuff_mask_B);

                    mm256_dstA =
                        _mm256_shuffle_epi8(mm256_dst, mm256_shuff_mask_A);
                    mm256_dstB =
                        _mm256_shuffle_epi8(mm256_dst, mm256_shuff_mask_B);

                    mm256_dstA = _mm256_mullo_epi16(mm256_srcA, mm256_dstA);
                    mm256_dstA =
                        _mm256_add_epi16(mm256_dstA, mm256_two_five_fives);
                    mm256_dstA = _mm256_srli_epi16(mm256_dstA, 8);

                    mm256_dstB = _mm256_mullo_epi16(mm256_srcB, mm256_dstB);
                    mm256_dstB =
                        _mm256_add_epi16(mm256_dstB, mm256_two_five_fives);
                    mm256_dstB = _mm256_srli_epi16(mm256_dstB, 8);

                    mm256_dst = _mm256_packus_epi16(mm256_dstA, mm256_dstB);
                    _mm256_storeu_si256(dstp256, mm256_dst);

                    srcp256++;
                    dstp256++;
                },
                n, post_8_width);
        }
        srcp = (Uint32 *)srcp256 + srcskip;
        dstp = (Uint32 *)dstp256 + dstskip;
    }
}
#else
void
blit_blend_rgb_mul_avx2(SDL_BlitInfo *info)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
blit_blend_rgba_add_avx2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int pre_8_width = width % 8;
    int post_8_width = (width - pre_8_width) / 8;

    __m256i *srcp256 = (__m256i *)info->s_pixels;
    __m256i *dstp256 = (__m256i *)info->d_pixels;

    __m128i mm_src, mm_dst;
    __m256i mm256_src, mm256_dst;

    while (height--) {
        if (pre_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_dst = _mm_adds_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_8_width);
        }
        srcp256 = (__m256i *)srcp;
        dstp256 = (__m256i *)dstp;
        if (post_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm256_src = _mm256_loadu_si256(srcp256);
                    mm256_dst = _mm256_loadu_si256(dstp256);

                    mm256_dst = _mm256_adds_epu8(mm256_dst, mm256_src);

                    _mm256_storeu_si256(dstp256, mm256_dst);

                    srcp256++;
                    dstp256++;
                },
                n, post_8_width);
        }
        srcp = (Uint32 *)srcp256 + srcskip;
        dstp = (Uint32 *)dstp256 + dstskip;
    }
}
#else
void
blit_blend_rgba_add_avx2(SDL_BlitInfo *info)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
blit_blend_rgb_add_avx2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int pre_8_width = width % 8;
    int post_8_width = (width - pre_8_width) / 8;

    /* if either surface has a non-zero alpha mask use that as our mask */
    Uint32 amask = info->src->Amask | info->dst->Amask;

    __m256i *srcp256 = (__m256i *)info->s_pixels;
    __m256i *dstp256 = (__m256i *)info->d_pixels;

    __m128i mm_src, mm_dst, mm_alpha_mask;
    __m256i mm256_src, mm256_dst, mm256_alpha_mask;

    mm_alpha_mask = _mm_cvtsi32_si128(amask);
    mm256_alpha_mask = _mm256_set_epi32(amask, amask, amask, amask, amask,
                                        amask, amask, amask);

    while (height--) {
        if (pre_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_src = _mm_subs_epu8(mm_src, mm_alpha_mask);
                    mm_dst = _mm_adds_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_8_width);
        }
        srcp256 = (__m256i *)srcp;
        dstp256 = (__m256i *)dstp;
        if (post_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm256_src = _mm256_loadu_si256(srcp256);
                    mm256_dst = _mm256_loadu_si256(dstp256);

                    mm256_src = _mm256_subs_epu8(mm256_src, mm256_alpha_mask);
                    mm256_dst = _mm256_adds_epu8(mm256_dst, mm256_src);

                    _mm256_storeu_si256(dstp256, mm256_dst);

                    srcp256++;
                    dstp256++;
                },
                n, post_8_width);
        }
        srcp = (Uint32 *)srcp256 + srcskip;
        dstp = (Uint32 *)dstp256 + dstskip;
    }
}
#else
void
blit_blend_rgb_add_avx2(SDL_BlitInfo *info)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
blit_blend_rgba_sub_avx2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int pre_8_width = width % 8;
    int post_8_width = (width - pre_8_width) / 8;

    __m256i *srcp256 = (__m256i *)info->s_pixels;
    __m256i *dstp256 = (__m256i *)info->d_pixels;

    __m128i mm_src, mm_dst;
    __m256i mm256_src, mm256_dst;

    while (height--) {
        if (pre_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_dst = _mm_subs_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_8_width);
        }
        srcp256 = (__m256i *)srcp;
        dstp256 = (__m256i *)dstp;
        if (post_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm256_src = _mm256_loadu_si256(srcp256);
                    mm256_dst = _mm256_loadu_si256(dstp256);

                    mm256_dst = _mm256_subs_epu8(mm256_dst, mm256_src);

                    _mm256_storeu_si256(dstp256, mm256_dst);

                    srcp256++;
                    dstp256++;
                },
                n, post_8_width);
        }
        srcp = (Uint32 *)srcp256 + srcskip;
        dstp = (Uint32 *)dstp256 + dstskip;
    }
}
#else
void
blit_blend_rgba_sub_avx2(SDL_BlitInfo *info)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
blit_blend_rgb_sub_avx2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int pre_8_width = width % 8;
    int post_8_width = (width - pre_8_width) / 8;

    /* if either surface has a non-zero alpha mask use that as our mask */
    Uint32 amask = info->src->Amask | info->dst->Amask;

    __m256i *srcp256 = (__m256i *)info->s_pixels;
    __m256i *dstp256 = (__m256i *)info->d_pixels;

    __m128i mm_src, mm_dst, mm_alpha_mask;
    __m256i mm256_src, mm256_dst, mm256_alpha_mask;

    mm_alpha_mask = _mm_cvtsi32_si128(amask);
    mm256_alpha_mask = _mm256_set_epi32(amask, amask, amask, amask, amask,
                                        amask, amask, amask);

    while (height--) {
        if (pre_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_src = _mm_subs_epu8(mm_src, mm_alpha_mask);
                    mm_dst = _mm_subs_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_8_width);
        }
        srcp256 = (__m256i *)srcp;
        dstp256 = (__m256i *)dstp;
        if (post_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm256_src = _mm256_loadu_si256(srcp256);
                    mm256_dst = _mm256_loadu_si256(dstp256);

                    mm256_src = _mm256_subs_epu8(mm256_src, mm256_alpha_mask);
                    mm256_dst = _mm256_subs_epu8(mm256_dst, mm256_src);

                    _mm256_storeu_si256(dstp256, mm256_dst);

                    srcp256++;
                    dstp256++;
                },
                n, post_8_width);
        }
        srcp = (Uint32 *)srcp256 + srcskip;
        dstp = (Uint32 *)dstp256 + dstskip;
    }
}
#else
void
blit_blend_rgb_sub_avx2(SDL_BlitInfo *info)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
blit_blend_rgba_max_avx2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int pre_8_width = width % 8;
    int post_8_width = (width - pre_8_width) / 8;

    __m256i *srcp256 = (__m256i *)info->s_pixels;
    __m256i *dstp256 = (__m256i *)info->d_pixels;

    __m128i mm_src, mm_dst;
    __m256i mm256_src, mm256_dst;

    while (height--) {
        if (pre_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_dst = _mm_max_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_8_width);
        }
        srcp256 = (__m256i *)srcp;
        dstp256 = (__m256i *)dstp;
        if (post_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm256_src = _mm256_loadu_si256(srcp256);
                    mm256_dst = _mm256_loadu_si256(dstp256);

                    mm256_dst = _mm256_max_epu8(mm256_dst, mm256_src);

                    _mm256_storeu_si256(dstp256, mm256_dst);

                    srcp256++;
                    dstp256++;
                },
                n, post_8_width);
        }
        srcp = (Uint32 *)srcp256 + srcskip;
        dstp = (Uint32 *)dstp256 + dstskip;
    }
}
#else
void
blit_blend_rgba_max_avx2(SDL_BlitInfo *info)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
blit_blend_rgb_max_avx2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int pre_8_width = width % 8;
    int post_8_width = (width - pre_8_width) / 8;

    /* if either surface has a non-zero alpha mask use that as our mask */
    Uint32 amask = info->src->Amask | info->dst->Amask;

    __m256i *srcp256 = (__m256i *)info->s_pixels;
    __m256i *dstp256 = (__m256i *)info->d_pixels;

    __m128i mm_src, mm_dst, mm_alpha_mask;
    __m256i mm256_src, mm256_dst, mm256_alpha_mask;

    mm_alpha_mask = _mm_cvtsi32_si128(amask);
    mm256_alpha_mask = _mm256_set_epi32(amask, amask, amask, amask, amask,
                                        amask, amask, amask);

    while (height--) {
        if (pre_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_src = _mm_subs_epu8(mm_src, mm_alpha_mask);
                    mm_dst = _mm_max_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_8_width);
        }
        srcp256 = (__m256i *)srcp;
        dstp256 = (__m256i *)dstp;
        if (post_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm256_src = _mm256_loadu_si256(srcp256);
                    mm256_dst = _mm256_loadu_si256(dstp256);

                    mm256_src = _mm256_subs_epu8(mm256_src, mm256_alpha_mask);
                    mm256_dst = _mm256_max_epu8(mm256_dst, mm256_src);

                    _mm256_storeu_si256(dstp256, mm256_dst);

                    srcp256++;
                    dstp256++;
                },
                n, post_8_width);
        }
        srcp = (Uint32 *)srcp256 + srcskip;
        dstp = (Uint32 *)dstp256 + dstskip;
    }
}
#else
void
blit_blend_rgb_max_avx2(SDL_BlitInfo *info)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
blit_blend_rgba_min_avx2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int pre_8_width = width % 8;
    int post_8_width = (width - pre_8_width) / 8;

    __m256i *srcp256 = (__m256i *)info->s_pixels;
    __m256i *dstp256 = (__m256i *)info->d_pixels;

    __m128i mm_src, mm_dst;
    __m256i mm256_src, mm256_dst;

    while (height--) {
        if (pre_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_dst = _mm_min_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_8_width);
        }
        srcp256 = (__m256i *)srcp;
        dstp256 = (__m256i *)dstp;
        if (post_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm256_src = _mm256_loadu_si256(srcp256);
                    mm256_dst = _mm256_loadu_si256(dstp256);

                    mm256_dst = _mm256_min_epu8(mm256_dst, mm256_src);

                    _mm256_storeu_si256(dstp256, mm256_dst);

                    srcp256++;
                    dstp256++;
                },
                n, post_8_width);
        }
        srcp = (Uint32 *)srcp256 + srcskip;
        dstp = (Uint32 *)dstp256 + dstskip;
    }
}
#else
void
blit_blend_rgba_min_avx2(SDL_BlitInfo *info)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
blit_blend_rgb_min_avx2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int pre_8_width = width % 8;
    int post_8_width = (width - pre_8_width) / 8;

    /* if either surface has a non-zero alpha mask use that as our mask */
    Uint32 amask = info->src->Amask | info->dst->Amask;

    __m256i *srcp256 = (__m256i *)info->s_pixels;
    __m256i *dstp256 = (__m256i *)info->d_pixels;

    __m128i mm_src, mm_dst, mm_alpha_mask;
    __m256i mm256_src, mm256_dst, mm256_alpha_mask;

    mm_alpha_mask = _mm_cvtsi32_si128(amask);
    mm256_alpha_mask = _mm256_set_epi32(amask, amask, amask, amask, amask,
                                        amask, amask, amask);

    while (height--) {
        if (pre_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm_src = _mm_cvtsi32_si128(*srcp);
                    mm_dst = _mm_cvtsi32_si128(*dstp);

                    mm_src = _mm_adds_epu8(mm_src, mm_alpha_mask);
                    mm_dst = _mm_min_epu8(mm_dst, mm_src);

                    *dstp = _mm_cvtsi128_si32(mm_dst);

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_8_width);
        }
        srcp256 = (__m256i *)srcp;
        dstp256 = (__m256i *)dstp;
        if (post_8_width > 0) {
            LOOP_UNROLLED4(
                {
                    mm256_src = _mm256_loadu_si256(srcp256);
                    mm256_dst = _mm256_loadu_si256(dstp256);

                    mm256_src = _mm256_adds_epu8(mm256_src, mm256_alpha_mask);
                    mm256_dst = _mm256_min_epu8(mm256_dst, mm256_src);

                    _mm256_storeu_si256(dstp256, mm256_dst);

                    srcp256++;
                    dstp256++;
                },
                n, post_8_width);
        }
        srcp = (Uint32 *)srcp256 + srcskip;
        dstp = (Uint32 *)dstp256 + dstskip;
    }
}
#else
void
blit_blend_rgb_min_avx2(SDL_BlitInfo *info)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
blit_blend_premultiplied_avx2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;

    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    int srcpxskip = info->s_pxskip >> 2;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;

    int pre_8_width = width % 8;
    int post_8_width = width / 8;

    /* if either surface has a non-zero alpha mask use that as our mask */
    Uint32 amask = info->src->Amask | info->dst->Amask;

    __m256i *srcp256 = (__m256i *)info->s_pixels;
    __m256i *dstp256 = (__m256i *)info->d_pixels;

    __m128i mm_src, mm_dst, mm_zero, mm_alpha, mm_sub_dst, mm_ones;
    __m256i mm256_src, mm256_dst, mm256_shuff_mask_A, mm256_shuff_mask_B,
        mm256_src_shuff, mm256_dstA, mm256_dstB, mm256_ones, mm256_alpha,
        mm256_shuff_alpha_mask_A, mm256_shuff_alpha_mask_B;

    mm_zero = _mm_setzero_si128();
    mm_ones = _mm_set_epi8(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                           0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01);

    mm256_shuff_mask_A =
        _mm256_set_epi8(0x80, 23, 0x80, 22, 0x80, 21, 0x80, 20, 0x80, 19, 0x80,
                        18, 0x80, 17, 0x80, 16, 0x80, 7, 0x80, 6, 0x80, 5,
                        0x80, 4, 0x80, 3, 0x80, 2, 0x80, 1, 0x80, 0);

    mm256_shuff_alpha_mask_A =
        _mm256_set_epi8(0x80, 23, 0x80, 23, 0x80, 23, 0x80, 23, 0x80, 19, 0x80,
                        19, 0x80, 19, 0x80, 19, 0x80, 7, 0x80, 7, 0x80, 7,
                        0x80, 7, 0x80, 3, 0x80, 3, 0x80, 3, 0x80, 3);

    mm256_shuff_mask_B =
        _mm256_set_epi8(0x80, 31, 0x80, 30, 0x80, 29, 0x80, 28, 0x80, 27, 0x80,
                        26, 0x80, 25, 0x80, 24, 0x80, 15, 0x80, 14, 0x80, 13,
                        0x80, 12, 0x80, 11, 0x80, 10, 0x80, 9, 0x80, 8);

    mm256_shuff_alpha_mask_B =
        _mm256_set_epi8(0x80, 31, 0x80, 31, 0x80, 31, 0x80, 31, 0x80, 27, 0x80,
                        27, 0x80, 27, 0x80, 27, 0x80, 15, 0x80, 15, 0x80, 15,
                        0x80, 15, 0x80, 11, 0x80, 11, 0x80, 11, 0x80, 11);

    mm256_ones = _mm256_set_epi8(
        0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01,
        0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01,
        0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01);

    while (height--) {
        if (pre_8_width > 0) {
            /* one pixel at a time - same as current sse2 version */
            LOOP_UNROLLED4(
                {
                    Uint32 alpha = *srcp & amask;
                    if (alpha == 0) {
                        /* do nothing */
                    }
                    else if (alpha == amask) {
                        *dstp = *srcp;
                    }
                    else {
                        mm_src = _mm_cvtsi32_si128(*srcp);
                        /*mm_src = 0x000000000000000000000000AARRGGBB*/
                        mm_src = _mm_unpacklo_epi8(mm_src, mm_zero);
                        /*mm_src = 0x000000000000000000AA00RR00GG00BB*/
                        mm_dst = _mm_cvtsi32_si128(*dstp);
                        /*mm_dst = 0x000000000000000000000000AARRGGBB*/
                        mm_dst = _mm_unpacklo_epi8(mm_dst, mm_zero);
                        /*mm_dst = 0x000000000000000000AA00RR00GG00BB*/

                        mm_alpha = _mm_cvtsi32_si128(alpha);
                        /* alpha -> mm_alpha (000000000000A000) */
                        mm_alpha = _mm_srli_si128(mm_alpha, 3);
                        /* mm_alpha >> ashift -> mm_alpha(000000000000000A) */
                        mm_alpha = _mm_unpacklo_epi16(mm_alpha, mm_alpha);
                        /* 0000000000000A0A -> mm_alpha */
                        mm_alpha = _mm_unpacklo_epi32(mm_alpha, mm_alpha);
                        /* 000000000A0A0A0A -> mm_alpha2 */

                        /* pre-multiplied alpha blend */
                        mm_sub_dst = _mm_add_epi16(mm_dst, mm_ones);
                        mm_sub_dst = _mm_mullo_epi16(mm_sub_dst, mm_alpha);
                        mm_sub_dst = _mm_srli_epi16(mm_sub_dst, 8);
                        mm_dst = _mm_add_epi16(mm_src, mm_dst);
                        mm_dst = _mm_sub_epi16(mm_dst, mm_sub_dst);
                        mm_dst = _mm_packus_epi16(mm_dst, mm_zero);

                        *dstp = _mm_cvtsi128_si32(mm_dst);
                    }

                    srcp += srcpxskip;
                    dstp += dstpxskip;
                },
                n, pre_8_width);
        }
        srcp256 = (__m256i *)srcp;
        dstp256 = (__m256i *)dstp;
        if (post_8_width > 0) {
            /*8 pixels at a time, need to use shuffle to get everything
                lined up - see mul for an example*/
            LOOP_UNROLLED4(
                {
                    mm256_src = _mm256_loadu_si256(srcp256);
                    mm256_dst = _mm256_loadu_si256(dstp256);

                    /* At the start we shuffle our initial 8 bit source &
                     * destination pixel channels into a 16 bit configuration,
                     * spaced out by 00s.
                     * We need the room of 16bits to do a potential max size
                     * 8bit x 8bit multiplication (e.g. 255 x 255 = 65,025).
                     *
                     * At the same time we are working backwards from our
                     * final packing instruction that lets us squish 2 256bit
                     * registers divided into 16 16bit values into one 256bit
                     * register containing 32 8bit values (or 8 32bit pixels
                     * worth of data).
                     *
                     * This is why we end up with the strange seeming initial
                     * shuffling around of pixel channels into two 256 bit
                     * registers with 00 gaps. The goal is to be able to
                     * perform the necessary blend calculation on each channel
                     * of all 8 pixels in as few operations as possible and
                     * then leave them arranged ready to be packed back up
                     * into a single 8 x 32bit pixel chunk of data again at
                     * the end.
                     */

                    /* Do shuffle then blend to the A half first .
                     */

                    /* these shuffles prepare our source, destination and
                     * alpha only channels for 16 bit operation and puts them
                     * in the correct order for the final pack with the B half
                     */
                    mm256_dstA =
                        _mm256_shuffle_epi8(mm256_dst, mm256_shuff_mask_A);
                    /* mm256_dstA = (dst pixel 6:[00AA][00RR][00GG][00BB],
                     *               dst pixel 5:[00AA][00RR][00GG][00BB],
                     *               dst pixel 2:[00AA][00RR][00GG][00BB],
                     *               dst pixel 1:[00AA][00RR][00GG][00BB])
                     */
                    mm256_src_shuff =
                        _mm256_shuffle_epi8(mm256_src, mm256_shuff_mask_A);
                    /* mm256_src_shuff = (src pixel 6:[00AA][00RR][00GG][00BB],
                     *                    src pixel 5:[00AA][00RR][00GG][00BB],
                     *                    src pixel 2:[00AA][00RR][00GG][00BB],
                     *                    src pixel 1:[00AA][00RR][00GG][00BB])
                     */
                    mm256_alpha = _mm256_shuffle_epi8(
                        mm256_src, mm256_shuff_alpha_mask_A);
                    /* mm256_alpha = (src alpha 6:[00AA][00AA][00AA][00AA],
                     *                src alpha 5:[00AA][00AA][00AA][00AA],
                     *                src alpha 2:[00AA][00AA][00AA][00AA],
                     *                src alpha 1:[00AA][00AA][00AA][00AA])
                     */

                    /* blend on A half, at 16bit size, starts here.
                     * overall target blend (with colours and alpha represented
                     * as values between 0 and 1) is:
                     *
                     * result = source.RGB + (dest.RGB * (1 - source.A))
                     *
                     * Optimised and rearranged for values between 0 and 255
                     * the blend formula for a single colour channel is:
                     *
                     * (sC + dC - ((dC + 1) * sA >> 8))
                     */
                    mm256_src_shuff =
                        _mm256_add_epi16(mm256_src_shuff, mm256_dstA);
                    /* That was Source channels + Destination channels */
                    mm256_dstA = _mm256_add_epi16(mm256_dstA, mm256_ones);
                    /* That was Destination channels plus one */
                    mm256_dstA = _mm256_mullo_epi16(mm256_alpha, mm256_dstA);
                    /* That was each of the Destination channels plus one
                     * multiplied by the Source Alpha channels.
                     */
                    mm256_dstA = _mm256_srli_epi16(mm256_dstA, 8);
                    /* That was the right shift by 8bits on the result
                     * of the last two operations. It is, combined with the
                     * addition of the ones above, effectively
                     * a division operation by 255 for each channel putting
                     * us back in the 8bit, 0 - 255 range (but still with 00)
                     * padding taking us up to 16bit for now)
                     */

                    mm256_dstA = _mm256_sub_epi16(mm256_src_shuff, mm256_dstA);
                    /* this is the final subtraction completing the original
                     * colour channel blend formula. We now have blended
                     * channel values sitting in the same 16 bit, 00 padded
                     * arrangement of pixels as we did prior to the blend.
                     */

                    /* end of A half blend */

                    /* now do B half shuffle then blend.
                     * The register shapes are the same as for the A half but
                     * with pixels 8,7,4 & 3 instead
                     */
                    mm256_dstB =
                        _mm256_shuffle_epi8(mm256_dst, mm256_shuff_mask_B);
                    mm256_src_shuff =
                        _mm256_shuffle_epi8(mm256_src, mm256_shuff_mask_B);
                    mm256_alpha = _mm256_shuffle_epi8(
                        mm256_src, mm256_shuff_alpha_mask_B);
                    mm256_src_shuff =
                        _mm256_add_epi16(mm256_src_shuff, mm256_dstB);
                    mm256_dstB = _mm256_add_epi16(mm256_dstB, mm256_ones);
                    mm256_dstB = _mm256_mullo_epi16(mm256_alpha, mm256_dstB);
                    mm256_dstB = _mm256_srli_epi16(mm256_dstB, 8);

                    mm256_dstB = _mm256_sub_epi16(mm256_src_shuff, mm256_dstB);

                    /* now pack A & B together */
                    mm256_dst = _mm256_packus_epi16(mm256_dstA, mm256_dstB);
                    /* After this pack operation our pixels are pack in the
                     * right order - 8,7,6,5,4,3,2,1 and the right 8bit size
                     */
                    _mm256_storeu_si256(dstp256, mm256_dst);

                    srcp256++;
                    dstp256++;
                },
                n, post_8_width);
        }
        srcp = (Uint32 *)srcp256 + srcskip;
        dstp = (Uint32 *)dstp256 + dstskip;
    }
}
#else
void
blit_blend_premultiplied_avx2(SDL_BlitInfo *info)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */
