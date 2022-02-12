#include "simd_blitters.h"

#if defined(HAVE_IMMINTRIN_H) && !defined(SDL_DISABLE_IMMINTRIN_H)
#include <immintrin.h>
#endif /* defined(HAVE_IMMINTRIN_H) && !defined(SDL_DISABLE_IMMINTRIN_H) */

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
    int srceightpxskip = 8 * srcpxskip;

    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    int dstpxskip = info->d_pxskip >> 2;
    int dsteightpxskip = 8 * dstpxskip;

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
                    srcp += srceightpxskip;
                    dstp += dsteightpxskip;
                },
                n, post_8_width);
        }
        srcp += srcskip;
        dstp += dstskip;
    }
}
#else
void
blit_blend_rgba_mul_avx2(SDL_BlitInfo *info)
{
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */
