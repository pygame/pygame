#include "simd_transform.h"

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

/* This returns 1 when avx2 is available at runtime but support for it isn't
 * compiled in, 0 in all other cases */
int
pg_avx2_at_runtime_but_uncompiled()
{
    if (SDL_HasAVX2()) {
#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
        return 0;
#else
        return 1;
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */
    }
    return 0;
}

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
grayscale_avx2(SDL_Surface *src, PG_PixelFormat *src_fmt, SDL_Surface *newsurf)
{
    /* See the SSE2 code for a simpler overview of this algorithm
     * Current AVX2 process
     * ------------------
     * - pre loop: Load weights into register x8
     * - in loop:
     *     1. Load 8 pixels into register
     *     2. remove the alpha channel for every pixel and save it.
     *     3. multiply weights by pixels using standard shuffle to 2x 16bit
     *        register, mul + 255 then left shift. See multiply blitter mode
     *        for this operation in isolation.
     *     4. pack pixels back together from A & B while adding with a
     *        horizontal add (e.g. adds A+R and G+B in a ARGB layout)
     *     5. shift and add to make final gray pixel color in 0th
     *        8Bit channel in each 'pixel'
     *     6. shuffle again to push the gray from the 0th channel into every
     *        channel of every pixel.
     *     7. add the alpha channel back in.
     */
    int s_row_skip = (src->pitch - src->w * 4) / 4;

    // generate number of batches of pixels we need to loop through
    int pixel_batch_length = src->w * src->h;
    int num_batches = 1;
    if (s_row_skip > 0) {
        pixel_batch_length = src->w;
        num_batches = src->h;
    }

    int remaining_pixels = pixel_batch_length % 8;
    int perfect_8_pixels = pixel_batch_length / 8;

    int perfect_8_pixels_batch_counter = perfect_8_pixels;
    int remaining_pixels_batch_counter = remaining_pixels;

    Uint32 *srcp = (Uint32 *)src->pixels;
    Uint32 *dstp = (Uint32 *)newsurf->pixels;

    Uint32 amask = src_fmt->Amask;
    Uint32 rgbmask = ~amask;

    int rgb_weights = ((0x4C << src_fmt->Rshift) | (0x96 << src_fmt->Gshift) |
                       (0x1D << src_fmt->Bshift));

    __m256i *srcp256 = (__m256i *)src->pixels;
    __m256i *dstp256 = (__m256i *)newsurf->pixels;

    __m256i mm256_src, mm256_srcA, mm256_srcB, mm256_dst, mm256_dstA,
        mm256_dstB, mm256_shuff_mask_A, mm256_shuff_mask_B,
        mm256_two_five_fives, mm256_rgb_weights, mm256_shuff_mask_gray,
        mm256_alpha, mm256_rgb_mask, mm256_alpha_mask,
        mm256_shuffled_weights_A, mm256_shuffled_weights_B;

    mm256_shuff_mask_A =
        _mm256_set_epi8(0x80, 23, 0x80, 22, 0x80, 21, 0x80, 20, 0x80, 19, 0x80,
                        18, 0x80, 17, 0x80, 16, 0x80, 7, 0x80, 6, 0x80, 5,
                        0x80, 4, 0x80, 3, 0x80, 2, 0x80, 1, 0x80, 0);
    mm256_shuff_mask_B =
        _mm256_set_epi8(0x80, 31, 0x80, 30, 0x80, 29, 0x80, 28, 0x80, 27, 0x80,
                        26, 0x80, 25, 0x80, 24, 0x80, 15, 0x80, 14, 0x80, 13,
                        0x80, 12, 0x80, 11, 0x80, 10, 0x80, 9, 0x80, 8);

    mm256_shuff_mask_gray = _mm256_set_epi8(
        28, 28, 28, 28, 24, 24, 24, 24, 20, 20, 20, 20, 16, 16, 16, 16, 12, 12,
        12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);

    mm256_two_five_fives = _mm256_set1_epi16(0x00FF);
    mm256_rgb_weights = _mm256_set1_epi32(rgb_weights);
    mm256_rgb_mask = _mm256_set1_epi32(rgbmask);
    mm256_alpha_mask = _mm256_set1_epi32(amask);

    mm256_shuffled_weights_A =
        _mm256_shuffle_epi8(mm256_rgb_weights, mm256_shuff_mask_A);
    mm256_shuffled_weights_B =
        _mm256_shuffle_epi8(mm256_rgb_weights, mm256_shuff_mask_B);

    __m256i _partial8_mask = _mm256_set_epi32(
        0, (remaining_pixels > 6) ? -1 : 0, (remaining_pixels > 5) ? -1 : 0,
        (remaining_pixels > 4) ? -1 : 0, (remaining_pixels > 3) ? -1 : 0,
        (remaining_pixels > 2) ? -1 : 0, (remaining_pixels > 1) ? -1 : 0,
        (remaining_pixels > 0) ? -1 : 0);

    while (num_batches--) {
        perfect_8_pixels_batch_counter = perfect_8_pixels;
        remaining_pixels_batch_counter = remaining_pixels;
        while (perfect_8_pixels_batch_counter--) {
            mm256_src = _mm256_loadu_si256(srcp256);
            // strip out the alpha and store it
            mm256_alpha = _mm256_and_si256(mm256_src, mm256_alpha_mask);

            // shuffle out the 8 pixels into two spaced out registers
            // there are four pixels in each register with 16bits of room
            // per channel. This gives us bit space for multiplication.
            mm256_srcA = _mm256_shuffle_epi8(mm256_src, mm256_shuff_mask_A);
            mm256_srcB = _mm256_shuffle_epi8(mm256_src, mm256_shuff_mask_B);

            // Do the 'percentage multiplications' with the weights
            // with accuracy correction so values like 255 * '255'
            // (here effectively 1.0) = 255 and not 254.
            // For our grayscale this should mean 255 white stays 255 white
            // after grayscaling.
            mm256_dstA =
                _mm256_mullo_epi16(mm256_srcA, mm256_shuffled_weights_A);
            mm256_dstA = _mm256_add_epi16(mm256_dstA, mm256_two_five_fives);
            mm256_dstA = _mm256_srli_epi16(mm256_dstA, 8);

            mm256_dstB =
                _mm256_mullo_epi16(mm256_srcB, mm256_shuffled_weights_B);
            mm256_dstB = _mm256_add_epi16(mm256_dstB, mm256_two_five_fives);
            mm256_dstB = _mm256_srli_epi16(mm256_dstB, 8);

            // Add up weighted R+G+B into the first channel of each of the 8
            // pixels. This is the gray value we want in all our color
            // channels.
            mm256_dst = _mm256_hadd_epi16(mm256_dstA, mm256_dstB);
            mm256_dst =
                _mm256_add_epi16(mm256_dst, _mm256_srli_epi32(mm256_dst, 16));
            // Shuffle the gray value from the first channel of each pixel
            // into every channel of each pixel
            mm256_dst = _mm256_shuffle_epi8(mm256_dst, mm256_shuff_mask_gray);

            // Add the alpha back
            mm256_dst = _mm256_and_si256(mm256_dst, mm256_rgb_mask);
            mm256_dst = _mm256_or_si256(mm256_dst, mm256_alpha);

            _mm256_storeu_si256(dstp256, mm256_dst);

            srcp256++;
            dstp256++;
        }
        srcp = (Uint32 *)srcp256;
        dstp = (Uint32 *)dstp256;
        if (remaining_pixels_batch_counter > 0) {
            mm256_src = _mm256_maskload_epi32((int *)srcp, _partial8_mask);
            mm256_alpha = _mm256_and_si256(mm256_src, mm256_alpha_mask);

            mm256_srcA = _mm256_shuffle_epi8(mm256_src, mm256_shuff_mask_A);
            mm256_srcB = _mm256_shuffle_epi8(mm256_src, mm256_shuff_mask_B);

            mm256_dstA =
                _mm256_mullo_epi16(mm256_srcA, mm256_shuffled_weights_A);
            mm256_dstA = _mm256_add_epi16(mm256_dstA, mm256_two_five_fives);
            mm256_dstA = _mm256_srli_epi16(mm256_dstA, 8);

            mm256_dstB =
                _mm256_mullo_epi16(mm256_srcB, mm256_shuffled_weights_B);
            mm256_dstB = _mm256_add_epi16(mm256_dstB, mm256_two_five_fives);
            mm256_dstB = _mm256_srli_epi16(mm256_dstB, 8);

            mm256_dst = _mm256_hadd_epi16(mm256_dstA, mm256_dstB);
            mm256_dst =
                _mm256_add_epi16(mm256_dst, _mm256_srli_epi32(mm256_dst, 16));
            mm256_dst = _mm256_shuffle_epi8(mm256_dst, mm256_shuff_mask_gray);

            mm256_dst = _mm256_and_si256(mm256_dst, mm256_rgb_mask);
            mm256_dst = _mm256_or_si256(mm256_dst, mm256_alpha);

            _mm256_maskstore_epi32((int *)dstp, _partial8_mask, mm256_dst);

            srcp += remaining_pixels_batch_counter;
            dstp += remaining_pixels_batch_counter;
        }
        srcp += s_row_skip;
        srcp256 = (__m256i *)srcp;
    }
}

void
invert_avx2(SDL_Surface *src, PG_PixelFormat *src_fmt, SDL_Surface *newsurf)
{
    int s_row_skip = (src->pitch - src->w * 4) / 4;

    // generate number of batches of pixels we need to loop through
    int pixel_batch_length = src->w * src->h;
    int num_batches = 1;
    if (s_row_skip > 0) {
        pixel_batch_length = src->w;
        num_batches = src->h;
    }

    int remaining_pixels = pixel_batch_length % 8;
    int perfect_8_pixels = pixel_batch_length / 8;

    int perfect_8_pixels_batch_counter = perfect_8_pixels;
    int remaining_pixels_batch_counter = remaining_pixels;

    Uint32 *srcp = (Uint32 *)src->pixels;
    Uint32 *dstp = (Uint32 *)newsurf->pixels;

    Uint32 amask = src_fmt->Amask;
    Uint32 rgbmask = ~amask;

    __m256i *srcp256 = (__m256i *)src->pixels;
    __m256i *dstp256 = (__m256i *)newsurf->pixels;

    __m256i mm256_src, mm256_dst, mm256_rgb_invert_mask, mm256_alpha,
        mm256_alpha_mask;

    mm256_rgb_invert_mask = _mm256_set1_epi32(rgbmask);
    mm256_alpha_mask = _mm256_set1_epi32(amask);

    __m256i _partial8_mask = _mm256_set_epi32(
        0x00, (remaining_pixels > 6) ? -1 : 0, (remaining_pixels > 5) ? -1 : 0,
        (remaining_pixels > 4) ? -1 : 0, (remaining_pixels > 3) ? -1 : 0,
        (remaining_pixels > 2) ? -1 : 0, (remaining_pixels > 1) ? -1 : 0,
        (remaining_pixels > 0) ? -1 : 0);

    while (num_batches--) {
        perfect_8_pixels_batch_counter = perfect_8_pixels;
        remaining_pixels_batch_counter = remaining_pixels;
        while (perfect_8_pixels_batch_counter--) {
            mm256_src = _mm256_loadu_si256(srcp256);

            /* pull out the alpha */
            mm256_alpha = _mm256_and_si256(mm256_src, mm256_alpha_mask);

            /* do the invert */
            mm256_dst = _mm256_andnot_si256(mm256_src, mm256_rgb_invert_mask);

            /* put the alpha back in*/
            mm256_dst = _mm256_or_si256(mm256_dst, mm256_alpha);

            _mm256_storeu_si256(dstp256, mm256_dst);

            srcp256++;
            dstp256++;
        }
        srcp = (Uint32 *)srcp256;
        dstp = (Uint32 *)dstp256;
        if (remaining_pixels_batch_counter > 0) {
            mm256_src = _mm256_maskload_epi32((int *)srcp, _partial8_mask);

            /* pull out the alpha */
            mm256_alpha = _mm256_and_si256(mm256_src, mm256_alpha_mask);

            /* do the invert */
            mm256_dst = _mm256_andnot_si256(mm256_src, mm256_rgb_invert_mask);

            /* put the alpha back in*/
            mm256_dst = _mm256_or_si256(mm256_dst, mm256_alpha);

            _mm256_maskstore_epi32((int *)dstp, _partial8_mask, mm256_dst);

            srcp += remaining_pixels_batch_counter;
            dstp += remaining_pixels_batch_counter;
        }
        srcp += s_row_skip;
        srcp256 = (__m256i *)srcp;
    }
}
#else
void
grayscale_avx2(SDL_Surface *src, PG_PixelFormat *src_fmt, SDL_Surface *newsurf)
{
    BAD_AVX2_FUNCTION_CALL;
}
void
invert_avx2(SDL_Surface *src, PG_PixelFormat *src_fmt, SDL_Surface *newsurf)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */
