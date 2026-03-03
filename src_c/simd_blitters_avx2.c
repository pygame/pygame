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

/* just prints the first/lower 128 bits, in two chunks */
// static void
//_debug_print256_num(__m256i var, const char *msg)
//{
//    printf("%s\n", msg);
//    Uint64 *z = (Uint64 *)&var;
//    printf("l: %llX\n", *z);
//    printf("h: %llX\n", *(z + 1));
//}

/* Setup for RUN_AVX2_BLITTER */
#define SETUP_AVX2_BLITTER                                                \
    int width = info->width;                                              \
    int height = info->height;                                            \
                                                                          \
    Uint32 *srcp = (Uint32 *)info->s_pixels;                              \
    int srcskip = info->s_skip >> 2;                                      \
    int srcpxskip = info->s_pxskip >> 2;                                  \
                                                                          \
    Uint32 *dstp = (Uint32 *)info->d_pixels;                              \
    int dstskip = info->d_skip >> 2;                                      \
    int dstpxskip = info->d_pxskip >> 2;                                  \
                                                                          \
    int pre_8_width = width % 8;                                          \
    int post_8_width = width / 8;                                         \
                                                                          \
    __m256i *srcp256 = (__m256i *)info->s_pixels;                         \
    __m256i *dstp256 = (__m256i *)info->d_pixels;                         \
                                                                          \
    /* Since this operates on 8 pixels at a time, it needs to mask out */ \
    /* 0-7 pixels for partial loads/stores at periphery of blit area */   \
    __m256i _partial8_mask =                                              \
        _mm256_set_epi32(0x00, (pre_8_width > 6) ? 0x80000000 : 0x00,     \
                         (pre_8_width > 5) ? 0x80000000 : 0x00,           \
                         (pre_8_width > 4) ? 0x80000000 : 0x00,           \
                         (pre_8_width > 3) ? 0x80000000 : 0x00,           \
                         (pre_8_width > 2) ? 0x80000000 : 0x00,           \
                         (pre_8_width > 1) ? 0x80000000 : 0x00,           \
                         (pre_8_width > 0) ? 0x80000000 : 0x00);          \
                                                                          \
    __m256i pixels_src, pixels_dst;

/* Interface definition
 * Definitions needed: MACRO(SETUP_AVX2_BLITTER)
 * Input variables: None
 * Output variables: pixels_src, pixels_dst (containing raw pixel data)
 *
 * Operation: BLITTER_CODE takes pixels_src and pixels_dst and puts processed
 * results into pixels_dst
 */
#define RUN_AVX2_BLITTER(BLITTER_CODE)                                       \
    while (height--) {                                                       \
        for (int i = post_8_width; i > 0; i--) {                             \
            /* ==== load 8 pixels into AVX registers ==== */                 \
            pixels_src = _mm256_loadu_si256(srcp256);                        \
            pixels_dst = _mm256_loadu_si256(dstp256);                        \
                                                                             \
            {BLITTER_CODE}                                                   \
                                                                             \
            /* ==== store 8 pixels from AVX registers ==== */                \
            _mm256_storeu_si256(dstp256, pixels_dst);                        \
                                                                             \
            srcp256++;                                                       \
            dstp256++;                                                       \
        }                                                                    \
        srcp = (Uint32 *)srcp256;                                            \
        dstp = (Uint32 *)dstp256;                                            \
        if (pre_8_width > 0) {                                               \
            pixels_src = _mm256_maskload_epi32((int *)srcp, _partial8_mask); \
            pixels_dst = _mm256_maskload_epi32((int *)dstp, _partial8_mask); \
                                                                             \
            {BLITTER_CODE}                                                   \
                                                                             \
            /* ==== store 1-7 pixels from AVX registers ==== */              \
            _mm256_maskstore_epi32((int *)dstp, _partial8_mask, pixels_dst); \
                                                                             \
            srcp += srcpxskip * pre_8_width;                                 \
            dstp += dstpxskip * pre_8_width;                                 \
        }                                                                    \
                                                                             \
        srcp256 = (__m256i *)(srcp + srcskip);                               \
        dstp256 = (__m256i *)(dstp + dstskip);                               \
    }

/* Setup for RUN_16BIT_SHUFFLE_OUT */
#define SETUP_16BIT_SHUFFLE_OUT                                               \
    __m256i shuff_out_A =                                                     \
        _mm256_set_epi8(0x80, 23, 0x80, 22, 0x80, 21, 0x80, 20, 0x80, 19,     \
                        0x80, 18, 0x80, 17, 0x80, 16, 0x80, 7, 0x80, 6, 0x80, \
                        5, 0x80, 4, 0x80, 3, 0x80, 2, 0x80, 1, 0x80, 0);      \
                                                                              \
    __m256i shuff_out_B = _mm256_set_epi8(                                    \
        0x80, 31, 0x80, 30, 0x80, 29, 0x80, 28, 0x80, 27, 0x80, 26, 0x80, 25, \
        0x80, 24, 0x80, 15, 0x80, 14, 0x80, 13, 0x80, 12, 0x80, 11, 0x80, 10, \
        0x80, 9, 0x80, 8);                                                    \
                                                                              \
    __m256i shuff_src, shuff_dst, _shuff16_temp;

/* Interface definition
 * Definitions needed: MACRO(SETUP_16BIT_SHUFFLE_OUT)
 * Input variables: pixels_src, pixels_dst (containing raw pixel data)
 * Output variables: pixels_dst (containing processed and repacked pixel data)
 *
 * Operation: BLITTER_CODE takes shuff_src and shuff_dst and puts resulting
 * pixel data in shuff_dst
 */
#define RUN_16BIT_SHUFFLE_OUT(BLITTER_CODE)                    \
    /* ==== shuffle pixels out into two registers each, src */ \
    /* and dst set up for 16 bit math, like 0A0R0G0B ==== */   \
    shuff_src = _mm256_shuffle_epi8(pixels_src, shuff_out_A);  \
    shuff_dst = _mm256_shuffle_epi8(pixels_dst, shuff_out_A);  \
                                                               \
    {BLITTER_CODE}                                             \
                                                               \
    _shuff16_temp = shuff_dst;                                 \
                                                               \
    shuff_src = _mm256_shuffle_epi8(pixels_src, shuff_out_B);  \
    shuff_dst = _mm256_shuffle_epi8(pixels_dst, shuff_out_B);  \
                                                               \
    {BLITTER_CODE}                                             \
                                                               \
    /* ==== recombine A and B pixels ==== */                   \
    pixels_dst = _mm256_packus_epi16(_shuff16_temp, shuff_dst);

/* Interface definition
 * Input variables: "info" SDL_BlitInfo object
 * Output variables: `shuff_out_alpha` shuffle control mask.
 *
 * Takes a pixel in AVX2 registers like             [0][A][0][R][0][G][0][B]
 * (or however RGBA are aligned) and turns it into  [0][A][0][A][0][A][0][A]
 * */
#define ADD_SHUFFLE_OUT_ALPHA_CONTROL                                     \
    int _a_off = info->src->Ashift >> 2;                                  \
                                                                          \
    __m256i shuff_out_alpha = _mm256_set_epi8(                            \
        0x80, 8 + _a_off, 0x80, 8 + _a_off, 0x80, 8 + _a_off, 0x80,       \
        8 + _a_off, 0x80, 0 + _a_off, 0x80, 0 + _a_off, 0x80, 0 + _a_off, \
        0x80, 0 + _a_off, 0x80, 8 + _a_off, 0x80, 8 + _a_off, 0x80,       \
        8 + _a_off, 0x80, 8 + _a_off, 0x80, 0 + _a_off, 0x80, 0 + _a_off, \
        0x80, 0 + _a_off, 0x80, 0 + _a_off);

/* Divides each element in input mm256i by 255
 * See: https://stackoverflow.com/a/35286833/13816541 */
#define DO_AVX2_DIV255_U16(MM256I) \
    _mm256_srli_epi16(             \
        _mm256_mulhi_epu16(MM256I, _mm256_set1_epi16((short)0x8081)), 7);

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
alphablit_alpha_avx2_argb_no_surf_alpha_opaque_dst(SDL_BlitInfo *info)
{
    SETUP_AVX2_BLITTER
    SETUP_16BIT_SHUFFLE_OUT
    ADD_SHUFFLE_OUT_ALPHA_CONTROL

    __m256i src_alpha, temp;

    /* The location of the destination's missing alpha channel can be masked
     * out by composing all the other channels' masks. */
    __m256i mask_out_alpha = _mm256_set1_epi32(
        (info->dst->Rmask | info->dst->Gmask | info->dst->Bmask));

    /* Original 'Straight Alpha' blending equation:
        --------------------------------------------
        dstRGB = (srcRGB * srcA) + (dstRGB * (1-srcA))
            dstA = srcA + (dstA * (1-srcA))

        We use something slightly different to simulate
        SDL1, as follows:
        dstRGB = (((dstRGB << 8) + (srcRGB - dstRGB) * srcA + srcRGB) >> 8)
            dstA = srcA + dstA - ((srcA * dstA) / 255);
    */

    /* Alpha blend procedure in this blitter:
        --------------------------------------
        srcRGBA = 16 bit interspersed source pixels
            i.e. [0][A][0][R][0][G][0][B]
        srcA = 16 bit interspersed alpha channel
            i.e. [0][A][0][A][0][A][0][A]
        As you can see, each pixel is moved out into 64 bits of register
        for processing. This blitter loads in 8 pixels at once, but
        processes 4 at a time (4x64=256 bits).

        * In this blitter, the interspersed RGBA layout is not determined,
        because the only thing that matters is alpha location, and a mask
        is provided to select alpha out of the registers. This way the blitter
        can easily support different pixel arrangements
        (RGBA, BGRA, ARGB, ABGR)

        Order of operations:
            temp = srcRGBA - dstRGBA
            temp *= srcA
            dstRGBA <<= 8
            dstRGBA += temp
            dstRGBA += srcRGBA
            dstRGBA >>= 8
        */

    RUN_AVX2_BLITTER(
        RUN_16BIT_SHUFFLE_OUT(
            src_alpha = _mm256_shuffle_epi8(shuff_src, shuff_out_alpha);
            temp = _mm256_sub_epi16(shuff_src, shuff_dst);
            temp = _mm256_mullo_epi16(temp, src_alpha);
            shuff_dst = _mm256_slli_epi16(shuff_dst, 8);
            shuff_dst = _mm256_add_epi16(shuff_dst, temp);
            shuff_dst = _mm256_add_epi16(shuff_dst, shuff_src);
            shuff_dst = _mm256_srli_epi16(shuff_dst, 8););

        /* It seems like the destination pixel alpha shouldn't matter, since
         * it is RGBX, but it has to be zero. I suspect this is a weak spot
         * in the other blitting routines, that they need alpha 0 from these
         * surfaces. */
        pixels_dst = _mm256_and_si256(pixels_dst, mask_out_alpha);)
}
#else
void
alphablit_alpha_avx2_argb_no_surf_alpha_opaque_dst(SDL_BlitInfo *info)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
alphablit_alpha_avx2_argb_no_surf_alpha(SDL_BlitInfo *info)
{
    SETUP_AVX2_BLITTER
    SETUP_16BIT_SHUFFLE_OUT
    ADD_SHUFFLE_OUT_ALPHA_CONTROL

    // Used to choose which byte to use when pulling from the RGBX buffer and
    // the dst alpha buffer. Chooses the RGBX buffer most of the time, chooses
    // A when on the low byte of the X 16 bit region.
    // RGB buffer looks like: [0][R][0][G][0][B][0][X]
    // Dst alpha looks like:  [0][A][0][A][0][A][0][A]
    // We want:               [0][R][0][G][0][B][0][A]
    // *ofc real memory layout differs based on pixelformat.
    __m256i combine_rgba_mask =
        _mm256_set1_epi64x(1LL << (((info->src->Ashift) * 2) + 7));

    __m256i src_alpha, temp, dst_alpha, new_dst_alpha;

    /*
     dstRGB = (((dstRGB << 8) + (srcRGB - dstRGB) * srcA + srcRGB) >> 8)
     dstA = srcA + dstA - ((srcA * dstA) / 255);
     */

    RUN_AVX2_BLITTER(RUN_16BIT_SHUFFLE_OUT(
        src_alpha = _mm256_shuffle_epi8(shuff_src, shuff_out_alpha);
        dst_alpha = _mm256_shuffle_epi8(shuff_dst, shuff_out_alpha);

        // figure out alpha
        temp = _mm256_mullo_epi16(src_alpha, dst_alpha);
        temp = DO_AVX2_DIV255_U16(temp);
        new_dst_alpha = _mm256_sub_epi16(dst_alpha, temp);
        new_dst_alpha = _mm256_add_epi16(src_alpha, new_dst_alpha);

        // if preexisting dst alpha is 0, src alpha should be set to 255
        // enforces that dest alpha 0 means "copy source RGB"
        // happens after real src alpha values used to calculate dst alpha
        // compares each 16 bit block to zeroes, yielding 0xFFFF or 0x0000--
        // shifts out bottom 8 bits to get to 0x00FF or 0x0000.
        dst_alpha = _mm256_cmpeq_epi16(dst_alpha, _mm256_setzero_si256());
        dst_alpha = _mm256_srli_epi16(dst_alpha, 8);
        src_alpha = _mm256_max_epu8(dst_alpha, src_alpha);

        // figure out RGB
        temp = _mm256_sub_epi16(shuff_src, shuff_dst);
        temp = _mm256_mullo_epi16(temp, src_alpha);
        temp = _mm256_add_epi16(temp, shuff_src);
        shuff_dst = _mm256_slli_epi16(shuff_dst, 8);
        shuff_dst = _mm256_add_epi16(shuff_dst, temp);
        shuff_dst = _mm256_srli_epi16(shuff_dst, 8);

        // blend together dstRGB and dstA
        shuff_dst =
            _mm256_blendv_epi8(shuff_dst, new_dst_alpha, combine_rgba_mask);))
}
#else
void
alphablit_alpha_avx2_argb_no_surf_alpha(SDL_BlitInfo *info)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
alphablit_alpha_avx2_argb_surf_alpha(SDL_BlitInfo *info)
{
    SETUP_AVX2_BLITTER
    SETUP_16BIT_SHUFFLE_OUT
    ADD_SHUFFLE_OUT_ALPHA_CONTROL

    // Used to choose which byte to use when pulling from the RGBX buffer and
    // the dst alpha buffer. Chooses the RGBX buffer most of the time, chooses
    // A when on the low byte of the X 16 bit region.
    // RGB buffer looks like: [0][R][0][G][0][B][0][X]
    // Dst alpha looks like:  [0][A][0][A][0][A][0][A]
    // We want:               [0][R][0][G][0][B][0][A]
    // *ofc real memory layout differs based on pixelformat.
    __m256i combine_rgba_mask =
        _mm256_set1_epi64x(1LL << (((info->src->Ashift) * 2) + 7));

    __m256i src_alpha, temp, dst_alpha, new_dst_alpha;

    int dst_alpha_offset = (info->dst->Amask ? 0 : 255);

    __m256i modulate_alpha = _mm256_set1_epi16(info->src_blanket_alpha);

    /*
     dstRGB = (((dstRGB << 8) + (srcRGB - dstRGB) * srcA + srcRGB) >> 8)
     dstA = srcA + dstA - ((srcA * dstA) / 255);
     */

    RUN_AVX2_BLITTER(RUN_16BIT_SHUFFLE_OUT(
        src_alpha = _mm256_shuffle_epi8(shuff_src, shuff_out_alpha);

        // src_alpha = src_alpha * module_alpha / 255
        src_alpha = _mm256_mullo_epi16(src_alpha, modulate_alpha);
        src_alpha = _mm256_srli_epi16(
            _mm256_mulhi_epu16(src_alpha, _mm256_set1_epi16((short)0x8081)),
            7);

        dst_alpha = _mm256_shuffle_epi8(shuff_dst, shuff_out_alpha);
        // if the destination is opaque, it takes the max of each alpha
        // with 255 otherwise it takes with the max with 0. This is
        // equivalent to if opaque: alpha = 255
        dst_alpha =
            _mm256_max_epi16(dst_alpha, _mm256_set1_epi16(dst_alpha_offset));

        // figure out alpha
        temp = _mm256_mullo_epi16(src_alpha, dst_alpha);
        temp = DO_AVX2_DIV255_U16(temp);
        new_dst_alpha = _mm256_sub_epi16(dst_alpha, temp);
        new_dst_alpha = _mm256_add_epi16(src_alpha, new_dst_alpha);

        // if preexisting dst alpha is 0, src alpha should be set to 255
        // enforces that dest alpha 0 means "copy source RGB"
        // happens after real src alpha values used to calculate dst alpha
        // compares each 16 bit block to zeroes, yielding 0xFFFF or
        // 0x0000-- shifts out bottom 8 bits to get to 0x00FF or 0x0000.
        dst_alpha = _mm256_cmpeq_epi16(dst_alpha, _mm256_setzero_si256());
        dst_alpha = _mm256_srli_epi16(dst_alpha, 8);
        src_alpha = _mm256_max_epu8(dst_alpha, src_alpha);

        // figure out RGB
        temp = _mm256_sub_epi16(shuff_src, shuff_dst);
        temp = _mm256_mullo_epi16(temp, src_alpha);
        temp = _mm256_add_epi16(temp, shuff_src);
        shuff_dst = _mm256_slli_epi16(shuff_dst, 8);
        shuff_dst = _mm256_add_epi16(shuff_dst, temp);
        shuff_dst = _mm256_srli_epi16(shuff_dst, 8);

        // blend together dstRGB and dstA
        shuff_dst =
            _mm256_blendv_epi8(shuff_dst, new_dst_alpha, combine_rgba_mask);))
}
#else
void
alphablit_alpha_avx2_argb_surf_alpha(SDL_BlitInfo *info)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
          !defined(SDL_DISABLE_IMMINTRIN_H) */

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
    mm_two_five_fives = _mm_set1_epi64x(0x00FF00FF00FF00FF);
    mm256_two_five_fives = _mm256_set1_epi16(0x00FF);

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

    mm_two_five_fives = _mm_set1_epi64x(0x00FF00FF00FF00FF);
    mm256_two_five_fives = _mm256_set1_epi16(0x00FF);

    mm_alpha_mask = _mm_cvtsi32_si128(amask);
    mm256_alpha_mask = _mm256_set1_epi32(amask);

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
    mm256_alpha_mask = _mm256_set1_epi32(amask);

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
    mm256_alpha_mask = _mm256_set1_epi32(amask);

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
    mm256_alpha_mask = _mm256_set1_epi32(amask);

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
    mm256_alpha_mask = _mm256_set1_epi32(amask);

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
    /* find the index 0, 1, 2 or 3 of the alpha channel within the pixel
     * this can vary depending on the channel order in the pixel format.
     * e.g. ARGB vs RGBA or BGRA
     */
    char a_index = ((amask >> 8) == 0)    ? 0
                   : ((amask >> 16) == 0) ? 1
                   : ((amask >> 24) == 0) ? 2
                                          : 3;

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
    /* use the alpha index to eventually grab the alpha channel of each pixel
     */
    mm256_shuff_alpha_mask_A = _mm256_set_epi8(
        0x80, 20 + a_index, 0x80, 20 + a_index, 0x80, 20 + a_index, 0x80,
        20 + a_index, 0x80, 16 + a_index, 0x80, 16 + a_index, 0x80,
        16 + a_index, 0x80, 16 + a_index, 0x80, 4 + a_index, 0x80, 4 + a_index,
        0x80, 4 + a_index, 0x80, 4 + a_index, 0x80, a_index, 0x80, a_index,
        0x80, a_index, 0x80, a_index);

    mm256_shuff_mask_B =
        _mm256_set_epi8(0x80, 31, 0x80, 30, 0x80, 29, 0x80, 28, 0x80, 27, 0x80,
                        26, 0x80, 25, 0x80, 24, 0x80, 15, 0x80, 14, 0x80, 13,
                        0x80, 12, 0x80, 11, 0x80, 10, 0x80, 9, 0x80, 8);
    /* use the alpha index to eventually grab the alpha channel of each pixel
     */
    mm256_shuff_alpha_mask_B = _mm256_set_epi8(
        0x80, 28 + a_index, 0x80, 28 + a_index, 0x80, 28 + a_index, 0x80,
        28 + a_index, 0x80, 24 + a_index, 0x80, 24 + a_index, 0x80,
        24 + a_index, 0x80, 24 + a_index, 0x80, 12 + a_index, 0x80,
        12 + a_index, 0x80, 12 + a_index, 0x80, 12 + a_index, 0x80,
        8 + a_index, 0x80, 8 + a_index, 0x80, 8 + a_index, 0x80, 8 + a_index);

    mm256_ones = _mm256_set1_epi16(0x0001);

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
                     * overall target blend (with colors and alpha represented
                     * as values between 0 and 1) is:
                     *
                     * result = source.RGB + (dest.RGB * (1 - source.A))
                     *
                     * Optimised and rearranged for values between 0 and 255
                     * the blend formula for a single color channel is:
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
                     * color channel blend formula. We now have blended
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

#define PREMUL_ALPHA_CODE                                   \
    /* extract the alpha */                                 \
    mm_alpha_in = _mm256_and_si256(mm_src, mm256_amask);    \
                                                            \
    /*redistribute the alphas across the R, G, B channels*/ \
    alphaA = _mm256_shuffle_epi8(mm_src, shuffle_maskA);    \
    alphaB = _mm256_shuffle_epi8(mm_src, shuffle_maskB);    \
                                                            \
    /*prep the pixels for 16-bit math*/                     \
    mm_srcA = _mm256_unpacklo_epi8(mm_src, mm_zero);        \
    mm_srcB = _mm256_unpackhi_epi8(mm_src, mm_zero);        \
                                                            \
    mm_srcA = _mm256_add_epi16(mm_srcA, mm256_ones);        \
    mm_srcB = _mm256_add_epi16(mm_srcB, mm256_ones);        \
                                                            \
    /*multiply the pixels by the alphas*/                   \
    mm_srcA = _mm256_mullo_epi16(mm_srcA, alphaA);          \
    mm_srcB = _mm256_mullo_epi16(mm_srcB, alphaB);          \
                                                            \
    /*shift the pixels back down to 8-bit*/                 \
    mm_srcA = _mm256_srli_epi16(mm_srcA, 8);                \
    mm_srcB = _mm256_srli_epi16(mm_srcB, 8);                \
                                                            \
    /*pack the pixels back together*/                       \
    mm_dst = _mm256_packus_epi16(mm_srcA, mm_srcB);         \
    /*add the original alpha back in*/                      \
    mm_dst = _mm256_or_si256(mm_dst, mm_alpha_in);

#if defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
    !defined(SDL_DISABLE_IMMINTRIN_H)
void
premul_surf_color_by_alpha_avx2(SDL_Surface *src, PG_PixelFormat *src_format,
                                SDL_Surface *dst)
{
    int i, height = src->h;
    const int width = src->w;
    const int n_iters_8 = width / 8;
    const int pxl_excess = width % 8;
    const int src_skip = src->pitch / 4 - width;
    const int dst_skip = dst->pitch / 4 - width;
    const int src_exc_skip = pxl_excess + src_skip;
    const int dst_exc_skip = pxl_excess + dst_skip;

    Uint32 *srcp = (Uint32 *)src->pixels;
    Uint32 *dstp = (Uint32 *)dst->pixels;

    __m256i mm_src, mm_dst, alphaA, alphaB, mm_alpha_in;
    __m256i mm_srcA, mm_srcB;

    const __m256i mm256_amask = _mm256_set1_epi32(src_format->Amask);
    const __m256i mm_zero = _mm256_setzero_si256();
    const __m256i partial_mask =
        _mm256_set_epi32(0, pxl_excess > 6 ? -1 : 0, pxl_excess > 5 ? -1 : 0,
                         pxl_excess > 4 ? -1 : 0, pxl_excess > 3 ? -1 : 0,
                         pxl_excess > 2 ? -1 : 0, pxl_excess > 1 ? -1 : 0,
                         pxl_excess > 0 ? -1 : 0);
    const __m256i mm256_ones = _mm256_set1_epi16(0x0001);

    char _a_off = ((src_format->Amask >> 8) == 0)    ? 0
                  : ((src_format->Amask >> 16) == 0) ? 1
                  : ((src_format->Amask >> 24) == 0) ? 2
                                                     : 3;

    /* masks for shuffling the alpha to the RGB channels for multiplication */
    const __m256i shuffle_maskA = _mm256_set_epi8(
        -1, -1, -1, 20 + _a_off, -1, 20 + _a_off, -1, 20 + _a_off, -1, -1, -1,
        16 + _a_off, -1, 16 + _a_off, -1, 16 + _a_off, -1, -1, -1, 4 + _a_off,
        -1, 4 + _a_off, -1, 4 + _a_off, -1, -1, -1, _a_off, -1, _a_off, -1,
        _a_off);

    const __m256i shuffle_maskB = _mm256_set_epi8(
        -1, -1, -1, 28 + _a_off, -1, 28 + _a_off, -1, 28 + _a_off, -1, -1, -1,
        24 + _a_off, -1, 24 + _a_off, -1, 24 + _a_off, -1, -1, -1, 12 + _a_off,
        -1, 12 + _a_off, -1, 12 + _a_off, -1, -1, -1, 8 + _a_off, -1,
        8 + _a_off, -1, 8 + _a_off);

    while (height--) {
        /* 8 pixels at a time */
        for (i = 0; i < n_iters_8; i++) {
            mm_src = _mm256_loadu_si256((__m256i *)srcp);

            PREMUL_ALPHA_CODE;

            _mm256_storeu_si256((__m256i *)dstp, mm_dst);

            dstp += 8;
            srcp += 8;
        }

        /* up to 7 pixels at a time */
        if (pxl_excess) {
            mm_src = _mm256_maskload_epi32((int *)srcp, partial_mask);

            PREMUL_ALPHA_CODE

            _mm256_maskstore_epi32((int *)dstp, partial_mask, mm_dst);

            srcp += src_exc_skip;
            dstp += dst_exc_skip;
            continue;
        }

        srcp += src_skip;
        dstp += dst_skip;
    }
}
#else
void
premul_surf_color_by_alpha_avx2(SDL_Surface *src, PG_PixelFormat *src_format,
                                SDL_Surface *dst)
{
    BAD_AVX2_FUNCTION_CALL;
}
#endif /* defined(__AVX2__) && defined(HAVE_IMMINTRIN_H) && \
!defined(SDL_DISABLE_IMMINTRIN_H) */
