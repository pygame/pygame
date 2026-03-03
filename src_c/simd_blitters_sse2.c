#include "simd_blitters.h"

#if PG_ENABLE_ARM_NEON
// sse2neon.h is from here: https://github.com/DLTcollab/sse2neon
#include "include/sse2neon.h"
#endif /* PG_ENABLE_ARM_NEON */

/* See if we are compiled 64 bit on GCC or MSVC */
#if _WIN32 || _WIN64
#if _WIN64
#define ENV64BIT
#endif
#endif

// Check GCC
#if __GNUC__
#if __x86_64__ || __ppc64__ || __aarch64__
#define ENV64BIT
#endif
#endif

/* This returns 1 when sse2 is available at runtime but support for it isn't
 * compiled in, 0 in all other cases */
int
pg_sse2_at_runtime_but_uncompiled()
{
    if (SDL_HasSSE2()) {
#ifdef __SSE2__
        return 0;
#else
        return 1;
#endif /* __SSE2__ */
    }
    return 0;
}

/* This returns 1 when neon is available at runtime but support for it isn't
 * compiled in, 0 in all other cases */
int
pg_neon_at_runtime_but_uncompiled()
{
    if (SDL_HasNEON()) {
#if PG_ENABLE_ARM_NEON
        return 0;
#else
        return 1;
#endif /* PG_ENABLE_ARM_NEON */
    }
    return 0;
}

#if (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON))
/* See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=32869
 * These are both the "movq" instruction, but apparently we need to use the
 * load low one on 32 bit. See:
 * https://github.com/gcc-mirror/gcc/blob/master/gcc/config/i386/emmintrin.h
 * According to the intel intrinsics guide the instructions this uses on 32
 * bit are slightly slower
 * ARGS: reg is a pointer to an m128, num is a pointer to a 64 bit integer */
#if defined(ENV64BIT)
#define LOAD_64_INTO_M128(num, reg) *reg = _mm_cvtsi64_si128(*num)
#define STORE_M128_INTO_64(reg, num) *num = _mm_cvtsi128_si64(*reg)
#else
#define LOAD_64_INTO_M128(num, reg) \
    *reg = _mm_loadl_epi64((const __m128i *)num)
#define STORE_M128_INTO_64(reg, num) _mm_storel_epi64((__m128i *)num, *reg)
#endif
#endif

#define SETUP_SSE2_BLITTER                        \
    int i, n;                                     \
    int width = info->width;                      \
    int height = info->height;                    \
                                                  \
    Uint32 *srcp = (Uint32 *)info->s_pixels;      \
    Uint32 *dstp = (Uint32 *)info->d_pixels;      \
    __m128i *srcp128 = (__m128i *)info->s_pixels; \
    __m128i *dstp128 = (__m128i *)info->d_pixels; \
                                                  \
    const int srcskip = info->s_skip / 4;         \
    const int dstskip = info->d_skip / 4;         \
    const int pxl_excess = width % 4;             \
    const int n_iters_4 = width / 4;              \
                                                  \
    __m128i mm128_src, mm128_dst;

#define SETUP_SSE2_16BIT_SHUFFLE_OUT                \
    const __m128i mm128_zero = _mm_setzero_si128(); \
    __m128i shuff_src, shuff_dst, _shuff16_temp;

#define RUN_SSE2_16BIT_SHUFFLE_OUT(BLITTER_CODE)               \
    /* ==== shuffle pixels out into two registers each, src */ \
    /* and dst set up for 16 bit math, like 0A0R0G0B ==== */   \
    shuff_src = _mm_unpacklo_epi8(mm128_src, mm128_zero);      \
    shuff_dst = _mm_unpacklo_epi8(mm128_dst, mm128_zero);      \
                                                               \
    {BLITTER_CODE}                                             \
                                                               \
    _shuff16_temp = shuff_dst;                                 \
                                                               \
    shuff_src = _mm_unpackhi_epi8(mm128_src, mm128_zero);      \
    shuff_dst = _mm_unpackhi_epi8(mm128_dst, mm128_zero);      \
                                                               \
    {BLITTER_CODE}                                             \
                                                               \
    /* ==== recombine A and B pixels ==== */                   \
    mm128_dst = _mm_packus_epi16(_shuff16_temp, shuff_dst);

#define RUN_SSE2_BLITTER(BLITTER_CODE)                    \
    while (height--) {                                    \
        for (i = 0; i < pxl_excess; i++) {                \
            mm128_src = _mm_cvtsi32_si128(*srcp);         \
            mm128_dst = _mm_cvtsi32_si128(*dstp);         \
                                                          \
            {BLITTER_CODE}                                \
                                                          \
                *dstp = _mm_cvtsi128_si32(mm128_dst);     \
            srcp++;                                       \
            dstp++;                                       \
        }                                                 \
                                                          \
        srcp128 = (__m128i *)srcp;                        \
        dstp128 = (__m128i *)dstp;                        \
        if (n_iters_4) {                                  \
            LOOP_UNROLLED4(                               \
                {                                         \
                    mm128_src = _mm_loadu_si128(srcp128); \
                    mm128_dst = _mm_loadu_si128(dstp128); \
                                                          \
                    {BLITTER_CODE}                        \
                                                          \
                    _mm_storeu_si128(dstp128, mm128_dst); \
                    srcp128++;                            \
                    dstp128++;                            \
                },                                        \
                n, n_iters_4);                            \
        }                                                 \
        srcp = (Uint32 *)srcp128 + srcskip;               \
        dstp = (Uint32 *)dstp128 + dstskip;               \
    }

#if defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)
void
alphablit_alpha_sse2_argb_surf_alpha(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;
    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;

    PG_PixelFormat *srcfmt = info->src;
    PG_PixelFormat *dstfmt = info->dst;

    // int srcbpp = PG_FORMAT_BytesPerPixel(srcfmt);
    // int dstbpp = PG_FORMAT_BytesPerPixel(dstfmt);

    Uint32 dst_amask = dstfmt->Amask;
    Uint32 src_amask = srcfmt->Amask;

    int dst_opaque = (dst_amask ? 0 : 255);

    Uint32 modulateA = info->src_blanket_alpha;

    Uint64 rgb_mask;

    __m128i src1, dst1, sub_dst, mm_src_alpha;
    __m128i rgb_src_alpha, mm_zero;
    __m128i mm_dst_alpha, mm_sub_alpha, rgb_mask_128;

    mm_zero = _mm_setzero_si128();

    rgb_mask = 0x0000000000FFFFFF;  // 0F0F0F0F
    rgb_mask_128 = _mm_loadl_epi64((const __m128i *)&rgb_mask);

    /* Original 'Straight Alpha' blending equation:
       --------------------------------------------
       dstRGB = (srcRGB * srcA) + (dstRGB * (1-srcA))
         dstA = srcA + (dstA * (1-srcA))

       We use something slightly different to simulate
       SDL1, as follows:
       dstRGB = (((dstRGB << 8) + (srcRGB - dstRGB) * srcA + srcRGB) >> 8)
         dstA = srcA + dstA - ((srcA * dstA) / 255);
                                                        */

    while (height--) {
        LOOP_UNROLLED4(
            {
                Uint32 src_alpha = (*srcp & src_amask);
                Uint32 dst_alpha = (*dstp & dst_amask) + dst_opaque;
                /* modulate src_alpha - need to do it here for
                   accurate testing */
                src_alpha = src_alpha >> 24;
                src_alpha = (src_alpha * modulateA) / 255;
                src_alpha = src_alpha << 24;

                if ((src_alpha == src_amask) || (dst_alpha == 0)) {
                    /* 255 src alpha or 0 dst alpha
                       So copy src pixel over dst pixel, also copy
                       modulated alpha */
                    *dstp = (*srcp & 0x00FFFFFF) | src_alpha;
                }
                else {
                    /* Do the actual blend */
                    /* src_alpha -> mm_src_alpha (000000000000A000) */
                    mm_src_alpha = _mm_cvtsi32_si128(src_alpha);
                    /* mm_src_alpha >> ashift ->
                     * rgb_src_alpha(000000000000000A) */
                    mm_src_alpha = _mm_srli_si128(mm_src_alpha, 3);

                    /* dst_alpha -> mm_dst_alpha (000000000000A000) */
                    mm_dst_alpha = _mm_cvtsi32_si128(dst_alpha);
                    /* mm_src_alpha >> ashift ->
                     * rgb_src_alpha(000000000000000A) */
                    mm_dst_alpha = _mm_srli_si128(mm_dst_alpha, 3);

                    /* Calc alpha first */

                    /* (srcA * dstA) */
                    mm_sub_alpha = _mm_mullo_epi16(mm_src_alpha, mm_dst_alpha);
                    /* (srcA * dstA) / 255 */
                    mm_sub_alpha = _mm_srli_epi16(
                        _mm_mulhi_epu16(mm_sub_alpha,
                                        _mm_set1_epi16((short)0x8081)),
                        7);
                    /* srcA + dstA */
                    mm_dst_alpha = _mm_add_epi16(mm_src_alpha, mm_dst_alpha);
                    /* srcA + dstA - ((srcA * dstA) / 255); */
                    mm_dst_alpha = _mm_slli_si128(
                        _mm_sub_epi16(mm_dst_alpha, mm_sub_alpha), 3);

                    /* Then Calc RGB */
                    /* 0000000000000A0A -> rgb_src_alpha */
                    rgb_src_alpha =
                        _mm_unpacklo_epi16(mm_src_alpha, mm_src_alpha);
                    /* 000000000A0A0A0A -> rgb_src_alpha */
                    rgb_src_alpha =
                        _mm_unpacklo_epi32(rgb_src_alpha, rgb_src_alpha);

                    /* src(ARGB) -> src1 (000000000000ARGB) */
                    src1 = _mm_cvtsi32_si128(*srcp);
                    /* 000000000A0R0G0B -> src1 */
                    src1 = _mm_unpacklo_epi8(src1, mm_zero);

                    /* dst(ARGB) -> dst1 (000000000000ARGB) */
                    dst1 = _mm_cvtsi32_si128(*dstp);
                    /* 000000000A0R0G0B -> dst1 */
                    dst1 = _mm_unpacklo_epi8(dst1, mm_zero);

                    /* (srcRGB - dstRGB) */
                    sub_dst = _mm_sub_epi16(src1, dst1);

                    /* (srcRGB - dstRGB) * srcA */
                    sub_dst = _mm_mullo_epi16(sub_dst, rgb_src_alpha);

                    /* (srcRGB - dstRGB) * srcA + srcRGB */
                    sub_dst = _mm_add_epi16(sub_dst, src1);

                    /* (dstRGB << 8) */
                    dst1 = _mm_slli_epi16(dst1, 8);

                    /* ((dstRGB << 8) + (srcRGB - dstRGB) * srcA + srcRGB) */
                    sub_dst = _mm_add_epi16(sub_dst, dst1);

                    /* (((dstRGB << 8) + (srcRGB - dstRGB) * srcA + srcRGB) >>
                     * 8)*/
                    sub_dst = _mm_srli_epi16(sub_dst, 8);

                    /* pack everything back into a pixel */
                    sub_dst = _mm_packus_epi16(sub_dst, mm_zero);
                    sub_dst = _mm_and_si128(sub_dst, rgb_mask_128);
                    /* add alpha to RGB */
                    sub_dst = _mm_add_epi16(mm_dst_alpha, sub_dst);
                    *dstp = _mm_cvtsi128_si32(sub_dst);
                }
                ++srcp;
                ++dstp;
            },
            n, width);
        srcp += srcskip;
        dstp += dstskip;
    }
}

void
alphablit_alpha_sse2_argb_no_surf_alpha(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;
    int srcskip = info->s_skip >> 2;
    int dstskip = info->d_skip >> 2;
    PG_PixelFormat *srcfmt = info->src;
    PG_PixelFormat *dstfmt = info->dst;

    /* Original 'Straight Alpha' blending equation:
       --------------------------------------------
       dstRGB = (srcRGB * srcA) + (dstRGB * (1-srcA))
         dstA = srcA + (dstA * (1-srcA))

       We use something slightly different to simulate
       SDL1, as follows:
       dstRGB = (((dstRGB << 8) + (srcRGB - dstRGB) * srcA + srcRGB) >> 8)
         dstA = srcA + dstA - ((srcA * dstA) / 255);
    */

    /* There are two paths through this blitter:
        1. Two pixels at once.
        2. One pixel at a time.
    */

    Uint64 *srcp64 = (Uint64 *)info->s_pixels;
    Uint64 *dstp64 = (Uint64 *)info->d_pixels;
    Uint64 src_amask64 = ((Uint64)srcfmt->Amask << 32) | srcfmt->Amask;

    Uint64 rgb_mask = 0x00FFFFFF00FFFFFF;
    Uint64 offset_rgb_mask = 0xFF00FFFFFF00FFFF;

    Uint32 *srcp32 = (Uint32 *)info->s_pixels;
    Uint32 *dstp32 = (Uint32 *)info->d_pixels;
    Uint32 src_amask32 = srcfmt->Amask;
    Uint32 dst_amask32 = dstfmt->Amask;

    __m128i src1, dst1, temp, sub_dst;
    __m128i temp2, mm_src_alpha, mm_dst_alpha, mm_sub_alpha;
    __m128i mm_alpha_mask, mm_zero, rgb_mask_128, offset_rgb_mask_128,
        alpha_mask_128;

    if (((width % 2) == 0) && ((srcskip % 2) == 0) && ((dstskip % 2) == 0)) {
        width = width / 2;
        srcskip = srcskip / 2;
        dstskip = dstskip / 2;

        mm_zero = _mm_setzero_si128();
        mm_alpha_mask = _mm_cvtsi32_si128(0x00FF00FF);

        /* two pixels at a time --
         * only works when blit width is an even number */
        LOAD_64_INTO_M128(&rgb_mask, &rgb_mask_128);
        LOAD_64_INTO_M128(&offset_rgb_mask, &offset_rgb_mask_128);
        LOAD_64_INTO_M128(&src_amask64, &alpha_mask_128);

        while (height--) {
            LOOP_UNROLLED4(
                {
                    /* load the pixels into SSE registers */
                    /* src(ARGB) -> src1 (00000000ARGBARGB) */
                    LOAD_64_INTO_M128(srcp64, &src1);
                    /* dst(ARGB) -> dst1 (00000000ARGBARGB) */
                    LOAD_64_INTO_M128(dstp64, &dst1);
                    /* src_alpha -> mm_src_alpha (00000000A000A000) */
                    mm_src_alpha = _mm_and_si128(src1, alpha_mask_128);
                    /* dst_alpha -> mm_dst_alpha (00000000A000A000) */
                    mm_dst_alpha = _mm_and_si128(dst1, alpha_mask_128);

                    /* Do the actual blend */

                    /* mm_src_alpha >> ashift ->
                     * rgb_src_alpha(000000000A000A00) */
                    mm_src_alpha = _mm_srli_si128(mm_src_alpha, 1);

                    /* mm_src_alpha >> ashift ->
                     * rgb_src_alpha(000000000A000A00) */
                    mm_dst_alpha = _mm_srli_si128(mm_dst_alpha, 1);
                    /* this makes sure we copy across src RGB data when dst is
                     * 0*/
                    temp2 = _mm_cmpeq_epi8(mm_dst_alpha, offset_rgb_mask_128);
                    /* Calc alpha first */

                    /* (srcA * dstA) */
                    temp = _mm_mullo_epi16(mm_src_alpha, mm_dst_alpha);

                    /* (srcA * dstA) / 255 */
                    temp = _mm_srli_epi16(
                        _mm_mulhi_epu16(temp, _mm_set1_epi16((short)0x8081)),
                        7);
                    /* srcA + dstA - ((srcA * dstA) / 255); */
                    mm_dst_alpha = _mm_sub_epi16(mm_dst_alpha, temp);
                    mm_dst_alpha = _mm_add_epi16(mm_src_alpha, mm_dst_alpha);
                    mm_dst_alpha = _mm_slli_si128(mm_dst_alpha, 1);

                    /* this makes sure we copy across src RGB data when dst is
                     * 0*/
                    mm_src_alpha = _mm_or_si128(mm_src_alpha, temp2);
                    // Create squashed src alpha
                    mm_src_alpha = _mm_add_epi16(
                        _mm_and_si128(_mm_srli_si128(mm_src_alpha, 2),
                                      mm_alpha_mask),
                        _mm_and_si128(_mm_srli_si128(mm_src_alpha, 4),
                                      mm_alpha_mask));

                    /* Then Calc RGB */
                    /* 0000000000000A0A -> mm_src_alpha */

                    mm_src_alpha =
                        _mm_unpacklo_epi16(mm_src_alpha, mm_src_alpha);
                    /* 000000000A0A0A0A -> rgb_src_alpha */
                    mm_src_alpha =
                        _mm_unpacklo_epi32(mm_src_alpha, mm_src_alpha);

                    /* 000000000A0R0G0B -> src1 */
                    src1 = _mm_unpacklo_epi8(src1, mm_zero);

                    /* 000000000A0R0G0B -> dst1 */
                    dst1 = _mm_unpacklo_epi8(dst1, mm_zero);

                    /* (srcRGB - dstRGB) */
                    temp = _mm_sub_epi16(src1, dst1);

                    /* (srcRGB - dstRGB) * srcA */
                    temp = _mm_mullo_epi16(temp, mm_src_alpha);

                    /* (srcRGB - dstRGB) * srcA + srcRGB */
                    temp = _mm_add_epi16(temp, src1);

                    /* (dstRGB << 8) */
                    dst1 = _mm_slli_epi16(dst1, 8);

                    /* ((dstRGB << 8) + (srcRGB - dstRGB) * srcA + srcRGB) */
                    temp = _mm_add_epi16(temp, dst1);

                    /* (((dstRGB << 8) + (srcRGB - dstRGB) * srcA + srcRGB) >>
                     * 8)*/
                    temp = _mm_srli_epi16(temp, 8);

                    /* pack everything back into a pixel */
                    temp = _mm_packus_epi16(temp, mm_zero);
                    temp = _mm_and_si128(temp, rgb_mask_128);
                    /* add alpha to RGB */
                    temp = _mm_add_epi16(mm_dst_alpha, temp);
                    STORE_M128_INTO_64(&temp, dstp64);

                    ++srcp64;
                    ++dstp64;
                },
                n, width);
            srcp64 += srcskip;
            dstp64 += dstskip;
        }
    }
    else {
        /* one pixel at a time */
        mm_zero = _mm_setzero_si128();
        rgb_mask_128 = _mm_cvtsi32_si128(0x00FFFFFF);

        while (height--) {
            LOOP_UNROLLED4(
                {
                    Uint32 src_alpha = (*srcp32 & src_amask32);
                    Uint32 dst_alpha = (*dstp32 & dst_amask32);
                    if ((src_alpha == src_amask32) || (dst_alpha == 0)) {
                        /* 255 src alpha or 0 dst alpha
                           So just copy src pixel over dst pixel*/
                        *dstp32 = *srcp32;
                    }
                    else {
                        /* Do the actual blend */
                        /* src_alpha -> mm_src_alpha (000000000000A000) */
                        mm_src_alpha = _mm_cvtsi32_si128(src_alpha);
                        /* mm_src_alpha >> ashift ->
                         * rgb_src_alpha(000000000000000A) */
                        mm_src_alpha = _mm_srli_si128(mm_src_alpha, 3);

                        /* dst_alpha -> mm_dst_alpha (000000000000A000) */
                        mm_dst_alpha = _mm_cvtsi32_si128(dst_alpha);
                        /* mm_src_alpha >> ashift ->
                         * rgb_src_alpha(000000000000000A) */
                        mm_dst_alpha = _mm_srli_si128(mm_dst_alpha, 3);

                        /* Calc alpha first */

                        /* (srcA * dstA) */
                        mm_sub_alpha =
                            _mm_mullo_epi16(mm_src_alpha, mm_dst_alpha);

                        /* (srcA * dstA) / 255 */
                        mm_sub_alpha = _mm_srli_epi16(
                            _mm_mulhi_epu16(mm_sub_alpha,
                                            _mm_set1_epi16((short)0x8081)),
                            7);
                        /* srcA + dstA - ((srcA * dstA) / 255); */
                        mm_dst_alpha =
                            _mm_sub_epi16(mm_dst_alpha, mm_sub_alpha);
                        mm_dst_alpha =
                            _mm_add_epi16(mm_src_alpha, mm_dst_alpha);
                        mm_dst_alpha = _mm_slli_si128(mm_dst_alpha, 3);

                        /* Then Calc RGB */
                        /* 0000000000000A0A -> rgb_src_alpha */
                        mm_src_alpha =
                            _mm_unpacklo_epi16(mm_src_alpha, mm_src_alpha);
                        /* 000000000A0A0A0A -> rgb_src_alpha */
                        mm_src_alpha =
                            _mm_unpacklo_epi32(mm_src_alpha, mm_src_alpha);

                        /* src(ARGB) -> src1 (000000000000ARGB) */
                        src1 = _mm_cvtsi32_si128(*srcp32);
                        /* 000000000A0R0G0B -> src1 */
                        src1 = _mm_unpacklo_epi8(src1, mm_zero);

                        /* dst(ARGB) -> dst1 (000000000000ARGB) */
                        dst1 = _mm_cvtsi32_si128(*dstp32);
                        /* 000000000A0R0G0B -> dst1 */
                        dst1 = _mm_unpacklo_epi8(dst1, mm_zero);

                        /* (srcRGB - dstRGB) */
                        sub_dst = _mm_sub_epi16(src1, dst1);

                        /* (srcRGB - dstRGB) * srcA */
                        sub_dst = _mm_mullo_epi16(sub_dst, mm_src_alpha);

                        /* (srcRGB - dstRGB) * srcA + srcRGB */
                        sub_dst = _mm_add_epi16(sub_dst, src1);

                        /* (dstRGB << 8) */
                        dst1 = _mm_slli_epi16(dst1, 8);

                        /* ((dstRGB << 8) + (srcRGB - dstRGB) * srcA + srcRGB)
                         */
                        sub_dst = _mm_add_epi16(sub_dst, dst1);

                        /* (((dstRGB << 8) + (srcRGB - dstRGB) * srcA + srcRGB)
                         * >> 8)*/
                        sub_dst = _mm_srli_epi16(sub_dst, 8);

                        /* pack everything back into a pixel */
                        sub_dst = _mm_packus_epi16(sub_dst, mm_zero);
                        sub_dst = _mm_and_si128(sub_dst, rgb_mask_128);
                        /* add alpha to RGB */
                        sub_dst = _mm_add_epi16(mm_dst_alpha, sub_dst);
                        *dstp32 = _mm_cvtsi128_si32(sub_dst);
                    }
                    ++srcp32;
                    ++dstp32;
                },
                n, width);
            srcp32 += srcskip;
            dstp32 += dstskip;
        }
    }
}

/* Defines the blit procedure at the core of
 * alphablit_alpha_sse2_argb_no_surf_alpha_opaque_dst
 *
 * Input variables: src1, dst1, unpacked_alpha
 *       containing unpacked 16 bit lanes of src, dst, and src alpha
 * Output variables: sub_dst
 * */
#define ARGB_NO_SURF_ALPHA_OPAQUE_DST_PROCEDURE                      \
    /* (srcRGB - dstRGB) */                                          \
    sub_dst = _mm_sub_epi16(src1, dst1);                             \
    /* (srcRGB - dstRGB) * srcA */                                   \
    sub_dst = _mm_mullo_epi16(sub_dst, unpacked_alpha);              \
    /* (srcRGB - dstRGB) * srcA + srcRGB */                          \
    sub_dst = _mm_add_epi16(sub_dst, src1);                          \
    /* (dstRGB << 8) */                                              \
    dst1 = _mm_slli_epi16(dst1, 8);                                  \
    /* ((dstRGB << 8) + (srcRGB - dstRGB) * srcA + srcRGB) */        \
    sub_dst = _mm_add_epi16(sub_dst, dst1);                          \
    /* (((dstRGB << 8) + (srcRGB - dstRGB) * srcA + srcRGB) >> 8) */ \
    sub_dst = _mm_srli_epi16(sub_dst, 8);

void
alphablit_alpha_sse2_argb_no_surf_alpha_opaque_dst(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;
    int srcskip = info->s_skip >> 2;
    int dstskip = info->d_skip >> 2;

    Uint32 *srcp32 = (Uint32 *)info->s_pixels;
    Uint32 *dstp32 = (Uint32 *)info->d_pixels;

    int pxl_excess = width % 4;
    int n_iters_4 = width / 4;

    __m128i src1, dst1, sub_dst, mm_src_alpha;
    __m128i unpacked_alpha, pixels_src, pixels_dst, batch_a_dst;
    __m128i *srcp128, *dstp128;
    __m128i mm_rgb_mask = _mm_set1_epi32(0x00FFFFFF);
    __m128i mm_zero = _mm_setzero_si128();

    while (height--) {
        srcp128 = (__m128i *)srcp32;
        dstp128 = (__m128i *)dstp32;

        if (n_iters_4) {
            LOOP_UNROLLED4(
                {
                    /* ==== load 4 pixels into SSE registers ==== */

                    /*[AR][GB][AR][GB][AR][GB][AR][GB] -> pixels_src*/
                    pixels_src = _mm_loadu_si128(srcp128);

                    /* isolate alpha channels
                     * [A10][00 ][A20][00 ][A30][00 ][A40][00 ] ->
                     * mm_src_alpha*/
                    mm_src_alpha = _mm_andnot_si128(mm_rgb_mask, pixels_src);

                    /* shift right to position alpha channels for manipulation
                     * [0A1][00 ][0A2][00 ][0A3][00 ][0A4][00 ] ->
                     * mm_src_alpha*/
                    mm_src_alpha = _mm_srli_si128(mm_src_alpha, 1);

                    /*[AR][GB][AR][GB][AR][GB][AR][GB] -> pixels_dst*/
                    pixels_dst = _mm_loadu_si128(dstp128);

                    /* ==== BATCH A (the 2 low pixels) ==== */

                    /* shuffle alpha channels to duplicate 16 bit pairs
                     * [00 ][00 ][00 ][00 ][0A3][0A3][0A4][0A4] ->
                     * mm_src_alpha*/
                    unpacked_alpha =
                        _mm_shufflelo_epi16(mm_src_alpha, 0b11110101);

                    /* spread alpha into final config for 16 bit math
                     * [0A3][0A3][0A3][0A3][0A4][0A4][0A4][0A4] ->
                     * unpacked_alpha*/
                    unpacked_alpha =
                        _mm_unpacklo_epi16(unpacked_alpha, unpacked_alpha);

                    /* 0A0R0G0B0A0R0G0B -> src1 */
                    src1 = _mm_unpacklo_epi8(pixels_src, mm_zero);

                    /* 0A0R0G0B0A0R0G0B -> dst1 */
                    dst1 = _mm_unpacklo_epi8(pixels_dst, mm_zero);

                    ARGB_NO_SURF_ALPHA_OPAQUE_DST_PROCEDURE

                    batch_a_dst = sub_dst;

                    /* ==== BATCH B (the 2 high pixels) ==== */

                    /*[00 ][00 ][00 ][00 ][0A1][0A1][0A2][0A2] ->
                     * unpacked_alpha*/
                    unpacked_alpha =
                        _mm_shufflehi_epi16(mm_src_alpha, 0b11110101);

                    /*[0A1][0A1][0A1][0A1][0A2][0A2][0A2][0A2] ->
                     * unpacked_alpha*/
                    unpacked_alpha =
                        _mm_unpackhi_epi16(unpacked_alpha, unpacked_alpha);

                    /*[0A][0R][0G][0B][0A][0R][0G][0B] -> src1*/
                    src1 = _mm_unpackhi_epi8(pixels_src, mm_zero);

                    /*[0A][0R][0G][0B][0A][0R][0G][0B] -> dst1*/
                    dst1 = _mm_unpackhi_epi8(pixels_dst, mm_zero);

                    ARGB_NO_SURF_ALPHA_OPAQUE_DST_PROCEDURE

                    /* ==== combine batches and store ==== */

                    sub_dst = _mm_packus_epi16(batch_a_dst, sub_dst);
                    /* zero out alpha */
                    sub_dst = _mm_and_si128(sub_dst, mm_rgb_mask);
                    _mm_storeu_si128(dstp128, sub_dst);

                    srcp128++;
                    dstp128++;
                },
                n, n_iters_4);
        }

        srcp32 = (Uint32 *)srcp128;
        dstp32 = (Uint32 *)dstp128;

        for (int i = 0; i < pxl_excess; i++) {
            /*[00][00][00][00][00][00][AR][GB] -> src1*/
            src1 = _mm_cvtsi32_si128(*srcp32);

            /*[00][00][00][00][00][00][00][0A] -> mm_src_alpha*/
            mm_src_alpha = _mm_srli_si128(src1, 3);

            /*[00][00][00][00][00][00][0A][0A] -> mm_src_alpha*/
            mm_src_alpha = _mm_unpacklo_epi16(mm_src_alpha, mm_src_alpha);

            /*[00][00][00][00][0A][0A][0A][0A] -> mm_src_alpha*/
            unpacked_alpha = _mm_unpacklo_epi32(mm_src_alpha, mm_src_alpha);

            /*[00][00][00][00][0A][0R][0G][0B] -> src1*/
            src1 = _mm_unpacklo_epi8(src1, mm_zero);

            /*[00][00][00][00][00][00][AR][GB] -> dst1*/
            dst1 = _mm_cvtsi32_si128(*dstp32);

            /*[00][00][00][00][0A][0R][0G][0B] -> dst1*/
            dst1 = _mm_unpacklo_epi8(dst1, mm_zero);

            ARGB_NO_SURF_ALPHA_OPAQUE_DST_PROCEDURE

            /* pack everything back into a pixel */
            sub_dst = _mm_packus_epi16(sub_dst, mm_zero);
            sub_dst = _mm_and_si128(sub_dst, mm_rgb_mask);
            /* reset alpha to 0 */
            *dstp32 = _mm_cvtsi128_si32(sub_dst);

            srcp32++;
            dstp32++;
        }

        srcp32 += srcskip;
        dstp32 += dstskip;
    }
}

void
blit_blend_premultiplied_sse2(SDL_BlitInfo *info)
{
    int n;
    int width = info->width;
    int height = info->height;
    Uint32 *srcp = (Uint32 *)info->s_pixels;
    int srcskip = info->s_skip >> 2;
    Uint32 *dstp = (Uint32 *)info->d_pixels;
    int dstskip = info->d_skip >> 2;
    PG_PixelFormat *srcfmt = info->src;
    Uint32 amask = srcfmt->Amask;
    // Uint64 multmask;
    Uint64 ones;

    // __m128i multmask_128;
    __m128i src1, dst1, sub_dst, mm_alpha, mm_zero, ones_128;

    mm_zero = _mm_setzero_si128();
    // multmask = 0x00FF00FF00FF00FF;  // 0F0F0F0F
    // multmask_128 = _mm_loadl_epi64((const __m128i *)&multmask);
    ones = 0x0001000100010001;
    ones_128 = _mm_loadl_epi64((const __m128i *)&ones);

    while (height--) {
        /* *INDENT-OFF* */
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
                    src1 = _mm_cvtsi32_si128(
                        *srcp); /* src(ARGB) -> src1 (000000000000ARGB) */
                    src1 = _mm_unpacklo_epi8(
                        src1, mm_zero); /* 000000000A0R0G0B -> src1 */

                    dst1 = _mm_cvtsi32_si128(
                        *dstp); /* dst(ARGB) -> dst1 (000000000000ARGB) */
                    dst1 = _mm_unpacklo_epi8(
                        dst1, mm_zero); /* 000000000A0R0G0B -> dst1 */

                    mm_alpha = _mm_cvtsi32_si128(
                        alpha); /* alpha -> mm_alpha (000000000000A000) */
                    mm_alpha = _mm_srli_si128(
                        mm_alpha, 3); /* mm_alpha >> ashift ->
                                         mm_alpha(000000000000000A) */
                    mm_alpha = _mm_unpacklo_epi16(
                        mm_alpha, mm_alpha); /* 0000000000000A0A -> mm_alpha */
                    mm_alpha = _mm_unpacklo_epi32(
                        mm_alpha,
                        mm_alpha); /* 000000000A0A0A0A -> mm_alpha2 */

                    /* pre-multiplied alpha blend */
                    sub_dst = _mm_add_epi16(dst1, ones_128);
                    sub_dst = _mm_mullo_epi16(sub_dst, mm_alpha);
                    sub_dst = _mm_srli_epi16(sub_dst, 8);
                    dst1 = _mm_add_epi16(src1, dst1);
                    dst1 = _mm_sub_epi16(dst1, sub_dst);
                    dst1 = _mm_packus_epi16(dst1, mm_zero);

                    *dstp = _mm_cvtsi128_si32(dst1);
                }
                ++srcp;
                ++dstp;
            },
            n, width);
        /* *INDENT-ON* */
        srcp += srcskip;
        dstp += dstskip;
    }
}

void
premul_surf_color_by_alpha_sse2(SDL_Surface *src, PG_PixelFormat *srcfmt,
                                SDL_Surface *dst)
{
    int n;
    int width = src->w;
    int height = src->h;
    Uint32 *srcp = (Uint32 *)src->pixels;
    int srcskip = src->pitch - width * PG_SURF_BytesPerPixel(src);
    Uint32 *dstp = (Uint32 *)dst->pixels;
    int dstskip = dst->pitch - width * PG_SURF_BytesPerPixel(dst);
    Uint32 amask = srcfmt->Amask;
    Uint64 ones;

    __m128i src1, dst1, rgb_mul_src, mm_alpha, mm_alpha_in, mm_zero, ones_128;

    mm_zero = _mm_setzero_si128();
    ones = 0x0001000100010001;
    ones_128 = _mm_loadl_epi64((const __m128i *)&ones);

    while (height--) {
        /* *INDENT-OFF* */
        LOOP_UNROLLED4(
            {
                Uint32 alpha = *srcp & amask;
                if (alpha == amask) {
                    *dstp = *srcp;
                }
                else {
                    /* extract source pixels */
                    src1 = _mm_cvtsi32_si128(
                        *srcp); /* src(ARGB) -> src1 (000000000000ARGB) */
                    src1 = _mm_unpacklo_epi8(
                        src1, mm_zero); /* 000000000A0R0G0B -> src1 */

                    /* extract source alpha and copy to r, g, b channels */
                    mm_alpha_in = _mm_cvtsi32_si128(
                        alpha); /* alpha -> mm_alpha (000000000000A000) */
                    mm_alpha = _mm_srli_si128(
                        mm_alpha_in, 3); /* mm_alpha >> ashift ->
                                            mm_alpha(000000000000000A) */
                    mm_alpha = _mm_unpacklo_epi16(
                        mm_alpha, mm_alpha); /* 0000000000000A0A -> mm_alpha */
                    mm_alpha = _mm_unpacklo_epi32(
                        mm_alpha,
                        mm_alpha); /* 000000000A0A0A0A -> mm_alpha2 */

                    /* rgb alpha multiply */
                    rgb_mul_src = _mm_add_epi16(src1, ones_128);
                    rgb_mul_src = _mm_mullo_epi16(rgb_mul_src, mm_alpha);
                    rgb_mul_src = _mm_srli_epi16(rgb_mul_src, 8);
                    dst1 = _mm_packus_epi16(rgb_mul_src, mm_zero);
                    dst1 = _mm_max_epu8(mm_alpha_in,
                                        dst1); /* restore original alpha */

                    *dstp = _mm_cvtsi128_si32(dst1);
                }
                ++srcp;
                ++dstp;
            },
            n, width);
        srcp = (Uint32 *)((Uint8 *)srcp + srcskip);
        dstp = (Uint32 *)((Uint8 *)dstp + dstskip);
    }
}

void
blit_blend_rgb_add_sse2(SDL_BlitInfo *info)
{
    SETUP_SSE2_BLITTER
    const __m128i mm128_rgbmask =
        _mm_set1_epi32(~(info->src->Amask | info->dst->Amask));

    RUN_SSE2_BLITTER({
        mm128_src = _mm_and_si128(mm128_src, mm128_rgbmask);
        mm128_dst = _mm_adds_epu8(mm128_dst, mm128_src);
    })
}

void
blit_blend_rgba_add_sse2(SDL_BlitInfo *info)
{
    SETUP_SSE2_BLITTER
    RUN_SSE2_BLITTER({ mm128_dst = _mm_adds_epu8(mm128_dst, mm128_src); })
}

void
blit_blend_rgb_sub_sse2(SDL_BlitInfo *info)
{
    SETUP_SSE2_BLITTER
    const __m128i mm128_rgbmask =
        _mm_set1_epi32(~(info->src->Amask | info->dst->Amask));

    RUN_SSE2_BLITTER({
        mm128_src = _mm_and_si128(mm128_src, mm128_rgbmask);
        mm128_dst = _mm_subs_epu8(mm128_dst, mm128_src);
    })
}

void
blit_blend_rgba_sub_sse2(SDL_BlitInfo *info)
{
    SETUP_SSE2_BLITTER
    RUN_SSE2_BLITTER({ mm128_dst = _mm_subs_epu8(mm128_dst, mm128_src); })
}

void
blit_blend_rgb_mul_sse2(SDL_BlitInfo *info)
{
    SETUP_SSE2_BLITTER
    SETUP_SSE2_16BIT_SHUFFLE_OUT
    const __m128i mm128_amask =
        _mm_set1_epi32(info->src->Amask | info->dst->Amask);
    const __m128i mm128_255 = _mm_set1_epi16(0x00FF);

    RUN_SSE2_BLITTER(mm128_src = _mm_or_si128(mm128_src, mm128_amask);
                     RUN_SSE2_16BIT_SHUFFLE_OUT({
                         shuff_dst = _mm_mullo_epi16(shuff_dst, shuff_src);
                         shuff_dst = _mm_add_epi16(shuff_dst, mm128_255);
                         shuff_dst = _mm_srli_epi16(shuff_dst, 8);
                     }))
}

void
blit_blend_rgba_mul_sse2(SDL_BlitInfo *info)
{
    SETUP_SSE2_BLITTER
    SETUP_SSE2_16BIT_SHUFFLE_OUT
    const __m128i mm128_255 = _mm_set1_epi16(0x00FF);

    RUN_SSE2_BLITTER(RUN_SSE2_16BIT_SHUFFLE_OUT({
        shuff_dst = _mm_mullo_epi16(shuff_dst, shuff_src);
        shuff_dst = _mm_add_epi16(shuff_dst, mm128_255);
        shuff_dst = _mm_srli_epi16(shuff_dst, 8);
    }))
}

void
blit_blend_rgb_min_sse2(SDL_BlitInfo *info)
{
    SETUP_SSE2_BLITTER
    const __m128i mm128_amask =
        _mm_set1_epi32(info->src->Amask | info->dst->Amask);

    RUN_SSE2_BLITTER({
        mm128_src = _mm_or_si128(mm128_src, mm128_amask);
        mm128_dst = _mm_min_epu8(mm128_dst, mm128_src);
    })
}

void
blit_blend_rgba_min_sse2(SDL_BlitInfo *info)
{
    SETUP_SSE2_BLITTER;
    RUN_SSE2_BLITTER({ mm128_dst = _mm_min_epu8(mm128_dst, mm128_src); })
}

void
blit_blend_rgb_max_sse2(SDL_BlitInfo *info)
{
    SETUP_SSE2_BLITTER
    const __m128i mm128_rgbmask =
        _mm_set1_epi32(~(info->src->Amask | info->dst->Amask));

    RUN_SSE2_BLITTER({
        mm128_src = _mm_and_si128(mm128_src, mm128_rgbmask);
        mm128_dst = _mm_max_epu8(mm128_dst, mm128_src);
    })
}

void
blit_blend_rgba_max_sse2(SDL_BlitInfo *info)
{
    SETUP_SSE2_BLITTER
    RUN_SSE2_BLITTER({ mm128_dst = _mm_max_epu8(mm128_dst, mm128_src); })
}
#endif /* __SSE2__ || PG_ENABLE_ARM_NEON*/
