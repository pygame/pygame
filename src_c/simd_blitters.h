#define NO_PYGAME_C_API
#include "_surface.h"
#include "_blit_info.h"

#if !defined(PG_ENABLE_ARM_NEON) && defined(__aarch64__)
// arm64 has neon optimisations enabled by default, even when fpu=neon is not
// passed
#define PG_ENABLE_ARM_NEON 1
#endif

#if (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON))
void
alphablit_alpha_sse2_argb_surf_alpha(SDL_BlitInfo *info);
void
alphablit_alpha_sse2_argb_no_surf_alpha(SDL_BlitInfo *info);
void
alphablit_alpha_sse2_argb_no_surf_alpha_opaque_dst(SDL_BlitInfo *info);
void
blit_blend_rgba_mul_sse2(SDL_BlitInfo *info);
void
blit_blend_rgb_mul_sse2(SDL_BlitInfo *info);
void
blit_blend_rgba_add_sse2(SDL_BlitInfo *info);
void
blit_blend_rgb_add_sse2(SDL_BlitInfo *info);
void
blit_blend_rgba_sub_sse2(SDL_BlitInfo *info);
void
blit_blend_rgb_sub_sse2(SDL_BlitInfo *info);
void
blit_blend_rgba_max_sse2(SDL_BlitInfo *info);
void
blit_blend_rgb_max_sse2(SDL_BlitInfo *info);
void
blit_blend_rgba_min_sse2(SDL_BlitInfo *info);
void
blit_blend_rgb_min_sse2(SDL_BlitInfo *info);
void
blit_blend_premultiplied_sse2(SDL_BlitInfo *info);
#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */

/* Deliberately putting these outside of the preprocessor guards as I want to
   move to a system of trusting the runtime checks to head to the right
   function and having a fallback function there if pygame is not compiled
   with the right stuff (this is the strategy used for AVX2 right now.
   Potentially I might want to shift both these into a slightly different
   file as they are not exactly blits (though v. similar) - or I could rename
   the SIMD trilogy of files to replace the word blit with something more
   generic like surface_ops*/

void
premul_surf_color_by_alpha_non_simd(SDL_Surface *src,
                                    PG_PixelFormat *src_format,
                                    SDL_Palette *src_palette, SDL_Surface *dst,
                                    PG_PixelFormat *dst_format,
                                    SDL_Palette *dst_palette);
void
premul_surf_color_by_alpha_sse2(SDL_Surface *src, PG_PixelFormat *srcfmt,
                                SDL_Surface *dst);

void
alphablit_alpha_avx2_argb_no_surf_alpha_opaque_dst(SDL_BlitInfo *info);
void
alphablit_alpha_avx2_argb_no_surf_alpha(SDL_BlitInfo *info);
void
alphablit_alpha_avx2_argb_surf_alpha(SDL_BlitInfo *info);
void
blit_blend_rgba_mul_avx2(SDL_BlitInfo *info);
void
blit_blend_rgb_mul_avx2(SDL_BlitInfo *info);
void
blit_blend_rgba_add_avx2(SDL_BlitInfo *info);
void
blit_blend_rgb_add_avx2(SDL_BlitInfo *info);
void
blit_blend_rgba_sub_avx2(SDL_BlitInfo *info);
void
blit_blend_rgb_sub_avx2(SDL_BlitInfo *info);
void
blit_blend_rgba_max_avx2(SDL_BlitInfo *info);
void
blit_blend_rgb_max_avx2(SDL_BlitInfo *info);
void
blit_blend_rgba_min_avx2(SDL_BlitInfo *info);
void
blit_blend_rgb_min_avx2(SDL_BlitInfo *info);
void
blit_blend_premultiplied_avx2(SDL_BlitInfo *info);
void
premul_surf_color_by_alpha_avx2(SDL_Surface *src, PG_PixelFormat *src_format,
                                SDL_Surface *dst);
