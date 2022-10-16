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

int
pg_has_avx2();
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
