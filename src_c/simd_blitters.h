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
blit_blend_rgba_mul_sse2(SDL_BlitInfo *info);
void
blit_blend_rgb_mul_sse2(SDL_BlitInfo *info);
#endif /* (defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)) */

void
blit_blend_rgba_mul_avx2(SDL_BlitInfo *info);
void
blit_blend_rgb_mul_avx2(SDL_BlitInfo *info);
