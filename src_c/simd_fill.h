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
#if __x86_64__ || __ppc64__ || __aarch64__
#define ENV64BIT
#endif
#endif

#if defined(__SSE2__)
#define PG_ENABLE_SSE_NEON 1
#elif PG_ENABLE_ARM_NEON
#define PG_ENABLE_SSE_NEON 1
#else
#define PG_ENABLE_SSE_NEON 0
#endif

int
_pg_has_avx2();

/* This returns True if either SSE2 or NEON is present at runtime.
 * Relevant because they use the same codepaths. Only the relevant runtime
 * SDL cpu feature check is compiled in.*/
int
_pg_HasSSE_NEON();

// AVX2 functions
int
surface_fill_blend_add_avx2(SDL_Surface *surface,
                            PG_PixelFormat *surface_format, SDL_Rect *rect,
                            Uint32 color);
int
surface_fill_blend_rgba_add_avx2(SDL_Surface *surface,
                                 PG_PixelFormat *surface_format,
                                 SDL_Rect *rect, Uint32 color);

int
surface_fill_blend_sub_avx2(SDL_Surface *surface,
                            PG_PixelFormat *surface_format, SDL_Rect *rect,
                            Uint32 color);
int
surface_fill_blend_rgba_sub_avx2(SDL_Surface *surface,
                                 PG_PixelFormat *surface_format,
                                 SDL_Rect *rect, Uint32 color);
int
surface_fill_blend_mult_avx2(SDL_Surface *surface,
                             PG_PixelFormat *surface_format, SDL_Rect *rect,
                             Uint32 color);
int
surface_fill_blend_rgba_mult_avx2(SDL_Surface *surface,
                                  PG_PixelFormat *surface_format,
                                  SDL_Rect *rect, Uint32 color);
int
surface_fill_blend_min_avx2(SDL_Surface *surface,
                            PG_PixelFormat *surface_format, SDL_Rect *rect,
                            Uint32 color);
int
surface_fill_blend_rgba_min_avx2(SDL_Surface *surface,
                                 PG_PixelFormat *surface_format,
                                 SDL_Rect *rect, Uint32 color);
int
surface_fill_blend_max_avx2(SDL_Surface *surface,
                            PG_PixelFormat *surface_format, SDL_Rect *rect,
                            Uint32 color);
int
surface_fill_blend_rgba_max_avx2(SDL_Surface *surface,
                                 PG_PixelFormat *surface_format,
                                 SDL_Rect *rect, Uint32 color);
// SSE2 functions
int
surface_fill_blend_add_sse2(SDL_Surface *surface,
                            PG_PixelFormat *surface_format, SDL_Rect *rect,
                            Uint32 color);
int
surface_fill_blend_rgba_add_sse2(SDL_Surface *surface,
                                 PG_PixelFormat *surface_format,
                                 SDL_Rect *rect, Uint32 color);
int
surface_fill_blend_sub_sse2(SDL_Surface *surface,
                            PG_PixelFormat *surface_format, SDL_Rect *rect,
                            Uint32 color);
int
surface_fill_blend_rgba_sub_sse2(SDL_Surface *surface,
                                 PG_PixelFormat *surface_format,
                                 SDL_Rect *rect, Uint32 color);
int
surface_fill_blend_mult_sse2(SDL_Surface *surface,
                             PG_PixelFormat *surface_format, SDL_Rect *rect,
                             Uint32 color);
int
surface_fill_blend_rgba_mult_sse2(SDL_Surface *surface,
                                  PG_PixelFormat *surface_format,
                                  SDL_Rect *rect, Uint32 color);
int
surface_fill_blend_min_sse2(SDL_Surface *surface,
                            PG_PixelFormat *surface_format, SDL_Rect *rect,
                            Uint32 color);
int
surface_fill_blend_rgba_min_sse2(SDL_Surface *surface,
                                 PG_PixelFormat *surface_format,
                                 SDL_Rect *rect, Uint32 color);
int
surface_fill_blend_max_sse2(SDL_Surface *surface,
                            PG_PixelFormat *surface_format, SDL_Rect *rect,
                            Uint32 color);
int
surface_fill_blend_rgba_max_sse2(SDL_Surface *surface,
                                 PG_PixelFormat *surface_format,
                                 SDL_Rect *rect, Uint32 color);
