#define NO_PYGAME_C_API
#ifndef SIMD_SHARED_H
#define SIMD_SHARED_H

#include "_surface.h"

int
pg_sse2_at_runtime_but_uncompiled();
int
pg_neon_at_runtime_but_uncompiled();
int
pg_avx2_at_runtime_but_uncompiled();
int
pg_has_avx2();

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

/* This defines PG_ENABLE_SSE_NEON as True if either SSE or NEON is available
 * at compile time. Since we do compile time translation of SSE2->NEON, they
 * have the same code paths, so this reduces code duplication of those paths.
 */
#if defined(__SSE2__)
#define PG_ENABLE_SSE_NEON 1
#elif PG_ENABLE_ARM_NEON
#define PG_ENABLE_SSE_NEON 1
#else
#define PG_ENABLE_SSE_NEON 0
#endif

/* This returns True if either SSE2 or NEON is present at runtime.
 * Relevant because they use the same codepaths. Only the relevant runtime
 * SDL cpu feature check is compiled in.*/
int
pg_HasSSE_NEON()
{
#if defined(__SSE2__)
    return SDL_HasSSE2();
#elif PG_ENABLE_ARM_NEON
    return SDL_HasNEON();
#else
    return 0;
#endif
}

#endif  // SIMD_SHARED_H
