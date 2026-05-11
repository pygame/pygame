# Phase 2A — Platform Guard Audit (2026-05-05)

**Auditor:** Runa Gridweaver Freyjasdottir, executing on Volmarr's behalf during the autonomous-mode cross-project housekeeping pass.
**Branch:** `development`
**Status:** **Audit complete; pgplatform.h proposal drafted; refactor execution deferred to a focused future slice.**

The Phase 2A scope per `TASK_pygame_viking_edition.md` lines 135-138 is three concrete sub-items:

1. Audit all `#ifdef` platform guards — map all platform-specific paths.
2. Create `pgplatform.h` extension with unified abstractions.
3. Platform capability detection at runtime vs compile time.

This document delivers item 1 (the audit) and a draft of item 2 (the pgplatform.h proposal). Item 3 (runtime vs compile-time detection) is a design decision that depends on item 2's shape.

---

## Audit method

Surveyed every `#ifdef` and `#if defined` directive across `src_c/*.c` and `src_c/*.h`. Top files by guard density:

| File | Guards | Notes |
|---|---|---|
| `_camera.c` | 14 | macOS / Linux / Windows camera backends |
| `simd_blitters_avx2.c` | 13 | x86 SIMD intrinsics (HAVE_IMMINTRIN_H, __AVX2__) |
| `rwobject.c` | 13 | macOS path resolution + Windows path quirks |
| `imageext.c` | 12 | optional image-format backends |
| `display.c` | 12 | per-OS video driver hints + x11/wayland nuances |
| `transform.c` | 10 | SSE2/MMX/NEON SIMD paths |
| `camera.h` | 8 | platform backend selector |
| `alphablit.c` | 7 | SIMD blitters |
| `base.c` | 6 | thread/timer init flags |
| `math.c` | 5 | epsilon precision per-platform |

**Total `#ifdef`/`#if defined` directives across src_c/: ~150 across ~40 files.**

---

## Guard taxonomy

The directives cluster into 5 distinct concerns. Mixing them in the same line (the most common pattern today) makes the code hard to read; the pgplatform.h proposal below addresses this.

### 1. OS guards (operating system family)

The most-prevalent class. Used to gate platform-specific syscalls / APIs / driver hints.

| Guard | Meaning | Sample sites |
|---|---|---|
| `_WIN32`, `MS_WIN32`, `__WIN32__`, `_WIN64` | Windows (any) | base.c, display.c, rwobject.c |
| `__APPLE__` (often + `darwin`) | macOS / iOS | rwobject.c, _camera.c |
| `__linux__` | Linux | _camera.c |
| `__ANDROID__` (or via `SDL_VIDEO_DRIVER_ANDROID`) | Android | display.c |
| `__EMSCRIPTEN__` | WebAssembly target | event.c, base.c |
| `__FreeBSD__`, `__OpenBSD__`, `__NetBSD__` | BSD family | _camera.c |
| `__unix__` | Generic POSIX | base.c |
| `macintosh` | classic Mac (legacy) | rwobject.c |

**Issue:** Windows alone has 4 different macros in active use (`_WIN32`, `MS_WIN32`, `__WIN32__`, `_WIN64`). The pgplatform.h proposal collapses these to one `PG_PLATFORM_WINDOWS` macro.

### 2. CPU architecture guards

| Guard | Meaning | Sample sites |
|---|---|---|
| `__x86_64__`, `_M_X64`, `ENV64BIT` | x86-64 | simd_blitters_avx2.c, alphablit.c |
| `__i386__` | x86 32-bit | base.c |
| `__PPC64__` | PowerPC | base.c |
| (none for arm64) | — | (gap! see below) |

**Issue:** No explicit `__aarch64__` / `_M_ARM64` checks despite `PG_ENABLE_ARM_NEON` being a build-config flag. The build assumes the CFLAGS provider sets `PG_ENABLE_ARM_NEON` correctly — fragile. pgplatform.h should derive `PG_PLATFORM_ARM64` from compiler-defined macros.

### 3. SIMD / instruction-set capability guards

| Guard | Meaning | Sample sites |
|---|---|---|
| `__SSE2__` | SSE2 enabled in compile | alphablit.c, transform.c, simd_blitters_*.c |
| `__AVX2__` + `HAVE_IMMINTRIN_H` | AVX2 + Intel intrinsics header | simd_blitters_avx2.c |
| `__MMX__` | MMX | transform.c |
| `PG_ENABLE_ARM_NEON` | ARM NEON (build-config flag) | alphablit.c, transform.c |
| `SCALE_MMX_SUPPORT` | scale.c MMX path | scale.c |
| `_NO_MMX_FOR_X86_64` | force-disable MMX on x64 | transform.c |
| `SDL_DISABLE_IMMINTRIN_H` | force-disable intrinsics | simd_blitters_avx2.c |

**Issue:** SIMD detection is a mess of intersecting build-config flags + compiler-defined macros + SDL feature flags. A unified `PG_HAVE_SIMD_AVX2` etc. would centralize the detection.

### 4. Compiler / runtime guards

| Guard | Meaning | Sample sites |
|---|---|---|
| `__GNUC__` | GCC or compatible | base.c |
| `_MSC_VER` | Microsoft Visual C++ | base.c, transform.c |
| `PYPY_VERSION` | PyPy interpreter | rwobject.c |
| `GRAALVM_PYTHON` | GraalPython interpreter | rwobject.c |
| `WITH_THREAD` | Python thread support | base.c, _sdl2/* |
| `BUILD_STATIC` | static build mode | _sdl2/__init__.c |
| `NO_PYGAME_C_API` | C-API disabled (cython?) | _sdl2/__init__.c |

These are mostly fine in their current shape — they capture genuinely independent concerns.

### 5. SDL feature/driver guards

Generally inherited from SDL2's own headers. Used to gate code paths that require a specific SDL backend:

| Guard | Used in |
|---|---|
| `SDL_VIDEO_DRIVER_COCOA` | display.c (macOS) |
| `SDL_VIDEO_DRIVER_UIKIT` | display.c (iOS) |
| `SDL_VIDEO_DRIVER_ANDROID` | display.c |
| `SDL_VIDEO_DRIVER_DIRECTFB` | display.c |

These are correct as-is — the SDL convention IS to gate on these. pgplatform.h should NOT shadow them.

---

## Notable findings

### Finding 1 — Windows macro inconsistency

Same code path uses different Windows macros in different places. There's no semantic difference between `_WIN32`, `MS_WIN32`, and `__WIN32__` for current pygame usage — they all mean "compiling on Windows." The historical drift makes audits harder.

**Recommendation:** Phase 2A introduces `PG_PLATFORM_WINDOWS` as the canonical macro; existing `_WIN32` etc. checks remain (additive) but new code uses `PG_PLATFORM_WINDOWS`. A future cleanup slice can rewrite the legacy spellings module-by-module.

### Finding 2 — No explicit arm64 / aarch64 detection

Pygame has `PG_ENABLE_ARM_NEON` as a build-time flag but no compile-time detection of arm64. On Apple Silicon (`__arm64__` / `__aarch64__`) and Linux ARM64, the build relies on the CFLAGS provider correctly setting `PG_ENABLE_ARM_NEON`.

**Recommendation:** pgplatform.h should derive `PG_PLATFORM_ARM64` automatically from compiler macros. Then SIMD selector logic becomes cleaner: `#if PG_ARCH_ARM64 && PG_ENABLE_ARM_NEON` rather than today's nested patterns.

### Finding 3 — The `darwin` macro is a legacy alias

In `rwobject.c`: `#if defined(__APPLE__) && defined(darwin)`. `darwin` is not a standard pre-defined macro on macOS — it was a quirk of the Pyrex/early-Cython era. Modern compilers don't define it; this branch is effectively unreachable on current builds.

**Recommendation:** Mark for cleanup in a follow-up slice.

### Finding 4 — `macintosh` macro is fully legacy

`rwobject.c` has `#if defined(macintosh)` for classic Mac OS support. Pygame doesn't support classic Mac OS in any released build for the past ~15 years.

**Recommendation:** Remove in a follow-up slice (subtractive change requiring explicit operator approval per the project's additive-only convention; flag for human review).

### Finding 5 — `BUILD_STATIC` + `NO_PYGAME_C_API` cluster

`_sdl2/__init__.c` has `#if defined(BUILD_STATIC) && defined(NO_PYGAME_C_API)`. This guards the static-build-without-C-API path used by some embedded scenarios. It works but the intent is opaque without context.

**Recommendation:** Document via the new `PG_BUILD_EMBEDDED` alias for forward-readable code.

---

## pgplatform.h proposal

A new header at `src_c/include/pgplatform.h` that defines the canonical macros pygame Viking Edition uses going forward. Existing compiler/SDL/build-config macros remain available; the new macros are **additive aliases** that capture the project's intent.

The header is delivered as `src_c/include/pgplatform.h` in the same commit as this audit document.

### Adoption pattern

Existing code is NOT mass-rewritten. Modules opt in at their leisure:

```c
/* Before */
#if defined(_WIN32) || defined(__WIN32__) || defined(MS_WIN32)
   /* Windows path */
#endif

/* After (in modules that include pgplatform.h) */
#include "include/pgplatform.h"
#if PG_PLATFORM_WINDOWS
   /* Windows path */
#endif
```

The legacy macros remain valid (they're never undefined) so a partial-adoption tree compiles cleanly. A future slice can sweep one module at a time toward the new spelling.

---

## Item 3 — Runtime vs compile-time capability detection

Phase 2A's third sub-item asks for "Platform capability detection at runtime vs compile time." pgplatform.h above is **compile-time** detection. The runtime question is separate and bigger:

- Should pygame ship one binary with all SIMD paths and pick at runtime (CPU dispatch)?
- Or one binary per CPU class (current model — build-time detection only)?

Modern Linux distros lean toward CPU dispatch (e.g. glibc's IFUNC mechanism) so a single .so works everywhere; macOS bundles ship per-arch via the universal binary mechanism; Windows historically picks at install time.

**Recommendation:** Defer this decision to Phase 2B (Platform Parity Audit). The choice depends on which platforms pygame VE prioritizes shipping pre-built wheels for. If pygame VE goes the manylinux2014 + Apple-universal-binary route, build-time detection is the right pattern. If it goes single-source-installs-everywhere, runtime CPU dispatch is needed.

This audit document captures the requirement; Phase 2B's parity work picks the strategy.

---

## Closeout

Phase 2A's audit (item 1) is done; the pgplatform.h header (item 2) lands as `src_c/include/pgplatform.h` in the same commit. Item 3 (runtime vs compile-time) is deferred to Phase 2B where the prioritization question naturally surfaces.

**Next step:** Phase 2B — per-platform parity audit. Use pgplatform.h's macros as the framework for documenting per-platform feature presence/absence in `docs/PLATFORM_MATRIX.md`.

The 5 minor cleanup recommendations (Finding 4 `macintosh` removal etc.) are flagged for human review since they're subtractive — the project's additive-only convention requires explicit Volmarr approval before they land.
