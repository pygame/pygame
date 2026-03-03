#if !defined(PGCOMPAT_H)
#define PGCOMPAT_H

#include <Python.h>

/* In CPython, Py_Exit finalises the python interpreter before calling C exit()
 * This does not exist on PyPy, so use exit() directly here */
#ifdef PYPY_VERSION
#define PG_EXIT(n) exit(n)
#else
#define PG_EXIT(n) Py_Exit(n)
#endif

/* define common types where SDL is not included */
#ifndef SDL_VERSION_ATLEAST
#ifdef _MSC_VER
typedef unsigned __int8 uint8_t;
typedef unsigned __int32 uint32_t;
#else
#include <stdint.h>
#endif
typedef uint32_t Uint32;
typedef uint8_t Uint8;
#endif /* no SDL */

/* SDL_VERSION_ATLEAST is in every supported SDL version, but the code gets a
 * warning without this check here, which is very weird. */
#ifdef SDL_VERSION_ATLEAST

// SDL does not provide endian independent names for 32 bit formats without
// alpha channels the way they do for ones with alpha channels.
// E.g. SDL_PIXELFORMAT_RGBA32. This macro allows us the convenience of the
// endian independent name.

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define PG_PIXELFORMAT_RGBX32 SDL_PIXELFORMAT_XBGR8888
#else
#define PG_PIXELFORMAT_RGBX32 SDL_PIXELFORMAT_RGBX8888
#endif

#if SDL_VERSION_ATLEAST(2, 0, 18)
#define PG_GetTicks SDL_GetTicks64
#else
#define PG_GetTicks SDL_GetTicks
#endif /* SDL_VERSION_ATLEAST(2, 0, 18) */

#endif /* defined(SDL_VERSION_ATLEAST) */

#ifndef SDL_MOUSEWHEEL_FLIPPED
#define NO_SDL_MOUSEWHEEL_FLIPPED
#endif

#endif /* ~defined(PGCOMPAT_H) */
