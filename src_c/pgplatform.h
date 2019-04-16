/* platform/compiler adjustments */
#ifndef PG_PLATFORM_H
#define PG_PLATFORM_H

#if defined(HAVE_SNPRINTF) /* defined in python.h (pyerrors.h) and SDL.h \
                              (SDL_config.h) */
#undef HAVE_SNPRINTF       /* remove GCC redefine warning */
#endif

/* This must be before all else */
#if defined(__SYMBIAN32__) && defined(OPENC)
#include <sys/types.h>

#if defined(__WINS__)
void *
_alloca(size_t size);
#define alloca _alloca
#endif /* __WINS__ */
#endif /* HAVE_SNPRINTF */

#ifndef PG_INLINE
#if defined(__clang__)
#define PG_INLINE __inline__ __attribute__((__unused__))
#elif defined(__GNUC__)
#define PG_INLINE __inline__
#elif defined(_MSC_VER)
#define PG_INLINE __inline
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#define PG_INLINE inline
#else
#define PG_INLINE
#endif
#endif /* ~PG_INLINE */

#if defined(macintosh) && defined(__MWERKS__) || defined(__SYMBIAN32__)
#define PYGAME_EXPORT __declspec(export)
#else
#define PYGAME_EXPORT
#endif

/* This is unconditionally defined in Python.h */
#if defined(_POSIX_C_SOURCE)
#undef _POSIX_C_SOURCE
#endif

/* No signal() */
#if defined(__SYMBIAN32__) && defined(HAVE_SIGNAL_H)
#undef HAVE_SIGNAL_H
#endif

#if defined(HAVE_SNPRINTF)
#undef HAVE_SNPRINTF
#endif

/*Python gives us MS_WIN32, SDL needs just WIN32*/
#ifdef MS_WIN32
#ifndef WIN32
#define WIN32
#endif
#endif

/* min/max */
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef ABS
#define ABS(a) (((a) < 0) ? -(a) : (a))
#endif

/* warnings */
#define PG_STRINGIZE_HELPER(x) #x
#define PG_STRINGIZE(x) PG_STRINGIZE_HELPER(x)
#define PG_WARN(desc) message(__FILE__ "(" PG_STRINGIZE(__LINE__) "): WARNING: " #desc)

#endif /* ~PG_PLATFORM_H */
