/* platform/compiler adjustments (internal) */
#ifndef PG_PLATFORM_INTERNAL_H
#define PG_PLATFORM_INTERNAL_H

/* This must be before all else */
#if defined(__SYMBIAN32__) && defined(OPENC)
#include <sys/types.h>
#if defined(__WINS__)
void *
_alloca(size_t size);
#define alloca _alloca
#endif /* __WINS__ */
#endif /* defined(__SYMBIAN32__) && defined(OPENC) */

#include "../include/pgplatform.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef ABS
#define ABS(a) (((a) < 0) ? -(a) : (a))
#endif

#if defined(macintosh) && defined(__MWERKS__) || defined(__SYMBIAN32__)
#define PYGAME_EXPORT __declspec(export)
#else
#define PYGAME_EXPORT
#endif

/* warnings */
#define PG_STRINGIZE_HELPER(x) #x
#define PG_STRINGIZE(x) PG_STRINGIZE_HELPER(x)
#define PG_WARN(desc) message(__FILE__ "(" PG_STRINGIZE(__LINE__) "): WARNING: " #desc)

#endif /* ~PG_PLATFORM_INTERNAL_H */
