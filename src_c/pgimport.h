#ifndef PGIMPORT_INTERNAL_H
#define PGIMPORT_INTERNAL_H

#include "include/pgimport.h"

#if PG_HAVE_CAPSULE
#define encapsulate_api(array, module) \
    PyCapsule_New(array, PG_CAPSULE_NAME(module), NULL)
#else /* ~PG_HAVE_CAPSULE */
#define encapsulate_api(array, module) \
    PyCObject_FromVoidPtr(array, NULL)
#endif /* ~PG_HAVE_CAPSULE */

#if 0
#ifndef NO_PYGAME_C_API
#define _PG_ASSERT_ROOT_SIZE(MODULE, api_root)                     \
        PG_STATIC_ASSERT(                                          \
            PYGAMEAPI_##MODULE##_FIRSTSLOT +                       \
            PYGAMEAPI_##MODULE##_NUMSLOTS                          \
            <= PYGAMEAPI_NUM_SLOTS(api_root), api_root_too_small)
#else /* NO_PYGAME_C_API */
#define _PG_ASSERT_ROOT_SIZE(MODULE, api_root)
#endif /* NO_PYGAME_C_API */
#endif

#define _PG_ASSERT_ROOT_SIZE(MODULE, api_root)

#define _encapsulate_api_safe(retptr, array,                       \
                              module, MODULE,                      \
                              api_root)                            \
    do {                                                           \
        PG_STATIC_ASSERT( sizeof(array) / sizeof(void*) ==         \
                          PYGAMEAPI_##MODULE##_NUMSLOTS,           \
                          encapsulate_wrong_size );                \
        _PG_ASSERT_ROOT_SIZE(MODULE, api_root);                    \
        *retptr = encapsulate_api(array, #module);                 \
    } while(0)

#define encapsulate_api_safe(retptr, array, module, MODULE) \
    _encapsulate_api_safe(retptr, array, module, MODULE, PyGAME_C_API)

#endif /* ~PGIMPORT_INTERNAL_H */
